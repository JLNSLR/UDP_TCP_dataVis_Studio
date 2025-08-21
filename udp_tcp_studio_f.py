# udp_studio_dock.py
import os
import sys
import time
import math
import socket
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Dict, List, Any, Callable, Tuple

import numpy as np
import yaml
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
import pyqtgraph as pg
from pyqtgraph.dockarea import DockArea, Dock

pg.setConfigOptions(useOpenGL=True, antialias=True)  # anti-aliasing hurts perf


# ------------------ Tiny helpers (NO regex) ------------------

def _split_prefix_num(s: str) -> Tuple[str, int]:
    s = str(s).strip()
    i = len(s)
    while i > 0 and s[i - 1].isdigit():
        i -= 1
    prefix = s[:i]
    if i == len(s):
        raise ValueError(f"Missing numeric suffix in '{s}'")
    return prefix, int(s[i:])


def expand_range_token(token: str) -> List[str]:
    token = str(token).strip()
    if ".." not in token:
        return [token]
    left, right = token.split("..", 1)
    lp, ln = _split_prefix_num(left)
    right = right.strip()
    if right and right[0].isdigit():
        rp, rn = "", int(right)
    else:
        rp, rn = _split_prefix_num(right)
    if lp and rp and lp != rp:
        raise ValueError(f"Range prefixes differ in '{token}': '{lp}' vs '{rp}'")
    prefix = lp or rp
    step = 1 if rn >= ln else -1
    return [f"{prefix}{i}" for i in range(ln, rn + step, step)]


def expand_list(tokens: List[str]) -> List[str]:
    out = []
    for t in tokens:
        out.extend(expand_range_token(t))
    return out


def parse_floats(payload: bytes, prefer_binary=True):
    if prefer_binary and len(payload) % 4 == 0 and len(payload) > 0:
        try:
            return np.frombuffer(payload, dtype="<f4").copy()
        except Exception:
            pass
    try:
        text = payload.decode("utf-8", "ignore").strip()
        if not text:
            return None
        parts = [p for p in text.replace(",", " ").split() if p]
        return np.array([float(p) for p in parts], dtype=np.float32)
    except Exception:
        return None


# ------------------ Config model ------------------

@dataclass
class ChannelSpec:
    name: str
    index: int
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    color: Any = None
    width: float = 1.5


class Config:
    def __init__(self, path="config.yaml"):
        self.path = path
        self.reload()

    def reload(self):
        with open(self.path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        self.app = cfg.get("app", {})
        self.data_sources = cfg.get("data_sources", [])
        self.channels = cfg.get("channels", [])
        self.derived = cfg.get("derived", [])
        self.plots = cfg.get("plots", [])
        self.layout = cfg.get("layout", {})
        self.control_groups = cfg.get("control_groups", [])
        self.controls = cfg.get("controls", [])
        self.senders = cfg.get("senders", [])
        self.logging = cfg.get("logging", {"enabled": False})


# ------------------ UDP / TCP sources ------------------

class UdpSource(QtCore.QObject):
    vector = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.sock = QUdpSocket(self)
        ip = spec.get("bind_ip", "0.0.0.0")
        port = int(spec["port"])
        addr = QHostAddress.AnyIPv4 if ip in ("0.0.0.0", "", None) else QHostAddress(ip)
        if not self.sock.bind(addr, port):
            raise RuntimeError(f"Could not bind UDP {ip}:{port}")
        self.sock.readyRead.connect(self._on_ready)

    @QtCore.pyqtSlot()
    def _on_ready(self):
        fmt = self.spec.get("format", "float32_le")
        prefer_binary = (fmt == "float32_le")
        while self.sock.hasPendingDatagrams():
            size = self.sock.pendingDatagramSize()
            data, host, port = self.sock.readDatagram(size)
            vec = parse_floats(data, prefer_binary=prefer_binary)
            if vec is not None:
                self.vector.emit(vec)


class TcpClientSource(QtCore.QObject):
    vector = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        from PyQt5.QtNetwork import QTcpSocket
        self.spec = spec
        self.socket = QTcpSocket(self)

        self.host   = spec.get("host", "127.0.0.1")
        self.port   = int(spec.get("port", 7000))
        self.format = spec.get("format", "float32_le_fixed")
        self.length = int(spec.get("length", 0))
        self.delim  = spec.get("line_delimiter", "\n").encode("utf-8")
        self._buf = bytearray()

        self._recon = QtCore.QTimer(self)
        self._recon.setSingleShot(True)
        self._recon.timeout.connect(self._connect)

        self.socket.readyRead.connect(self._on_ready)
        try:
            self.socket.errorOccurred.connect(lambda *_: self._schedule_reconnect())
        except Exception:
            try:
                self.socket.error.connect(lambda *_: self._schedule_reconnect())
            except Exception:
                pass
        self.socket.disconnected.connect(self._schedule_reconnect)

        self._connect()

    def _connect(self):
        self._buf.clear()
        self.socket.abort()
        self.socket.connectToHost(self.host, self.port)

    def _schedule_reconnect(self):
        if not self._recon.isActive():
            self._recon.start(500)

    @QtCore.pyqtSlot()
    def _on_ready(self):
        data = bytes(self.socket.readAll())
        if not data:
            return
        self._buf += data

        if self.format == "float32_le_fixed":
            if self.length <= 0:
                return
            frame_bytes = self.length * 4
            while len(self._buf) >= frame_bytes:
                chunk = self._buf[:frame_bytes]
                del self._buf[:frame_bytes]
                vec = np.frombuffer(chunk, dtype="<f4").copy()
                self.vector.emit(vec)

        elif self.format == "ascii_line":
            while True:
                idx = self._buf.find(self.delim)
                if idx < 0:
                    break
                line = self._buf[:idx]
                del self._buf[:idx + len(self.delim)]
                s = line.decode("utf-8", "ignore").strip()
                if not s:
                    continue
                parts = [p for p in s.replace(",", " ").split() if p]
                try:
                    vec = np.array([float(p) for p in parts], dtype=np.float32)
                    self.vector.emit(vec)
                except Exception:
                    pass


class TcpServerSource(QtCore.QObject):
    vector = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        from PyQt5.QtNetwork import QTcpServer
        self.spec = spec
        self.server = QTcpServer(self)

        ip   = spec.get("bind_ip", "0.0.0.0")
        port = int(spec.get("port", 7000))
        addr = QHostAddress.AnyIPv4 if ip in ("0.0.0.0", "", None) else QHostAddress(ip)
        if not self.server.listen(addr, port):
            raise RuntimeError(f"Could not bind TCP {ip}:{port}")

        self.format = spec.get("format", "float32_le_fixed")
        self.length = int(spec.get("length", 0))
        self.delim  = spec.get("line_delimiter", "\n").encode("utf-8")

        self.client = None
        self._buf = bytearray()

        self.server.newConnection.connect(self._accept)

    def _accept(self):
        sock = self.server.nextPendingConnection()
        if self.client:
            try:
                self.client.readyRead.disconnect()
                self.client.disconnected.disconnect()
            except Exception:
                pass
            self.client.close()
        self.client = sock
        self.client.readyRead.connect(self._on_ready)
        self.client.disconnected.connect(self._on_disc)
        self._buf.clear()

    def _on_disc(self):
        self.client = None
        self._buf.clear()

    @QtCore.pyqtSlot()
    def _on_ready(self):
        if not self.client:
            return
        data = bytes(self.client.readAll())
        if not data:
            return
        self._buf += data

        if self.format == "float32_le_fixed":
            if self.length <= 0:
                return
            frame_bytes = self.length * 4
            while len(self._buf) >= frame_bytes:
                chunk = self._buf[:frame_bytes]
                del self._buf[:frame_bytes]
                vec = np.frombuffer(chunk, dtype="<f4").copy()
                self.vector.emit(vec)

        elif self.format == "ascii_line":
            while True:
                idx = self._buf.find(self.delim)
                if idx < 0:
                    break
                line = self._buf[:idx]
                del self._buf[:idx + len(self.delim)]
                s = line.decode("utf-8", "ignore").strip()
                if not s:
                    continue
                parts = [p for p in s.replace(",", " ").split() if p]
                try:
                    vec = np.array([float(p) for p in parts], dtype=np.float32)
                    self.vector.emit(vec)
                except Exception:
                    pass


# ------------------ Sender ------------------

class UdpSender(QtCore.QObject):
    tx_rate = QtCore.pyqtSignal(float)

    def __init__(self, spec: dict, value_getter: Callable[[str], float], parent=None):
        super().__init__(parent)
        self.spec = spec
        self.get = value_getter
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.addr = (spec["remote_ip"], int(spec["remote_port"]))
        self.format = spec.get("format", "float32_le")
        self.template = spec.get("vector", {}).get("template", [])
        self._t_last = time.time()
        self._sent = 0

    def start(self):
        hz = float(self.spec.get("rate_hz", 100.0))
        self.timer.start(max(1, int(1000.0 / hz)))
        self._t_last = time.time()
        self._sent = 0

    def stop(self):
        self.timer.stop()

    def send_once(self):
        self._send()

    def _tick(self):
        self._send()
        self._sent += 1
        now = time.time()
        if now - self._t_last >= 0.5:
            self.tx_rate.emit(self._sent / (now - self._t_last))
            self._sent = 0
            self._t_last = now

    def _resolve_item(self, item):
        if isinstance(item, (int, float)):
            return float(item)
        if isinstance(item, str) and item.startswith("$"):
            v = self.get(item[1:])
            return float(v) if v is not None else 0.0
        try:
            return float(item)
        except Exception:
            return 0.0

    def _send(self):
        vec = [self._resolve_item(x) for x in self.template]
        try:
            if self.format == "float32_le":
                payload = np.asarray(vec, dtype=np.float32).tobytes()
            else:
                payload = (" ".join(f"{x:.6g}" for x in vec)).encode("utf-8")
            self.sock.sendto(payload, self.addr)
        except Exception:
            pass


# ------------------ Main app ------------------

class Studio(QtWidgets.QMainWindow):
    def __init__(self, cfg_path="config.yaml"):
        super().__init__()
        self.cfg = Config(cfg_path)
        self.setWindowTitle(self.cfg.app.get("title", "UDP/TCP Studio"))
        self.resize(1500, 900)
        pg.setConfigOptions(antialias=True)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        self.dock_area = DockArea()
        h.addWidget(self.dock_area, 1)

        self.side = QtWidgets.QScrollArea()
        self.side.setWidgetResizable(True)
        self.side_in = QtWidgets.QWidget()
        self.side.setWidget(self.side_in)
        self.vside = QtWidgets.QVBoxLayout(self.side_in)
        h.addWidget(self.side, 0)

        self.status = self.statusBar()
        self.rx_rate_lbl = QtWidgets.QLabel("RX: 0 pps")
        self.last_len_lbl = QtWidgets.QLabel("Vec: -")
        self.status.addPermanentWidget(self.rx_rate_lbl)
        self.status.addPermanentWidget(self.last_len_lbl)

        # State
        self.sources: Dict[str, Any] = {}
        self.channel_specs: Dict[str, ChannelSpec] = {}
        self.buffers_by_source: Dict[str, Dict[str, deque]] = {}
        self.curves: Dict[tuple, pg.PlotDataItem] = {}
        self.panels: Dict[str, pg.PlotWidget] = {}
        self.docks: Dict[str, Dock] = {}
        self.window_samples: Dict[str, int] = {}
        self.decimate: Dict[str, int] = {}
        self.plot_x_mode: Dict[str, str] = {}
        self.plot_fs: Dict[str, float] = {}
        self.plot_color_cycle: Dict[str, List[Any]] = {}
        self.timebufs: Dict[str, deque] = {}   # seconds
        self.ds_ts_cfg: Dict[str, dict] = {}   # {index, scale, zero, t0, fs, cnt}
        self.plot_source: Dict[str, str] = {}
        # Per-plot UX
        self.plot_paused: Dict[str, bool] = {}
        self.follow_x: Dict[str, bool] = {}
        self.y_locked: Dict[str, bool] = {}
        self.locked_range: Dict[str, Tuple[float, float]] = {}

        # Controls & senders
        self.control_meta: Dict[str, dict] = {}
        self.staged: Dict[str, float] = {}
        self.active: Dict[str, float] = {}
        self._group_to_controls: Dict[str, List[str]] = defaultdict(list)
        self._button_timers: Dict[str, QtCore.QTimer] = {}
        self.senders: Dict[str, UdpSender] = {}

        # RX meter
        self._rx_count = 0
        self._rx_t0 = time.time()
        self._rx_meter = QtCore.QTimer(self)
        self._rx_meter.timeout.connect(self._update_rx_rate)
        self._rx_meter.start(500)

        # Build UI
        self._build_everything()

        fps = int(self.cfg.app.get("fps", 60))
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(max(1, int(1000 / max(1, fps))))

        self._mtime = self._file_mtime()
        self.reload_timer = QtCore.QTimer(self)
        self.reload_timer.timeout.connect(self._hot_reload)
        self.reload_timer.start(500)

    # ---------- build / rebuild ----------
    def _file_mtime(self):
        try:
            return os.path.getmtime(self.cfg.path)
        except Exception:
            return 0

    def _hot_reload(self):
        m = self._file_mtime()
        if m != self._mtime:
            self._mtime = m
            try:
                self.cfg.reload()
                self._rebuild_from_cfg()
                self.status.showMessage("Config reloaded.", 1500)
            except Exception as e:
                self.status.showMessage(f"Reload failed: {e}", 3000)

    def _rebuild_from_cfg(self):
        for _, dock in list(self.docks.items()):
            self.dock_area.removeDock(dock)
        self.docks.clear()
        while self.vside.count():
            item = self.vside.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        self.sources.clear()
        self.channel_specs.clear()
        self.buffers_by_source.clear()
        self.curves.clear()
        self.panels.clear()
        self.window_samples.clear()
        self.decimate.clear()
        self.plot_x_mode.clear()
        self.plot_fs.clear()
        self.plot_color_cycle.clear()
        self.timebufs.clear()
        self.ds_ts_cfg.clear()
        self.plot_source.clear()
        self.plot_paused.clear()
        self.follow_x.clear()
        self.y_locked.clear()
        self.locked_range.clear()
        self.control_meta.clear()
        self._group_to_controls.clear()
        self.staged.clear()
        self.active.clear()
        self._button_timers.clear()
        for sd in list(self.senders.values()):
            sd.stop()
        self.senders.clear()

        self._build_everything()

    def _timebuf_len(self) -> int:
        win_sizes = list(self.window_samples.values())
        return max(win_sizes) * 2 if win_sizes else 10000

    def _build_everything(self):
        # 1) Sources
        for ds in self.cfg.data_sources:
            t = ds.get("type")
            if t == "udp":
                src = UdpSource(ds, self)
            elif t == "tcp_client":
                src = TcpClientSource(ds, self)
            elif t == "tcp_server":
                src = TcpServerSource(ds, self)
            else:
                continue
            src.vector.connect(self._rx_handler(ds))
            self.sources[ds["name"]] = src

            # Timestamp config (and synthetic clock support)
            idx = ds.get("timestamp_index", None)
            scale = ds.get("timestamp_scale", None)
            if scale is None:
                units = (ds.get("timestamp_units") or "s").lower()
                units_map = {"s": 1.0, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
                scale = units_map.get(units, 1.0)
            zero = bool(ds.get("timestamp_zero", True))
            fs_src = float(ds.get("sample_rate_hz", 0.0))
            self.ds_ts_cfg[ds["name"]] = {"index": idx, "scale": float(scale), "zero": zero, "t0": None, "fs": fs_src, "cnt": 0}
            self.timebufs[ds["name"]] = deque(maxlen=10000)
            self.buffers_by_source[ds["name"]] = {}

        # 2) Channels
        self._build_channels_from_cfg()

        # 3) Controls & layout tools
        self._build_controls_from_cfg()
        self._build_layout_controls()

        # 4) Senders
        self._build_senders_from_cfg()

        # 5) Plots
        self._build_plots_and_docks()

        # Resize time buffers based on windows
        tlen = self._timebuf_len()
        for k in list(self.timebufs.keys()):
            self.timebufs[k] = deque(self.timebufs[k], maxlen=tlen)

        print("Known channels:", list(self.channel_specs.keys())[:8], "… total:", len(self.channel_specs))

    def _build_channels_from_cfg(self):
        entries = self.cfg.channels or []
        for c in entries:
            if isinstance(c, dict) and "range" in c and isinstance(c["range"], str):
                for name in expand_range_token(c["range"]):
                    _, idx = _split_prefix_num(name)
                    self.channel_specs[name] = ChannelSpec(
                        name=name,
                        index=idx,
                        unit=c.get("unit", ""),
                        scale=float(c.get("scale", 1.0)),
                        offset=float(c.get("offset", 0.0)),
                        color=c.get("color", None),
                        width=float(c.get("width", 1.5)),
                    )
            elif isinstance(c, dict) and "name" in c and "index" in c:
                self.channel_specs[c["name"]] = ChannelSpec(
                    name=c["name"],
                    index=int(c["index"]),
                    unit=c.get("unit", ""),
                    scale=float(c.get("scale", 1.0)),
                    offset=float(c.get("offset", 0.0)),
                    color=c.get("color", None),
                    width=float(c.get("width", 1.5)),
                )
            elif isinstance(c, str):
                for name in expand_range_token(c):
                    _, idx = _split_prefix_num(name)
                    self.channel_specs[name] = ChannelSpec(name=name, index=idx)

        # Ensure per-source buffers exist for all channels
        for src_name, bufmap in self.buffers_by_source.items():
            for ch_name in self.channel_specs.keys():
                bufmap.setdefault(ch_name, deque(maxlen=5000))

        # Fallback autogen
        if not self.channel_specs:
            length = 0
            for ds in (self.cfg.data_sources or []):
                if "length" in ds:
                    length = int(ds["length"]); break
            for i in range(length):
                nm = f"ch{i}"
                self.channel_specs[nm] = ChannelSpec(name=nm, index=i)
            for src_name in self.buffers_by_source.keys():
                self.buffers_by_source[src_name] = {nm: deque(maxlen=5000) for nm in self.channel_specs.keys()}

    def _build_controls_from_cfg(self):
        group_meta = {g["name"]: g for g in (self.cfg.control_groups or [])}
        groups_in_order = [g["name"] for g in (self.cfg.control_groups or [])]
        group_boxes: Dict[str, QtWidgets.QGroupBox] = {}

        def ensure_group_box(gname: str):
            if gname in group_boxes:
                return group_boxes[gname]
            box = QtWidgets.QGroupBox(group_meta.get(gname, {}).get("title", gname))
            v = QtWidgets.QVBoxLayout(box)
            form = QtWidgets.QFormLayout()
            v.addLayout(form)
            box._form = form  # type: ignore
            group_boxes[gname] = box
            self.vside.addWidget(box)
            return box

        for ctl in self.cfg.controls or []:
            cid = ctl["id"]
            grp = ctl.get("group", "default")
            self.control_meta[cid] = ctl
            self._group_to_controls[grp].append(cid)
            default = ctl.get("default", 0.0 if ctl.get("type") != "bool" else False)
            self.staged[cid] = 1.0 if isinstance(default, bool) and default else (float(default) if not isinstance(default, bool) else 0.0)
            self.active[cid] = self.staged[cid]

        for gname in groups_in_order + [g for g in self._group_to_controls if g not in groups_in_order]:
            box = ensure_group_box(gname)
            form = box._form
            for cid in self._group_to_controls.get(gname, []):
                meta = self.control_meta[cid]
                typ = meta.get("type", "float")
                label = meta.get("label", cid)
                apply_required = bool(meta.get("apply_required", True))

                if typ == "float":
                    minv = float(meta.get("min", 0.0)); maxv = float(meta.get("max", 100.0))
                    step = float(meta.get("step", 0.1)); steps = int(meta.get("slider_steps", 1000))
                    container = QtWidgets.QWidget(); h = QtWidgets.QHBoxLayout(container); h.setContentsMargins(0,0,0,0)
                    spin = QtWidgets.QDoubleSpinBox(); spin.setRange(minv, maxv); spin.setSingleStep(step)
                    decimals = min(6, max(0, int(-math.log10(step))) if step < 1 else 3); spin.setDecimals(decimals)
                    spin.setValue(self.staged[cid])
                    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal); slider.setMinimum(0); slider.setMaximum(steps)

                    def to_pos(val): return int(round((float(val)-minv)/(maxv-minv)*steps)) if maxv>minv else 0
                    def to_val(pos): return minv + (maxv - minv) * (pos/steps) if steps>0 else minv

                    slider.setValue(to_pos(self.staged[cid]))

                    def on_spin(v, _cid=cid, _apply=apply_required):
                        self.staged[_cid] = float(v)
                        if not _apply: self.active[_cid] = float(v)
                        slider.blockSignals(True); slider.setValue(to_pos(v)); slider.blockSignals(False)
                    def on_slider(pos, _cid=cid, _apply=apply_required):
                        val = float(to_val(pos)); self.staged[_cid] = val
                        if not _apply: self.active[_cid] = val
                        spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)

                    spin.valueChanged.connect(on_spin); slider.valueChanged.connect(on_slider)
                    h.addWidget(slider, 2); h.addWidget(spin, 1)
                    form.addRow(label, container)

                elif typ == "bool":
                    chk = QtWidgets.QCheckBox(); chk.setChecked(bool(self.staged[cid] >= 0.5))
                    def on_chk(state, _cid=cid, _apply=apply_required):
                        self.staged[_cid] = 1.0 if state else 0.0
                        if not _apply: self.active[_cid] = self.staged[_cid]
                    chk.stateChanged.connect(on_chk)
                    form.addRow(label, chk)

                elif typ == "button":
                    pulse_ms = int(meta.get("pulse_ms", 50))
                    btn = QtWidgets.QPushButton(label)
                    def on_btn(_cid=cid, _ms=pulse_ms):
                        self.active[_cid] = 1.0
                        if _cid in self._button_timers: self._button_timers[_cid].stop()
                        t = QtCore.QTimer(self); t.setSingleShot(True); t.timeout.connect(lambda: self._reset_button(_cid)); t.start(_ms)
                        self._button_timers[_cid] = t
                    btn.clicked.connect(on_btn)
                    form.addRow(btn)

            apply_label = group_meta.get(gname, {}).get("apply_label", None)
            if apply_label:
                abtn = QtWidgets.QPushButton(apply_label)
                def do_apply(_g=gname):
                    for cid in self._group_to_controls.get(_g, []):
                        meta = self.control_meta[cid]
                        if meta.get("type") == "button": continue
                        if bool(meta.get("apply_required", True)): self.active[cid] = self.staged[cid]
                    self.status.showMessage(f"Applied group '{_g}'", 1200)
                abtn.clicked.connect(do_apply)
                form.addRow("", abtn)

        self.vside.addStretch()

    def _build_layout_controls(self):
        box = QtWidgets.QGroupBox("Layout")
        h = QtWidgets.QHBoxLayout(box)
        btn_save = QtWidgets.QPushButton("Save Layout")
        btn_load = QtWidgets.QPushButton("Load Layout")
        h.addWidget(btn_save); h.addWidget(btn_load)
        self.vside.addWidget(box)

        state_path = os.path.join(os.path.dirname(self.cfg.path), "dock_layout.state")
        def save_state():
            st = self.dock_area.saveState()
            with open(state_path, "wb") as f: f.write(bytes(st))
            self.status.showMessage(f"Layout saved to {state_path}", 1500)
        def load_state():
            try:
                with open(state_path, "rb") as f: data = f.read()
                self.dock_area.restoreState(data)
                self.status.showMessage("Layout restored.", 1500)
            except Exception as e:
                self.status.showMessage(f"Load failed: {e}", 2500)
        btn_save.clicked.connect(save_state); btn_load.clicked.connect(load_state)

    def _build_senders_from_cfg(self):
        if not self.cfg.senders:
            return
        box = QtWidgets.QGroupBox("Senders"); v = QtWidgets.QVBoxLayout(box)
        for sd in self.cfg.senders:
            if sd.get("type") != "udp": continue
            row = QtWidgets.QGroupBox(sd.get("name", "tx")); form = QtWidgets.QFormLayout(row)
            form.addRow("Remote:", QtWidgets.QLabel(f"{sd['remote_ip']}:{sd['remote_port']}"))
            form.addRow("Rate:",   QtWidgets.QLabel(f"{sd.get('rate_hz', 100.0)} Hz"))
            form.addRow("Format:", QtWidgets.QLabel(sd.get("format", "float32_le")))
            hb = QtWidgets.QHBoxLayout(); btn_start = QtWidgets.QPushButton("Start"); btn_stop = QtWidgets.QPushButton("Stop"); btn_once = QtWidgets.QPushButton("Send Once")
            hb.addWidget(btn_start); hb.addWidget(btn_stop); hb.addWidget(btn_once); form.addRow(hb)
            meas = QtWidgets.QLabel("TX: - pps"); form.addRow(meas)

            sender = UdpSender(sd, self._get_value, self)
            sender.tx_rate.connect(lambda r, _m=meas: _m.setText(f"TX: {r:.0f} pps"))
            btn_start.clicked.connect(sender.start); btn_stop.clicked.connect(sender.stop); btn_once.clicked.connect(sender.send_once)
            self.senders[sd["name"]] = sender; v.addWidget(row)
        self.vside.addWidget(box)

    def _make_plot_toolbar(self, title: str, w: pg.PlotWidget, default_follow: bool) -> QtWidgets.QWidget:
        bar = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(bar); h.setContentsMargins(0, 0, 0, 0)

        btn_pause  = QtWidgets.QPushButton("Pause");   btn_pause.setCheckable(True)
        btn_follow = QtWidgets.QPushButton("Follow");  btn_follow.setCheckable(True); btn_follow.setChecked(default_follow)
        btn_autoy  = QtWidgets.QPushButton("Auto-Y")
        btn_locky  = QtWidgets.QPushButton("Lock-Y");  btn_locky.setCheckable(True)
        btn_shot   = QtWidgets.QPushButton("Screenshot")

        self.plot_paused[title] = False
        self.follow_x[title] = default_follow
        self.y_locked[title] = False

        def toggle_pause():
            val = btn_pause.isChecked()
            self.plot_paused[title] = val
            btn_pause.setText("Resume" if val else "Pause")

        def toggle_follow():
            self.follow_x[title] = btn_follow.isChecked()

        def auto_y():
            vb = w.getViewBox()
            vb.enableAutoRange(y=True)
            w.autoRange()
            self.y_locked[title] = False
            self.locked_range.pop(title, None)
            btn_locky.setChecked(False)

        def toggle_locky():
            if btn_locky.isChecked():
                vb = w.getViewBox()
                r = vb.viewRange()[1]
                self.locked_range[title] = (float(r[0]), float(r[1]))
                vb.enableAutoRange(y=False)
                self.y_locked[title] = True
            else:
                self.y_locked[title] = False
                w.getViewBox().enableAutoRange(y=True)

        def screenshot():
            try:
                from pyqtgraph.exporters import ImageExporter
                outdir = os.path.join(os.path.dirname(self.cfg.path), "screenshots")
                os.makedirs(outdir, exist_ok=True)
                ts = time.strftime("%Y%m%d-%H%M%S")
                fname = os.path.join(outdir, f"{title.replace(' ', '_')}-{ts}.png")
                exporter = ImageExporter(w.plotItem); exporter.export(fname)
                self.status.showMessage(f"Saved screenshot: {fname}", 2000)
            except Exception as e:
                self.status.showMessage(f"Screenshot failed: {e}", 3000)

        btn_pause.clicked.connect(toggle_pause)
        btn_follow.clicked.connect(toggle_follow)
        btn_autoy.clicked.connect(auto_y)
        btn_locky.clicked.connect(toggle_locky)
        btn_shot.clicked.connect(screenshot)

        h.addWidget(btn_pause); h.addWidget(btn_follow); h.addWidget(btn_autoy); h.addWidget(btn_locky); h.addWidget(btn_shot); h.addStretch(1)
        return bar

    def _build_plots_and_docks(self):
        hide_ts_default = bool(self.cfg.app.get("hide_timestamp_legend", True))

        for p in self.cfg.plots:
            title = p["title"]
            source_name = p.get("source")
            self.plot_source[title] = source_name

            x_axis_cfg = p.get("x_axis", {})
            mode = x_axis_cfg.get("mode", "samples").lower()   # samples | time | timestamp
            fs = float(x_axis_cfg.get("sample_rate_hz", self._infer_fs()))
            self.plot_x_mode[title] = mode
            self.plot_fs[title] = fs

            default_follow = bool(p.get("follow_x", True))

            w = pg.PlotWidget(title=title)
            g = p.get("grid", [False, False])
            w.showGrid(x=bool(g[0]) if isinstance(g, list) and len(g) > 0 else False,
                       y=bool(g[1]) if isinstance(g, list) and len(g) > 1 else False)
            if p.get("y_range"):
                lo, hi = p["y_range"]; w.setYRange(lo, hi)
            w.setLabel("bottom", "Time (s)" if mode in ("time", "timestamp") else "Samples")

            self.panels[title] = w
            self.window_samples[title] = int(p.get("window_samples", 2000))
            self.decimate[title] = int(p.get("decimate", 1))
            self.plot_color_cycle[title] = p.get("color_cycle", []) or []

            container = QtWidgets.QWidget()
            vbox = QtWidgets.QVBoxLayout(container); vbox.setContentsMargins(0, 0, 0, 0)
            vbox.addWidget(self._make_plot_toolbar(title, w, default_follow))
            vbox.addWidget(w)

            dock = Dock(title, size=(400, 300)); dock.addWidget(container)
            self.docks[title] = dock

        rows = self.cfg.layout.get("rows", [[pl["title"]] for pl in self.cfg.plots])
        row_anchors = []
        for r, row in enumerate(rows):
            prev = None
            for c, title in enumerate(row):
                d = self.docks[title]
                if r == 0 and c == 0:
                    self.dock_area.addDock(d, "left"); prev = d; row_anchors.append(d)
                elif c == 0:
                    anchor = row_anchors[r - 1] if r - 1 < len(row_anchors) else prev
                    self.dock_area.addDock(d, "bottom", anchor); prev = d; row_anchors.append(d)
                else:
                    self.dock_area.addDock(d, "right", prev); prev = d

        # Map timestamp-channel names by source for hiding in legend
        ts_names_by_src: Dict[str, set] = {}
        for src_name, cfg in self.ds_ts_cfg.items():
            idx = cfg.get("index")
            if idx is None: continue
            ts_names_by_src[src_name] = {ch.name for ch in self.channel_specs.values() if ch.index == idx}

        # Curves per plot
        # Curves per plot
        for p in self.cfg.plots:
            title = p["title"]
            w = self.panels[title]
            legend = w.addLegend()

            # expand requested names
            names = expand_list(p.get("channels", []))
            cycle = self.plot_color_cycle.get(title, [])
            src = self.plot_source.get(title)

            # --- NEW: optionally drop the timestamp channel entirely ---
            drop_ts_default = bool(self.cfg.app.get("drop_timestamp_channel", True))
            drop_ts = bool(p.get("drop_timestamp_channel", drop_ts_default))

            # build a set of channel names that correspond to the timestamp index for this source
            ts_names_by_src: Dict[str, set] = {}
            for src_name, cfg in self.ds_ts_cfg.items():
                idx = cfg.get("index")
                if idx is None:
                    continue
                ts_names_by_src[src_name] = {
                    ch.name for ch in self.channel_specs.values() if ch.index == idx
                }
            ts_hide_set = ts_names_by_src.get(src, set())

            if drop_ts and ts_hide_set:
                names = [nm for nm in names if nm not in ts_hide_set]

            # optional: still allow legend hiding if some plot leaves timestamp in on purpose
            hide_ts_default = bool(self.cfg.app.get("hide_timestamp_legend", True))
            hide_ts = bool(p.get("hide_timestamp_legend", hide_ts_default))

            # ensure per-source buffers exist
            if src and src in self.buffers_by_source:
                bufmap = self.buffers_by_source[src]
                for nm in names:
                    bufmap.setdefault(nm, deque(maxlen=self.window_samples[title]))

            for i, name in enumerate(names):
                ch_spec = self.channel_specs.get(name)
                color = (ch_spec.color if ch_spec and ch_spec.color
                         else (cycle[i % len(cycle)] if cycle else None))
                pen = pg.mkPen(color=color, width=(ch_spec.width if ch_spec else 1.5)) \
                      if color else pg.mkPen(width=(ch_spec.width if ch_spec else 1.5))

                legend_name = None if (hide_ts and name in ts_hide_set) else name
                curve = w.plot([], [], name=legend_name, pen=pen)
                curve.setClipToView(True)
                try:
                    curve.setDownsampling(auto=True, mode="subsample")
                except Exception:
                    pass
                self.curves[(title, name)] = curve


    # ---------- controls helpers ----------
    def _reset_button(self, cid):
        self.active[cid] = 0.0

    def _get_value(self, cid: str):
        return self.active.get(cid, 0.0)

    # ---------- RX/plotting ----------
    def _rx_handler(self, ds_spec: dict):
        pad = ds_spec.get("pad_missing", True)
        trunc = ds_spec.get("drop_excess", True)
        expected = int(ds_spec.get("length", 0))
        ds_name = ds_spec.get("name")

        def handler(vec: np.ndarray):
            if expected:
                if pad and len(vec) < expected:
                    v = np.zeros(expected, dtype=np.float32); v[:len(vec)] = vec; vec = v
                if trunc and len(vec) > expected:
                    vec = vec[:expected]

            # Timestamp (real or synthetic)
            ts_cfg = self.ds_ts_cfg.get(ds_name, {})
            ts_idx = ts_cfg.get("index", None)
            if ts_idx is not None and 0 <= ts_idx < len(vec):
                raw = float(vec[ts_idx]) * float(ts_cfg.get("scale", 1.0))
                if ts_cfg.get("zero", True):
                    if ts_cfg.get("t0") is None: ts_cfg["t0"] = raw
                    ts = raw - ts_cfg["t0"]
                else:
                    ts = raw
                self.timebufs[ds_name].append(ts)
            else:
                fs_src = float(ts_cfg.get("fs", 0.0))
                if fs_src > 0:
                    cnt = int(ts_cfg.get("cnt", 0))
                    ts = cnt / fs_src
                    if ts_cfg.get("zero", True) and ts_cfg.get("t0") is None:
                        ts_cfg["t0"] = 0.0
                    self.timebufs[ds_name].append(ts)
                    ts_cfg["cnt"] = cnt + 1

            # Channels into per-source buffers
            bufmap = self.buffers_by_source.get(ds_name)
            if bufmap is None:
                bufmap = self.buffers_by_source[ds_name] = {}
            for name, spec in self.channel_specs.items():
                if spec.index < len(vec):
                    val = float(vec[spec.index]) * spec.scale + spec.offset
                    if name not in bufmap: bufmap[name] = deque(maxlen=5000)
                    bufmap[name].append(val)

            # Derived per-source
            if self.cfg.derived:
                ns = {k: (float(bufmap[k][-1]) if bufmap.get(k) else 0.0) for k in self.channel_specs.keys()}
                ns.update({"np": np, "math": math})
                for d in self.cfg.derived:
                    try:
                        val = eval(d["expr"], {"__builtins__": {}}, ns)
                        nm = d["name"]
                        if nm not in bufmap: bufmap[nm] = deque(maxlen=5000)
                        bufmap[nm].append(float(val))
                    except Exception:
                        pass

            self._rx_count += 1
            self.last_len_lbl.setText(f"Vec: {len(vec)}")
        return handler

    def _update_rx_rate(self):
        now = time.time(); dt = now - self._rx_t0
        if dt > 0:
            rate = self._rx_count / dt
            self.rx_rate_lbl.setText(f"RX: {rate:.0f} pps")
        self._rx_count = 0; self._rx_t0 = now

    def _infer_fs(self) -> float:
        for ds in self.cfg.data_sources or []:
            if "sample_rate_hz" in ds:
                return float(ds["sample_rate_hz"])
        return 0.0

    def _update_plots(self):
        for (title, name), curve in self.curves.items():
            if self.plot_paused.get(title, False):
                continue

            src = self.plot_source.get(title)
            bufmap = self.buffers_by_source.get(src, {}) if src else {}
            buf = bufmap.get(name)
            if not buf:
                continue

            N = int(self.window_samples.get(title, 2000))
            dec = int(max(1, self.decimate.get(title, 1)))
            y_full = np.fromiter(buf, dtype=np.float32)
            if y_full.size == 0:
                continue
            y = y_full[-N::dec] if y_full.size > N else y_full[::dec]
            if y.size == 0:
                continue

            mode = self.plot_x_mode.get(title, "samples")
            x = None

            # Prefer actual/synthetic time buffer when available
            tbuf = self.timebufs.get(src) if src else None
            if mode in ("timestamp", "time") and tbuf and len(tbuf) > 0:
                t_full = np.fromiter(tbuf, dtype=np.float32)
                t = t_full[-N::dec] if t_full.size > N else t_full[::dec]
                m = min(len(t), len(y))
                if m == 0:
                    continue
                x = t[-m:]; y = y[-m:]
                curve.setData(x, y)
            elif mode == "time":
                fs = float(self.plot_fs.get(title, 0.0))
                if fs > 0:
                    dt = 1.0 / fs
                    x = np.arange(len(y), dtype=np.float32) * (dt * dec)
                    curve.setData(x, y)
                else:
                    curve.setData(y)   # no rolling without X
            elif mode == "samples":
                # Use absolute sample index to allow rolling
                L = len(tbuf) if tbuf is not None else len(y_full)
                x_full = np.arange(L, dtype=np.float32)
                x = x_full[-N::dec] if L > N else x_full[::dec]
                m = min(len(x), len(y))
                if m == 0:
                    continue
                x = x[-m:]; y = y[-m:]
                curve.setData(x, y)
            else:
                curve.setData(y)

            # Maintain Y lock
            if self.y_locked.get(title, False):
                rng = self.locked_range.get(title)
                if rng:
                    self.panels[title].setYRange(rng[0], rng[1])

            # Auto-rolling follow
            if self.follow_x.get(title, True) and x is not None and x.size > 0:
                xcfg = next((pl.get("x_axis", {}) for pl in self.cfg.plots if pl.get("title") == title), {})
                if mode in ("timestamp", "time"):
                    win_s = xcfg.get("window_s", None)
                    if win_s is None:
                        if x.size > 1:
                            win_s = float(x[-1] - x[0])
                            if win_s <= 0:
                                fs = float(self.plot_fs.get(title, 0.0))
                                win_s = float(min(N, len(tbuf) if tbuf is not None else N) / fs) if fs > 0 else 5.0
                        else:
                            fs = float(self.plot_fs.get(title, 0.0))
                            win_s = float(N / fs) if fs > 0 else 5.0
                    x_end = float(x[-1]); x_start = x_end - float(win_s)
                    self.panels[title].setXRange(x_start, x_end, padding=0)
                elif mode == "samples":
                    x_end = float(x[-1]); x_start = x_end - float(max(1, min(N, int(x_end + 1))))
                    self.panels[title].setXRange(x_start, x_end, padding=0)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Studio("config.yaml")
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
