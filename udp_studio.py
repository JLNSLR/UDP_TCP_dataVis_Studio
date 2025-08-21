import sys, time, socket, re, os
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Any, Callable

import numpy as np
import yaml
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
import pyqtgraph as pg

# --------- Helpers ---------
# accepts "ch0..ch21", "0..21", "adc5..adc9", with optional spaces
RANGE_RE = re.compile(r"^\s*(?P<prefix>[A-Za-z_]*)?(?P<start>\d+)\.\.(?P<end>\d+)\s*$")

def expand_range_token(token: str):
    m = RANGE_RE.match(str(token))
    if not m:
        return [str(token)]
    start, end = int(m.group("start")), int(m.group("end"))
    step = 1 if end >= start else -1
    pre = m.group("prefix") or ""
    return [f"{pre}{i}" for i in range(start, end + step, step)]

def expand_list(tokens):
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

# --------- Config ---------
@dataclass
class ChannelSpec:
    name: str
    index: int
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    color: Any = None

class Config:
    """Minimal loader with a few conveniences and defaults."""
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

# --------- UDP RX ---------
class UdpSource(QtCore.QObject):
    vector = QtCore.pyqtSignal(np.ndarray)

    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.sock = QUdpSocket(self)
        # inside UdpSource.__init__
        # in UdpSource.__init__ (replace your bind lines)
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

# --------- Sender ---------
class UdpSender(QtCore.QObject):
    """Timer-based sender building vector from a template."""
    tx_rate = QtCore.pyqtSignal(float)  # measured sends/sec

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
        self.enabled = False
        self._t_last = time.time()
        self._sent = 0

    def start(self):
        hz = float(self.spec.get("rate_hz", 100.0))
        interval = max(1, int(1000.0 / hz))
        self.timer.start(interval)
        self.enabled = True
        self._t_last = time.time()
        self._sent = 0

    def stop(self):
        self.timer.stop()
        self.enabled = False

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
        # number
        if isinstance(item, (int, float)):
            return float(item)
        # placeholder "$id"
        if isinstance(item, str) and item.startswith("$"):
            val = self.get(item[1:])
            return float(val) if val is not None else 0.0
        # string number
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

# --------- Main App ---------
class Studio(QtWidgets.QMainWindow):
    def __init__(self, cfg_path="config.yaml"):
        super().__init__()
        self.cfg = Config(cfg_path)
        self.setWindowTitle(self.cfg.app.get("title", "UDP Studio"))
        self.resize(1400, 850)
        pg.setConfigOptions(antialias=True)

        # Central split
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        h = QtWidgets.QHBoxLayout(central)

        # Left plots grid
        self.plot_container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.plot_container)
        h.addWidget(self.plot_container, 1)

        # Right side (scroll)
        self.side = QtWidgets.QScrollArea()
        self.side.setWidgetResizable(True)
        self.side_in = QtWidgets.QWidget()
        self.side.setWidget(self.side_in)
        self.vside = QtWidgets.QVBoxLayout(self.side_in)
        h.addWidget(self.side, 0)

        # status
        self.status = self.statusBar()
        self.rx_rate_lbl = QtWidgets.QLabel("RX: 0 pps")
        self.last_len_lbl = QtWidgets.QLabel("Vec: -")
        self.status.addPermanentWidget(self.last_len_lbl)

        self.status.addPermanentWidget(self.rx_rate_lbl)
                # in Studio.__init__ after rx_rate label:
        self.last_len_lbl = QtWidgets.QLabel("Vec: -")
        self.buf_len_lbl  = QtWidgets.QLabel("ch0:0 ch1:0")
        self.status.addPermanentWidget(self.last_len_lbl)
        self.status.addPermanentWidget(self.buf_len_lbl)


        # state
        self.sources: Dict[str, UdpSource] = {}
        self.channel_specs: Dict[str, ChannelSpec] = {}
        self.buffers: Dict[str, deque] = {}
        self.curves: Dict[tuple, pg.PlotDataItem] = {}
        self.panels: Dict[str, pg.PlotWidget] = {}
        self.window_samples: Dict[str, int] = {}
        self.decimate: Dict[str, int] = {}

        # controls (staged vs active)
        self.control_meta: Dict[str, dict] = {}    # id -> meta (group, apply_required, type, etc.)
        self.staged: Dict[str, float] = {}
        self.active: Dict[str, float] = {}
        self._group_to_controls: Dict[str, List[str]] = defaultdict(list)

        # buttons (momentary pulses)
        self._button_timers: Dict[str, QtCore.QTimer] = {}

        # senders
        self.senders: Dict[str, UdpSender] = {}

        # RX rate meter
        self._rx_count = 0
        self._rx_t0 = time.time()
        self._rx_meter = QtCore.QTimer(self)
        self._rx_meter.timeout.connect(self._update_rx_rate)
        self._rx_meter.start(500)

        # Build UI from config
        self._build_everything()

        # Plot update timer
        fps = int(self.cfg.app.get("fps", 60))
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self._update_plots)
        self.plot_timer.start(max(1, int(1000 / max(1, fps))))

        # Hot reload watcher
        self._mtime = self._file_mtime()
        self.reload_timer = QtCore.QTimer(self)
        self.reload_timer.timeout.connect(self._hot_reload)
        self.reload_timer.start(500)

    # ----- Build / Reload -----
    def _file_mtime(self):
        try: return os.path.getmtime(self.cfg.path)
        except Exception: return 0

    def _hot_reload(self):
        m = self._file_mtime()
        if m != self._mtime:
            self._mtime = m
            try:
                self.cfg.reload()
                self._rebuild_from_cfg()
                self.status.showMessage("Config reloaded.", 2000)
            except Exception as e:
                self.status.showMessage(f"Reload failed: {e}", 4000)

    def _rebuild_from_cfg(self):
        # For simplicity, clear and rebuild. (Could diff for smarter updates.)
        self._clear_ui()
        self._build_everything()

    def _clear_ui(self):
        for i in reversed(range(self.grid.count())):
            w = self.grid.itemAt(i).widget()
            if w: w.setParent(None)
        self.panels.clear()
        self.curves.clear()
        self.window_samples.clear()
        self.decimate.clear()

        while self.vside.count():
            item = self.vside.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)

        for s in list(self.sources.values()):
            s.deleteLater()
        self.sources.clear()

        self.channel_specs.clear()
        self.buffers.clear()

        # keep active values if IDs persist
        self.control_meta.clear()
        self._group_to_controls.clear()

        # stop senders
        for sd in list(self.senders.values()):
            sd.stop()
        self.senders.clear()
        
    def _build_channels_from_cfg(self):
        self.channel_specs.clear()
        self.buffers.clear()

        entries = self.cfg.channels or []
        for c in entries:
            # Case A: dict with a range: "ch0..ch21" or list of ranges
            if isinstance(c, dict) and "range" in c:
                rng = c["range"]
                tokens = rng if isinstance(rng, (list, tuple)) else [rng]
                names = expand_list(tokens)
                for name in names:
                    m = re.search(r"(\d+)$", name)
                    if not m:
                        continue  # need numeric suffix to infer index
                    idx = int(m.group(1))
                    spec = ChannelSpec(
                        name=name, index=idx,
                        unit=c.get("unit",""), scale=float(c.get("scale",1.0)),
                        offset=float(c.get("offset",0.0)), color=c.get("color", None)
                    )
                    self.channel_specs[name] = spec
                    self.buffers[name] = deque(maxlen=5000)

            # Case B: explicit one-by-one channels
            elif isinstance(c, dict) and "name" in c and "index" in c:
                spec = ChannelSpec(
                    name=c["name"], index=int(c["index"]),
                    unit=c.get("unit",""), scale=float(c.get("scale",1.0)),
                    offset=float(c.get("offset",0.0)), color=c.get("color", None)
                )
                self.channel_specs[spec.name] = spec
                self.buffers[spec.name] = deque(maxlen=5000)

            # Case C: plain string like "ch0..ch21"
            elif isinstance(c, str):
                for name in expand_range_token(c):
                    m = re.search(r"(\d+)$", name)
                    if not m:
                        continue
                    idx = int(m.group(1))
                    spec = ChannelSpec(name=name, index=idx)
                    self.channel_specs[name] = spec
                    self.buffers[name] = deque(maxlen=5000)

        # --- Fallbacks so you’re never blank ---
        if not self.channel_specs:
            # (1) Autowire from plots’ channel names
            plot_names = []
            for p in (self.cfg.plots or []):
                plot_names += expand_list(p.get("channels", []))
            for name in plot_names:
                m = re.search(r"(\d+)$", name)
                if m and name not in self.channel_specs:
                    idx = int(m.group(1))
                    self.channel_specs[name] = ChannelSpec(name=name, index=idx)
                    self.buffers[name] = deque(maxlen=5000)

        if not self.channel_specs:
            # (2) Autowire from data_source length → ch0..ch{N-1}
            try:
                ds0 = next(d for d in self.cfg.data_sources if "length" in d)
                N = int(ds0["length"])
                for i in range(N):
                    name = f"ch{i}"
                    self.channel_specs[name] = ChannelSpec(name=name, index=i)
                    self.buffers[name] = deque(maxlen=5000)
            except StopIteration:
                pass

        print("Known channels:", list(self.channel_specs.keys())[:8], "… total:", len(self.channel_specs))
    

    def _build_everything(self):
        # ----- Data sources -----
        for ds in self.cfg.data_sources:
            if ds.get("type") == "udp":
                src = UdpSource(ds, self)
                src.vector.connect(self._rx_handler(ds))
                self.sources[ds["name"]] = src

        self._build_channels_from_cfg()

        # Derived placeholders
        for d in self.cfg.derived or []:
            self.buffers[d["name"]] = deque(maxlen=5000)

        # ----- Controls (groups with staged/apply) -----
        group_meta = {g["name"]: g for g in (self.cfg.control_groups or [])}
        groups_in_order = [g["name"] for g in (self.cfg.control_groups or [])]

        # prepare UI per group
        group_boxes: Dict[str, QtWidgets.QGroupBox] = {}

        for ctl in self.cfg.controls or []:
            cid = ctl["id"]
            grp = ctl.get("group", "default")
            self.control_meta[cid] = ctl
            self._group_to_controls[grp].append(cid)
            # staged & active defaults
            default = ctl.get("default", 0.0 if ctl.get("type") != "bool" else False)
            if isinstance(default, bool):
                self.staged[cid] = 1.0 if default else 0.0
                self.active[cid] = 1.0 if default else 0.0
            else:
                self.staged[cid] = float(default)
                self.active[cid] = float(default)

        # build groups UI in declared order, then any remaining
        done_groups = set()
        def ensure_group_box(gname: str):
            if gname in group_boxes: return group_boxes[gname]
            box = QtWidgets.QGroupBox(group_meta.get(gname, {}).get("title", gname))
            v = QtWidgets.QVBoxLayout(box)
            form = QtWidgets.QFormLayout()
            v.addLayout(form)
            # store layout for later
            box._form = form  # type: ignore
            group_boxes[gname] = box
            self.vside.addWidget(box)
            return box

        for gname in groups_in_order + [g for g in self._group_to_controls if g not in groups_in_order]:
            box = ensure_group_box(gname)
            form = box._form

            # add controls belonging to this group
            for cid in self._group_to_controls.get(gname, []):
                meta = self.control_meta[cid]
                typ = meta.get("type", "float")
                label = meta.get("label", cid)

                if typ == "float":
                    # slider + spinbox (staged)
                    minv = float(meta.get("min", 0.0))
                    maxv = float(meta.get("max", 100.0))
                    step = float(meta.get("step", 0.1))
                    steps = int(meta.get("slider_steps", 1000))
                    container = QtWidgets.QWidget()
                    h = QtWidgets.QHBoxLayout(container); h.setContentsMargins(0,0,0,0)
                    spin = QtWidgets.QDoubleSpinBox()
                    spin.setRange(minv, maxv); spin.setSingleStep(step)
                    spin.setDecimals(min(6, max(0, -int(np.log10(step))) if step > 0 else 3))
                    spin.setValue(self.staged[cid])
                    slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
                    slider.setMinimum(0); slider.setMaximum(steps)
                    # mapping
                    def to_pos(val): return int(round((val - minv) / (maxv - minv) * steps)) if maxv > minv else 0
                    def to_val(pos): return minv + (maxv - minv) * (pos / steps) if steps > 0 else minv
                    slider.setValue(to_pos(self.staged[cid]))

                    apply_required = bool(meta.get("apply_required", True))
                    def on_spin(v, _cid=cid, _apply=apply_required, _to_pos=to_pos):
                        self.staged[_cid] = float(v)
                        if not _apply:
                            self.active[_cid] = float(v)
                        slider.blockSignals(True); slider.setValue(_to_pos(v)); slider.blockSignals(False)
                    def on_slider(pos, _cid=cid, _apply=apply_required, _to_val=to_val):
                        val = float(_to_val(pos))
                        self.staged[_cid] = val
                        if not _apply:
                            self.active[_cid] = val
                        spin.blockSignals(True); spin.setValue(val); spin.blockSignals(False)

                    spin.valueChanged.connect(on_spin)
                    slider.valueChanged.connect(on_slider)

                    h.addWidget(slider, 2)
                    h.addWidget(spin, 1)
                    form.addRow(label, container)

                elif typ == "bool":
                    chk = QtWidgets.QCheckBox()
                    chk.setChecked(bool(self.staged[cid] >= 0.5))
                    apply_required = bool(meta.get("apply_required", True))
                    def on_chk(state, _cid=cid, _apply=apply_required):
                        self.staged[_cid] = 1.0 if state else 0.0
                        if not _apply:
                            self.active[_cid] = self.staged[_cid]
                    chk.stateChanged.connect(on_chk)
                    form.addRow(label, chk)

                elif typ == "button":
                    # momentary pulse -> sets active[cid] to 1 for pulse_ms then back to 0
                    pulse_ms = int(meta.get("pulse_ms", 50))
                    btn = QtWidgets.QPushButton(label)
                    def on_btn(_cid=cid, _ms=pulse_ms):
                        self.active[_cid] = 1.0
                        if _cid in self._button_timers:
                            self._button_timers[_cid].stop()
                        t = QtCore.QTimer(self)
                        t.setSingleShot(True)
                        t.timeout.connect(lambda: self._reset_button(_cid))
                        t.start(_ms)
                        self._button_timers[_cid] = t
                    btn.clicked.connect(on_btn)
                    form.addRow(btn)

            # apply button if group has one
            apply_label = group_meta.get(gname, {}).get("apply_label", None)
            if apply_label:
                abtn = QtWidgets.QPushButton(apply_label)
                def do_apply(_g=gname):
                    for cid in self._group_to_controls.get(_g, []):
                        meta = self.control_meta[cid]
                        typ = meta.get("type", "float")
                        if typ == "button":
                            continue  # momentary
                        # only apply if flagged or default True
                        if bool(meta.get("apply_required", True)):
                            self.active[cid] = self.staged[cid]
                    self.status.showMessage(f"Applied group '{_g}'", 1500)
                form.addRow("", abtn)
                abtn.clicked.connect(do_apply)

        # ----- Senders (UI + engine) -----
        if self.cfg.senders:
            senders_box = QtWidgets.QGroupBox("Senders")
            v = QtWidgets.QVBoxLayout(senders_box)
            for sd in self.cfg.senders:
                if sd.get("type") != "udp": continue
                row = QtWidgets.QGroupBox(sd.get("name", "tx"))
                form = QtWidgets.QFormLayout(row)
                ip_lbl = QtWidgets.QLabel(f"{sd['remote_ip']}:{sd['remote_port']}")
                rate_lbl = QtWidgets.QLabel(f"{sd.get('rate_hz', 100.0)} Hz")
                fmt_lbl = QtWidgets.QLabel(sd.get("format", "float32_le"))
                form.addRow("Remote:", ip_lbl)
                form.addRow("Rate:", rate_lbl)
                form.addRow("Format:", fmt_lbl)
                hb = QtWidgets.QHBoxLayout()
                btn_start = QtWidgets.QPushButton("Start")
                btn_stop  = QtWidgets.QPushButton("Stop")
                btn_once  = QtWidgets.QPushButton("Send Once")
                hb.addWidget(btn_start); hb.addWidget(btn_stop); hb.addWidget(btn_once)
                form.addRow(hb)
                meas = QtWidgets.QLabel("TX: - pps")
                form.addRow(meas)

                sender = UdpSender(sd, self._get_value, self)
                sender.tx_rate.connect(lambda r, _m=meas: _m.setText(f"TX: {r:.0f} pps"))
                btn_start.clicked.connect(sender.start)
                btn_stop.clicked.connect(sender.stop)
                btn_once.clicked.connect(sender.send_once)
                self.senders[sd["name"]] = sender
                v.addWidget(row)
            self.vside.addWidget(senders_box)

        self.vside.addStretch()

        # ----- Plots & layout -----
        # create plot widgets first
        for p in self.cfg.plots:
            w = pg.PlotWidget(title=p["title"])
            g = p.get("grid", [False, False])
            w.showGrid(x=bool(g[0]) if isinstance(g, list) and len(g) > 0 else False,
                       y=bool(g[1]) if isinstance(g, list) and len(g) > 1 else False)
            if p.get("y_range"):
                lo, hi = p["y_range"]
                w.setYRange(lo, hi)
            self.panels[p["title"]] = w
            self.window_samples[p["title"]] = int(p.get("window_samples", 2000))
            self.decimate[p["title"]] = int(p.get("decimate", 1))

        # layout rows
        rows = self.cfg.layout.get("rows", [[pl["title"]] for pl in self.cfg.plots])
        for r, row in enumerate(rows):
            for c, title in enumerate(row):
                self.grid.addWidget(self.panels[title], r, c)

        # curves
        for p in self.cfg.plots:
            w = self.panels[p["title"]]
            legend = w.addLegend()
            names = expand_list(p["channels"])
            for name in names:
                curve = self.panels[p["title"]].plot([], [], name=name)
                self.curves[(p["title"], name)] = curve
                if name not in self.buffers:
                    self.buffers[name] = deque(maxlen=self.window_samples[p["title"]])



    # ----- Controls helpers -----
    def _reset_button(self, cid):
        self.active[cid] = 0.0

    def _get_value(self, cid: str):
        # active value for sender template
        return self.active.get(cid, 0.0)

    # ----- RX handling -----
    def _rx_handler(self, ds_spec: dict):
        pad = ds_spec.get("pad_missing", True)
        trunc = ds_spec.get("drop_excess", True)
        expected = int(ds_spec.get("length", 0))
        fmt = ds_spec.get("format", "float32_le")

        def handler(vec: np.ndarray):
            if expected:
                if pad and len(vec) < expected:
                    v = np.zeros(expected, dtype=np.float32)
                    v[:len(vec)] = vec
                    vec = v
                if trunc and len(vec) > expected:
                    vec = vec[:expected]
            # base channels
            for name, spec in self.channel_specs.items():
                if spec.index < len(vec):
                    val = float(vec[spec.index]) * spec.scale + spec.offset
                    self._append(name, val)
            # derived
            if self.cfg.derived:
                ns = {k: (float(self.buffers[k][-1]) if self.buffers.get(k) else 0.0)
                      for k in self.buffers.keys()}
                try:
                    import math
                    ns.update({"np": np, "math": math})
                except Exception:
                    pass
                for d in self.cfg.derived:
                    try:
                        val = eval(d["expr"], {"__builtins__": {}}, ns)
                        self._append(d["name"], float(val))
                    except Exception:
                        pass
            # at the end of handler(vec) inside _rx_handler(...)
            self.last_len_lbl.setText(f"Vec: {len(vec)}")
            L0 = len(self.buffers.get("ch0", []))
            L1 = len(self.buffers.get("ch1", []))
            self.buf_len_lbl.setText(f"ch0:{L0} ch1:{L1}")
            # rx rate
            self._rx_count += 1
            
            
        return handler

    def _append(self, name, val):
        if name not in self.buffers:
            self.buffers[name] = deque(maxlen=5000)
        self.buffers[name].append(val)

    # ----- Meters / Plots -----
    def _update_rx_rate(self):
        now = time.time()
        dt = now - self._rx_t0
        if dt > 0:
            rate = self._rx_count / dt
            self.rx_rate_lbl.setText(f"RX: {rate:.0f} pps")
        self._rx_count = 0
        self._rx_t0 = now

    def _update_plots(self):
        for (title, name), curve in self.curves.items():
            buf = self.buffers.get(name)
            if not buf:  # no buffer or empty deque
                continue

            N   = int(self.window_samples.get(title, 2000))
            dec = int(max(1, self.decimate.get(title, 1)))

            y_full = np.fromiter(buf, dtype=np.float32)
            if y_full.size == 0:
                continue

            # take the most recent N samples (or all if shorter), then decimate
            if y_full.size > N:
                y = y_full[-N::dec]
            else:
                y = y_full[::dec]

            if y.size == 0:
                continue

            # Optional: sanitize to avoid NaN/Inf wrecking autorange
            # y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

            curve.setData(y)  # implicit x = 0..len(y)-1


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = Studio("config.yaml")
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
