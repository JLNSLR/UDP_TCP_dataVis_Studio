# dynamic_plotter.py
import sys, socket, time, re
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import yaml
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
import pyqtgraph as pg

RANGE_RE = re.compile(r"^(?P<prefix>[A-Za-z_]+)?(?P<start>\d+)\.\.(?P<end>\d+)$")

def expand_names(names, registry):
    out = []
    for n in names:
        m = RANGE_RE.match(n)
        if m:
            start, end = int(m.group("start")), int(m.group("end"))
            pre = (m.group("prefix") or "")
            step = 1 if end >= start else -1
            for i in range(start, end + step, step):
                out.append(f"{pre}{i}")
        else:
            out.append(n)
    # check existence later when registry ready
    return out

@dataclass
class ChannelSpec:
    name: str
    index: int
    unit: str = ""
    scale: float = 1.0
    offset: float = 0.0
    color: Any = None

class Config:
    def __init__(self, path):
        self.path = path
        self.reload()

    def reload(self):
        with open(self.path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.app = self.cfg.get("app", {})
        self.ds = self.cfg.get("data_sources", [])
        self.ch = self.cfg.get("channels", [])
        self.derived = self.cfg.get("derived", [])
        self.plots = self.cfg.get("plots", [])
        self.layout = self.cfg.get("layout", {})
        self.senders = self.cfg.get("senders", [])
        self.controls = self.cfg.get("controls", [])
        self.logging = self.cfg.get("logging", {})

class UdpSource(QtCore.QObject):
    vector = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, spec: dict, parent=None):
        super().__init__(parent)
        self.spec = spec
        self.sock = QUdpSocket(self)
        ok = self.sock.bind(QHostAddress.AnyIPv4, int(spec["port"]))
        if not ok:
            raise RuntimeError(f"Could not bind UDP port {spec['port']}")
        self.sock.readyRead.connect(self._on_ready)

    def _on_ready(self):
        while self.sock.hasPendingDatagrams():
            size = self.sock.pendingDatagramSize()
            data, host, port = self.sock.readDatagram(size)
            fmt = self.spec.get("format", "float32_le")
            vec = None
            if fmt == "float32_le" and len(data) % 4 == 0:
                vec = np.frombuffer(data, dtype="<f4").copy()
            else:
                try:
                    parts = [p for p in data.decode("utf-8", "ignore").replace(",", " ").split() if p]
                    vec = np.array([float(p) for p in parts], dtype=np.float32)
                except Exception:
                    continue
            self.vector.emit(vec)

class PlotApp(QtWidgets.QMainWindow):
    def __init__(self, config_path="config.yaml"):
        super().__init__()
        self.cfg = Config(config_path)

        self.setWindowTitle(self.cfg.app.get("title", "Dynamic UDP Plotter"))
        self.resize(1280, 800)

        # central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.hbox = QtWidgets.QHBoxLayout(central)

        # left: plots container
        self.plot_container = QtWidgets.QWidget()
        self.grid = QtWidgets.QGridLayout(self.plot_container)
        self.hbox.addWidget(self.plot_container, 1)

        # right: controls
        self.side = QtWidgets.QScrollArea()
        self.side.setWidgetResizable(True)
        self.side_content = QtWidgets.QWidget()
        self.side.setWidget(self.side_content)
        self.form = QtWidgets.QFormLayout(self.side_content)
        self.hbox.addWidget(self.side, 0)

        # registries
        self.channel_specs: Dict[str, ChannelSpec] = {}
        self.buffers: Dict[str, deque] = {}
        self.curves: Dict[str, pg.PlotDataItem] = {}
        self.panels = []   # list of PlotWidget for updates
        self.sources: Dict[str, UdpSource] = {}
        self.window_samples_per_plot: Dict[str, int] = {}
        self.decimate_per_plot: Dict[str, int] = {}

        self._build_everything()

        # plot timer
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_plots)
        self.timer.start(int(1000 / max(1, int(self.cfg.app.get("fps", 60)))))

        # hot reload watcher (simple: check mtime)
        self._mtime = self._file_mtime()
        self.reload_timer = QtCore.QTimer(self)
        self.reload_timer.timeout.connect(self._hot_reload)
        self.reload_timer.start(500)

    def _file_mtime(self):
        try:
            import os
            return os.path.getmtime(self.cfg.path)
        except Exception:
            return 0

    def _hot_reload(self):
        m = self._file_mtime()
        if m != self._mtime:
            self._mtime = m
            try:
                self.cfg.reload()
                self._clear_ui()
                self._build_everything()
                self.statusBar().showMessage("Config reloaded.", 2000)
            except Exception as e:
                self.statusBar().showMessage(f"Reload failed: {e}", 4000)

    def _clear_ui(self):
        # remove plots
        for i in reversed(range(self.grid.count())):
            item = self.grid.itemAt(i)
            w = item.widget()
            if w:
                w.setParent(None)
        self.panels.clear()
        self.curves.clear()
        self.window_samples_per_plot.clear()
        self.decimate_per_plot.clear()
        # controls
        while self.form.count():
            item = self.form.takeAt(0)
            w = item.widget()
            if w: w.setParent(None)
        # buffers: keep (optional) or clear
        self.buffers.clear()
        # sources: rebuild
        for s in list(self.sources.values()):
            s.deleteLater()
        self.sources.clear()
        self.channel_specs.clear()

    def _build_everything(self):
        # sources
        for ds in self.cfg.ds:
            if ds["type"] == "udp":
                src = UdpSource(ds, self)
                src.vector.connect(self._on_vector(ds["name"], ds))
                self.sources[ds["name"]] = src

        # channels
        for c in self.cfg.ch:
            spec = ChannelSpec(
                name=c["name"], index=int(c["index"]),
                unit=c.get("unit", ""), scale=float(c.get("scale", 1.0)),
                offset=float(c.get("offset", 0.0)),
                color=c.get("color", None)
            )
            self.channel_specs[spec.name] = spec
            self.buffers[spec.name] = deque(maxlen=5000)

        # derived placeholders (buffers)
        for d in self.cfg.derived or []:
            self.buffers[d["name"]] = deque(maxlen=5000)

        # controls (very small demo: float sliders/spinboxes + bool checkbox)
        self.control_values = {}
        for ctl in self.cfg.controls or []:
            cid = ctl["id"]
            t = ctl["type"]
            if t == "float":
                spin = QtWidgets.QDoubleSpinBox()
                spin.setRange(float(ctl["min"]), float(ctl["max"]))
                spin.setSingleStep(float(ctl.get("step", 0.1)))
                spin.setValue(float(ctl.get("default", 0.0)))
                spin.valueChanged.connect(lambda v, _cid=cid: self._set_control(_cid, v))
                self._set_control(cid, spin.value())
                self.form.addRow(ctl.get("label", cid), spin)
            elif t == "bool":
                chk = QtWidgets.QCheckBox()
                chk.setChecked(bool(ctl.get("default", False)))
                chk.stateChanged.connect(lambda s, _cid=cid: self._set_control(_cid, bool(s)))
                self._set_control(cid, chk.isChecked())
                self.form.addRow(ctl.get("label", cid), chk)

        # plots
        title_to_widget = {}
        for p in self.cfg.plots:
            w = pg.PlotWidget(title=p["title"])
            w.showGrid(x=bool(p.get("grid", [False, False])[0]),
                       y=bool(p.get("grid", [False, False])[1]))
            yr = p.get("y_range")
            if yr: w.setYRange(yr[0], yr[1])
            row_idx = len(self.panels)  # temp
            title_to_widget[p["title"]] = w
            self.panels.append(w)
            self.window_samples_per_plot[p["title"]] = int(p.get("window_samples", 2000))
            self.decimate_per_plot[p["title"]] = int(p.get("decimate", 1))

        # layout rows
        for r, row in enumerate(self.cfg.layout.get("rows", [[pl["title"]] for pl in self.cfg.plots])):
            for c, title in enumerate(row):
                self.grid.addWidget(title_to_widget[title], r, c)

        # curves
        for p in self.cfg.plots:
            w = title_to_widget[p["title"]]
            ch_names = expand_names(p["channels"], self.channel_specs)
            legend = w.addLegend()
            for name in ch_names:
                curve = w.plot([], [], name=name)
                self.curves[(p["title"], name)] = curve

    def _set_control(self, cid, value):
        self.control_values[cid] = float(value) if isinstance(value, (int, float)) else bool(value)

    def _on_vector(self, ds_name, ds_spec):
        pad = ds_spec.get("pad_missing", True)
        trunc = ds_spec.get("drop_excess", True)
        length = int(ds_spec.get("length", 0))

        def handler(vec: np.ndarray):
            if length:
                if pad and len(vec) < length:
                    v = np.zeros(length, dtype=np.float32)
                    v[:len(vec)] = vec
                    vec = v
                if trunc and len(vec) > length:
                    vec = vec[:length]

            # base channels
            for name, spec in self.channel_specs.items():
                if spec.index < len(vec):
                    val = float(vec[spec.index]) * spec.scale + spec.offset
                    self.buffers[name].append(val)

            # derived
            if self.cfg.derived:
                # build namespace with channel values at this tick
                ns = {}
                for k in self.buffers.keys():
                    if len(self.buffers[k]) > 0:
                        ns[k] = float(self.buffers[k][-1])
                ns.update({"np": np})
                for d in self.cfg.derived:
                    try:
                        val = eval(d["expr"], {"__builtins__": {}}, ns)
                        self.buffers[d["name"]].append(float(val))
                    except Exception:
                        pass
        return handler

    def _update_plots(self):
        for (title, name), curve in self.curves.items():
            buf = self.buffers.get(name)
            if not buf: continue
            N = self.window_samples_per_plot.get(title, 2000)
            dec = max(1, self.decimate_per_plot.get(title, 1))
            y = np.fromiter(buf, dtype=np.float32, count=len(buf))[-N::dec]
            if y.size == 0: 
                continue
            x = np.arange(len(y))
            curve.setData(x, y)


def main():
    app = QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    w = PlotApp("config.yaml")
    w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
