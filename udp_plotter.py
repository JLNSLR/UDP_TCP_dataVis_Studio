import sys
import socket
import time
from collections import deque

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
import pyqtgraph as pg


def parse_floats(payload: bytes):
    """Try binary little-endian float32 first, else csv-like text."""
    if len(payload) % 4 == 0 and len(payload) > 0:
        try:
            arr = np.frombuffer(payload, dtype="<f4")
            # make a copy because buffer may be ephemeral
            return np.array(arr, dtype=np.float32)
        except Exception:
            pass
    # fallback: text
    try:
        text = payload.decode("utf-8", errors="ignore").strip()
        if not text:
            return None
        # split on commas or whitespace
        parts = [p for p in text.replace(",", " ").split() if p]
        return np.array([float(p) for p in parts], dtype=np.float32)
    except Exception:
        return None


class UdpPlotter(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("UDP Vector Plotter & Sender")
        self.resize(1100, 700)

        # --- Central layout
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QHBoxLayout(central)

        # Left: plot
        left = QtWidgets.QVBoxLayout()
        layout.addLayout(left, 2)

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("bottom", "Samples")
        self.plot.setLabel("left", "Value")
        left.addWidget(self.plot)

        # Controls below plot
        plot_controls = QtWidgets.QHBoxLayout()
        left.addLayout(plot_controls)

        plot_controls.addWidget(QtWidgets.QLabel("Window (samples):"))
        self.window_spin = QtWidgets.QSpinBox()
        self.window_spin.setRange(100, 200000)
        self.window_spin.setValue(2000)
        plot_controls.addWidget(self.window_spin)

        self.autorange_btn = QtWidgets.QPushButton("Auto-Range Y")
        plot_controls.addWidget(self.autorange_btn)
        plot_controls.addStretch()

        # Right: controls panel
        right = QtWidgets.QVBoxLayout()
        layout.addLayout(right, 1)

        # --- Receiver group
        rx_group = QtWidgets.QGroupBox("Receiver")
        right.addWidget(rx_group)
        rx_form = QtWidgets.QFormLayout(rx_group)

        self.rx_port = QtWidgets.QSpinBox()
        self.rx_port.setRange(1, 65535)
        self.rx_port.setValue(5005)
        rx_form.addRow("Listen Port:", self.rx_port)

        self.rx_channels = QtWidgets.QSpinBox()
        self.rx_channels.setRange(1, 64)
        self.rx_channels.setValue(4)
        rx_form.addRow("Expected Channels:", self.rx_channels)

        self.start_rx = QtWidgets.QPushButton("Start Receiving")
        self.stop_rx = QtWidgets.QPushButton("Stop")
        self.stop_rx.setEnabled(False)
        rx_h = QtWidgets.QHBoxLayout()
        rx_h.addWidget(self.start_rx)
        rx_h.addWidget(self.stop_rx)
        rx_form.addRow(rx_h)

        # --- Sender group
        tx_group = QtWidgets.QGroupBox("Sender")
        right.addWidget(tx_group)
        tx_form = QtWidgets.QFormLayout(tx_group)

        self.tx_ip = QtWidgets.QLineEdit("127.0.0.1")
        tx_form.addRow("Remote IP:", self.tx_ip)

        self.tx_port = QtWidgets.QSpinBox()
        self.tx_port.setRange(1, 65535)
        self.tx_port.setValue(5006)
        tx_form.addRow("Remote Port:", self.tx_port)

        self.tx_rate = QtWidgets.QDoubleSpinBox()
        self.tx_rate.setRange(0.1, 2000.0)
        self.tx_rate.setDecimals(1)
        self.tx_rate.setValue(100.0)  # Hz
        tx_form.addRow("Rate (Hz):", self.tx_rate)

        self.tx_mode_bin = QtWidgets.QCheckBox("Send binary float32")
        self.tx_mode_bin.setChecked(True)
        tx_form.addRow("", self.tx_mode_bin)

        self.tx_vector = QtWidgets.QLineEdit("1.0, 0.0, -1.0, 0.5")
        tx_form.addRow("Vector (comma/space sep.):", self.tx_vector)

        tx_btns = QtWidgets.QHBoxLayout()
        self.tx_start = QtWidgets.QPushButton("Start Sending")
        self.tx_stop = QtWidgets.QPushButton("Stop")
        self.tx_stop.setEnabled(False)
        self.tx_send_once = QtWidgets.QPushButton("Send Once")
        tx_btns.addWidget(self.tx_start)
        tx_btns.addWidget(self.tx_stop)
        tx_btns.addWidget(self.tx_send_once)
        tx_form.addRow(tx_btns)

        right.addStretch()

        # --- Status bar
        self.status = self.statusBar()
        self.rx_rate_lbl = QtWidgets.QLabel("RX: 0 pps")
        self.last_len_lbl = QtWidgets.QLabel("Last vector len: -")
        self.status.addPermanentWidget(self.rx_rate_lbl)
        self.status.addPermanentWidget(self.last_len_lbl)

        # --- Data & curves
        self.curves = []
        self.buffers = []
        self.maxlen = self.window_spin.value()

        self.window_spin.valueChanged.connect(self._realloc_buffers)
        self.autorange_btn.clicked.connect(lambda: self.plot.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True))

        # --- UDP receive
        self.udp = None
        self.start_rx.clicked.connect(self.start_receiving)
        self.stop_rx.clicked.connect(self.stop_receiving)

        # --- UDP send
        self.tx_timer = QtCore.QTimer(self)
        self.tx_timer.timeout.connect(self._send_vector)
        self.tx_start.clicked.connect(self.start_sending)
        self.tx_stop.clicked.connect(self.stop_sending)
        self.tx_send_once.clicked.connect(self._send_vector_once)

        # --- Plot update timer (decoupled from RX)
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(33)  # ~30 FPS

        # RX rate measurement
        self._rx_count = 0
        self._rx_t0 = time.time()
        self._rx_rate_timer = QtCore.QTimer(self)
        self._rx_rate_timer.timeout.connect(self._update_rx_rate)
        self._rx_rate_timer.start(500)

        # Prepare initial channels/curves
        self._ensure_channels(self.rx_channels.value())
        self.rx_channels.valueChanged.connect(lambda _: self._ensure_channels(self.rx_channels.value()))

    # ---------- Receiver ----------
    def start_receiving(self):
        if self.udp is not None:
            self.stop_receiving()

        self.udp = QUdpSocket(self)
        port = self.rx_port.value()
        ok = self.udp.bind(QHostAddress.AnyIPv4, port)
        if not ok:
            QtWidgets.QMessageBox.critical(self, "Bind Error", f"Could not bind UDP port {port}.")
            self.udp = None
            return

        self.udp.readyRead.connect(self.handle_ready_read)
        self.start_rx.setEnabled(False)
        self.stop_rx.setEnabled(True)
        self.status.showMessage(f"Receiving on UDP port {port} ...", 3000)

    def stop_receiving(self):
        if self.udp is not None:
            self.udp.close()
            self.udp.deleteLater()
            self.udp = None
        self.start_rx.setEnabled(True)
        self.stop_rx.setEnabled(False)
        self.status.showMessage("Receiver stopped.", 2000)

    @QtCore.pyqtSlot()
    def handle_ready_read(self):
        while self.udp and self.udp.hasPendingDatagrams():
            size = self.udp.pendingDatagramSize()
            data, host, port = self.udp.readDatagram(size)
            vec = parse_floats(data)
            if vec is None:
                continue
            self._rx_count += 1
            self.last_len_lbl.setText(f"Last vector len: {len(vec)}")
            self._push_vector(vec)

    def _push_vector(self, vec: np.ndarray):
        ch = self.rx_channels.value()
        if len(vec) < ch:
            # pad with zeros if fewer numbers than channels
            v = np.zeros(ch, dtype=np.float32)
            v[:len(vec)] = vec
            vec = v
        elif len(vec) > ch:
            vec = vec[:ch]

        for i in range(ch):
            self.buffers[i].append(float(vec[i]))

    def _update_rx_rate(self):
        now = time.time()
        dt = now - self._rx_t0
        if dt >= 0.001:
            rate = self._rx_count / dt
            self.rx_rate_lbl.setText(f"RX: {rate:.0f} pps")
        self._rx_count = 0
        self._rx_t0 = now

    # ---------- Sender ----------
    def start_sending(self):
        hz = float(self.tx_rate.value())
        interval_ms = max(1, int(1000.0 / hz))
        self.tx_timer.start(interval_ms)
        self.tx_start.setEnabled(False)
        self.tx_stop.setEnabled(True)
        self.status.showMessage(f"Sending at {hz:.1f} Hz", 2000)

    def stop_sending(self):
        self.tx_timer.stop()
        self.tx_start.setEnabled(True)
        self.tx_stop.setEnabled(False)
        self.status.showMessage("Sender stopped.", 2000)

    def _send_vector_once(self):
        self._send_vector()

    def _send_vector(self):
        vec = self._read_tx_vector()
        if vec is None:
            return
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            addr = (self.tx_ip.text().strip(), int(self.tx_port.value()))
            if self.tx_mode_bin.isChecked():
                payload = np.asarray(vec, dtype=np.float32).tobytes(order="C")
            else:
                payload = (" ".join(f"{x:.6g}" for x in vec)).encode("utf-8")
            sock.sendto(payload, addr)
            sock.close()
        except Exception as e:
            self.status.showMessage(f"Send error: {e}", 4000)

    def _read_tx_vector(self):
        text = self.tx_vector.text().strip()
        if not text:
            return None
        parts = [p for p in text.replace(",", " ").split() if p]
        try:
            return [float(p) for p in parts]
        except Exception:
            self.status.showMessage("Vector parse error. Use numbers separated by comma/space.", 4000)
            return None

    # ---------- Plotting ----------
    def _ensure_channels(self, ch: int):
        # Remove existing curves
        for c in self.curves:
            self.plot.removeItem(c)
        self.curves.clear()

        # Buffers
        self.buffers = [deque(maxlen=self.maxlen) for _ in range(ch)]

        # Create curves
        for i in range(ch):
            curve = self.plot.plot([], [], name=f"ch{i}")
            self.curves.append(curve)

        # Legend (create once)
        if not getattr(self, "_legend", None):
            self._legend = self.plot.addLegend(offset=(10, 10))

        # Rebuild legend
        self._legend.clear()
        for i, c in enumerate(self.curves):
            self._legend.addItem(c, f"ch{i}")

    def _realloc_buffers(self, val):
        self.maxlen = int(val)
        ch = self.rx_channels.value()
        old = self.buffers
        self.buffers = [deque(maxlen=self.maxlen) for _ in range(ch)]
        # copy tail of old into new
        for i in range(min(len(old), ch)):
            for v in list(old[i])[-self.maxlen:]:
                self.buffers[i].append(v)

    def update_plot(self):
        # X is sample index
        if not self.curves:
            return
        length = max((len(b) for b in self.buffers), default=0)
        if length == 0:
            return
        x = np.arange(max(0, length - self.maxlen), length, dtype=np.int32)
        for i, (curve, buf) in enumerate(zip(self.curves, self.buffers)):
            y = np.fromiter(buf, dtype=np.float32, count=len(buf))
            # align to x length (in case some channels shorter)
            y = y[-len(x):] if len(y) >= len(x) else y
            curve.setData(x[-len(y):], y)


def main():
    app = QtWidgets.QApplication(sys.argv)
    # nicer defaults
    pg.setConfigOptions(antialias=True)

    w = UdpPlotter()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
