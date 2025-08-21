# quick_scope.py
import sys, numpy as np
from collections import deque
from PyQt5 import QtWidgets
from PyQt5.QtNetwork import QUdpSocket, QHostAddress
import pyqtgraph as pg

app = QtWidgets.QApplication(sys.argv)
w = pg.PlotWidget(title="QuickScope")
w.showGrid(x=True, y=True); w.show()

buf = deque(maxlen=100000)
curve = w.plot([], [])

sock = QUdpSocket()
sock.bind(QHostAddress.AnyIPv4, 5005)

def on_rx():
    while sock.hasPendingDatagrams():
        size = sock.pendingDatagramSize()
        data, host, port = sock.readDatagram(size)
        if len(data)%4==0 and len(data)>0:
            v = np.frombuffer(data, dtype="<f4")  # float32 LE
            buf.append(float(v[0] if v.size else 0.0))

sock.readyRead.connect(on_rx)

timer = pg.QtCore.QTimer()
def upd():
    y = np.fromiter(buf, dtype=float)
    if y.size: curve.setData(y)
timer.timeout.connect(upd); timer.start(33)

sys.exit(app.exec_())
