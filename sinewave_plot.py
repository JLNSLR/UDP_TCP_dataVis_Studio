import sys
import numpy as np
from PyQt5 import QtWidgets, QtCore
import pyqtgraph as pg

class SineWavePlot(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Sine Waves")

        # Central widget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Two sine curves
        self.curve1 = self.plot_widget.plot(pen='y', name="sine1")
        self.curve2 = self.plot_widget.plot(pen='c', name="sine2")

        # Time axis
        self.fs = 10000  # internal sample rate (Hz)
        self.t = np.arange(0, 1, 1/self.fs)
        self.phase = 0

        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_plot)
        self.timer.start(2)  # ~500 Hz update (2 ms)

    def update_plot(self):
        self.phase += 0.1
        y1 = np.sin(2*np.pi*5*self.t + self.phase)   # 5 Hz sine
        y2 = np.sin(2*np.pi*1*self.t + self.phase)   # 7 Hz sine
        self.curve1.setData(self.t[:1000], y1[:1000])  # plot only first 1000 points
        self.curve2.setData(self.t[:1000], y2[:1000])

def main():
    app = QtWidgets.QApplication(sys.argv)
    win = SineWavePlot()
    win.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
