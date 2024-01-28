from PyQt5.QtCore import QObject, pyqtSignal
from PyQt5 import QtGui
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class Hist_Worker(QObject):

    hist_signal = pyqtSignal(tuple)

    def __init__(self, bins):
        super(Hist_Worker, self).__init__()
        self.bins = bins

    def run(self):
        pass

    def get_hist_pixmap(self, args):
        image, tab_idx = args
        image = image.reshape(3, -1)
        fig = Figure(dpi=600)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        colors = ["Red", "Green", "Blue"]
        for i in range(3):
            ax.hist(image[i, :], bins=self.bins, density=True, color=colors[2-i], label=colors[i])
        ax.set_title('Histogram of colors')
        ax.set_xlabel("Pixel Value")
        ax.set_ylabel("Sample pdf")
        ax.legend()
        canvas.draw()
        width, height = int(fig.figbbox.width), int(fig.figbbox.height)
        img = QtGui.QImage(canvas.buffer_rgba(), width, height, QtGui.QImage.Format_ARGB32)
        self.hist_signal.emit((QtGui.QPixmap(img), tab_idx))
