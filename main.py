from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt
from gui import Ui_MainWindow

class GUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(GUI, self).__init__()
        self.main_window = Ui_MainWindow()
        self.main_window.setupUi(self)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui_player = GUI()
    gui_player.show()

    sys.exit(app.exec())