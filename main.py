from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, QDir
from gui import Ui_MainWindow

# TODO Allow users to open and close tabs as they please
class GUI(QtWidgets.QMainWindow): # TODO Add QThreads to deal with algorithms in parallel
    def __init__(self):
        super(GUI, self).__init__()
        self.main_window = Ui_MainWindow()
        self.main_window.setupUi(self)

        self.picture_labels = [self.main_window.picture_label, self.main_window.picture_label_2, self.main_window.picture_label_3, self.main_window.picture_label_4]

        self.main_window.actionImport.triggered.connect(self.importButton_pressed)

    def importButton_pressed(self):
        curPath = QDir.currentPath()
        title = 'Please select an image'
        filt = 'File(*.png *.jpg *.bmp *.gif);;all(*.*)'
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, curPath, filt)

        cur_idx = self.main_window.tabWidget.currentIndex()
        self.picture_labels[cur_idx].setPixmap(QtGui.QPixmap(filename))

        # show message box, if no file is imported
        if filename == '':
            return QtWidgets.QMessageBox.information(self, 'Info', 'No File was selected', QtWidgets.QMessageBox.Ok)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    gui_player = GUI()
    gui_player.show()

    sys.exit(app.exec())