from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QThread, Qt, QDir, pyqtSignal
from gui import Ui_MainWindow
from model import ResNetSimCLR
import xgboost as xgb
import torch
from torchvision.io import read_image
from torchvision.transforms import v2
from model_worker import Model_Worker
from hist_worker import Hist_Worker

# TODO Allow users to open and close tabs as they please
class GUI(QtWidgets.QMainWindow):
    images = [None for _ in range(4)]
    model_signal = pyqtSignal(tuple)
    calc_hist_signal = pyqtSignal(tuple)

    def __init__(self, dl_model_path, ml_model_path, device):
        super(GUI, self).__init__()
        self.main_window = Ui_MainWindow()
        self.main_window.setupUi(self)

        self.device = device

        self.picture_labels = [self.main_window.picture_label, self.main_window.picture_label_2, self.main_window.picture_label_3, self.main_window.picture_label_4]
        self.hist_labels = [self.main_window.hist_label, self.main_window.hist_label_2, self.main_window.hist_label_3, self.main_window.hist_label_4]
        self.text_labels = [self.main_window.text_label, self.main_window.text_label_2, self.main_window.text_label_3, self.main_window.text_label_4]
        self.expert_labels = [self.main_window.expert_label, self.main_window.expert_label_2, self.main_window.expert_label_3, self.main_window.expert_label_4]
        self.combo_boxes_algo = [self.main_window.comboBox, self.main_window.comboBox_2, self.main_window.comboBox_4, self.main_window.comboBox_6]
        self.combo_boxes_view = [self.main_window.comboBox_1, self.main_window.comboBox_3, self.main_window.comboBox_5, self.main_window.comboBox_7]

        for hist_label, expert_label in zip(self.hist_labels, self.expert_labels):
            hist_label.setVisible(False)
            expert_label.setVisible(False)

        for combo_box_idx, combo_box_algo in zip(self.combo_boxes_view, self.combo_boxes_algo):
            combo_box_idx.currentIndexChanged.connect(self.enable_expert)
            combo_box_algo.currentIndexChanged.connect(self.display_expert_text)

        self.main_window.actionImport.triggered.connect(self.importButton_pressed)
        self.main_window.actionRun_Algorithm.triggered.connect(self.runButton_pressed)

        dl_model = ResNetSimCLR(base_model="resnet18", out_dim=128)
        dl_model.backbone.fc = torch.nn.Linear(512, 2)
        model_dict = torch.load(dl_model_path)
        dl_model.load_state_dict(model_dict["model_state_dict"])
        dl_model = dl_model.to(device)
        dl_model.eval()

        ml_model = xgb.XGBClassifier(tree_method="hist")
        ml_model.load_model(ml_model_path)

        self.normalize = v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # ImageNet mean and std

        self.model_worker = Model_Worker(models=(dl_model,ml_model), device=device)
        self.thread_model = QThread()
        self.thread_model.setObjectName("Model Thread")

        self.model_worker.moveToThread(self.thread_model)
        self.thread_model.started.connect(self.model_worker.run)
        self.thread_model.finished.connect(self.thread_model.quit)
        self.thread_model.finished.connect(self.model_worker.deleteLater)
        self.thread_model.finished.connect(self.thread_model.deleteLater)

        self.model_worker.output_signal.connect(self.display_output)
        self.model_signal.connect(self.model_worker.run_model)

        self.hist_worker = Hist_Worker(bins=50)
        self.thread_hist = QThread()
        self.thread_hist.setObjectName("Histogram Thread")

        self.hist_worker.moveToThread(self.thread_hist)
        self.thread_hist.started.connect(self.hist_worker.run)
        self.thread_hist.finished.connect(self.thread_hist.quit)
        self.thread_hist.finished.connect(self.hist_worker.deleteLater)
        self.thread_hist.finished.connect(self.thread_hist.deleteLater)

        self.hist_worker.hist_signal.connect(self.display_hist)
        self.calc_hist_signal.connect(self.hist_worker.get_hist_pixmap)

        self.thread_model.start()
        self.thread_hist.start()

    def importButton_pressed(self):
        curPath = QDir.currentPath()
        title = 'Please select an image'
        filt = 'File(*.png *.jpg *.bmp *.gif);;all(*.*)'
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(self, title, curPath, filt)

        cur_idx = self.main_window.tabWidget.currentIndex()
        self.picture_labels[cur_idx].setPixmap(QtGui.QPixmap(filename))
        
        image = read_image(filename)
        self.images[cur_idx] = self.normalize(image.float() / 255)

        self.calc_hist_signal.emit((image, cur_idx))

        # show message box, if no file is imported
        if filename == '':
            return QtWidgets.QMessageBox.information(self, 'Info', 'No File was selected', QtWidgets.QMessageBox.Ok)

    def runButton_pressed(self):
        tab_idx = self.main_window.tabWidget.currentIndex()
        image = self.images[tab_idx]
        if(image is None):
            return QtWidgets.QMessageBox.information(self, 'Info', 'Please import an image first', QtWidgets.QMessageBox.Ok)
        cur_idx = self.combo_boxes_algo[tab_idx].currentIndex()
        self.model_signal.emit((image, cur_idx, tab_idx))

    def display_output(self, args):
        output, tab_idx, model_idx = args
        self.text_labels[tab_idx].setText("%s Learning Prediction : The sample is %sbloody"%("Deep" if model_idx==0 else "Machine","" if output else "not "))

    def display_hist(self, args):
        hist, tab_idx = args
        self.hist_labels[tab_idx].setPixmap(hist)

    def enable_expert(self, idx):
        tab_idx = self.main_window.tabWidget.currentIndex()
        self.hist_labels[tab_idx].setVisible(idx==1)
        self.expert_labels[tab_idx].setVisible(idx==1)

    def display_expert_text(self, idx):
        tab_idx = self.main_window.tabWidget.currentIndex()
        if(idx == 0):
            self.expert_labels[tab_idx].setText("Architecture : Resnet18, Pretraining : SimCLR, Training : Last Layer")
        else:
            self.expert_labels[tab_idx].setText("Architecture : XGBoost, Pretraining : None, Training : All")

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    with_gpu = torch.cuda.is_available()
    if with_gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")    
    print('We are now using %s.' % device)
    gui_player = GUI(dl_model_path=r"saved_models\ResNet_SimCLR.pt", ml_model_path=r"xgboost_clf.json", device=device)
    gui_player.show()

    sys.exit(app.exec())