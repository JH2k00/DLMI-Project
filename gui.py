# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1439, 929)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setMaximumSize(QtCore.QSize(1439, 878))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab)
        self.verticalLayout.setObjectName("verticalLayout")
        self.comboBox = QtWidgets.QComboBox(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setMinimumSize(QtCore.QSize(200, 0))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.verticalLayout.addWidget(self.comboBox)
        self.picture_label = QtWidgets.QLabel(self.tab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picture_label.sizePolicy().hasHeightForWidth())
        self.picture_label.setSizePolicy(sizePolicy)
        self.picture_label.setText("")
        self.picture_label.setPixmap(QtGui.QPixmap("erdmaennchen.jpg"))
        self.picture_label.setScaledContents(True)
        self.picture_label.setObjectName("picture_label")
        self.verticalLayout.addWidget(self.picture_label)
        self.label = QtWidgets.QLabel(self.tab)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.verticalLayout.addWidget(self.label)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.tab_2)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.comboBox_2 = QtWidgets.QComboBox(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_2.sizePolicy().hasHeightForWidth())
        self.comboBox_2.setSizePolicy(sizePolicy)
        self.comboBox_2.setMinimumSize(QtCore.QSize(200, 0))
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.verticalLayout_2.addWidget(self.comboBox_2)
        self.picture_label_2 = QtWidgets.QLabel(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picture_label_2.sizePolicy().hasHeightForWidth())
        self.picture_label_2.setSizePolicy(sizePolicy)
        self.picture_label_2.setText("")
        self.picture_label_2.setPixmap(QtGui.QPixmap("erdmaennchen.jpg"))
        self.picture_label_2.setScaledContents(True)
        self.picture_label_2.setObjectName("picture_label_2")
        self.verticalLayout_2.addWidget(self.picture_label_2)
        self.label_2 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_2.addWidget(self.label_2)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.tab_3)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.comboBox_3 = QtWidgets.QComboBox(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_3.sizePolicy().hasHeightForWidth())
        self.comboBox_3.setSizePolicy(sizePolicy)
        self.comboBox_3.setMinimumSize(QtCore.QSize(200, 0))
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.verticalLayout_3.addWidget(self.comboBox_3)
        self.picture_label_3 = QtWidgets.QLabel(self.tab_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picture_label_3.sizePolicy().hasHeightForWidth())
        self.picture_label_3.setSizePolicy(sizePolicy)
        self.picture_label_3.setText("")
        self.picture_label_3.setPixmap(QtGui.QPixmap("erdmaennchen.jpg"))
        self.picture_label_3.setScaledContents(True)
        self.picture_label_3.setObjectName("picture_label_3")
        self.verticalLayout_3.addWidget(self.picture_label_3)
        self.label_3 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.verticalLayout_3.addWidget(self.label_3)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.tab_4)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.comboBox_4 = QtWidgets.QComboBox(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox_4.sizePolicy().hasHeightForWidth())
        self.comboBox_4.setSizePolicy(sizePolicy)
        self.comboBox_4.setMinimumSize(QtCore.QSize(200, 0))
        self.comboBox_4.setObjectName("comboBox_4")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.comboBox_4.addItem("")
        self.verticalLayout_4.addWidget(self.comboBox_4)
        self.picture_label_4 = QtWidgets.QLabel(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.picture_label_4.sizePolicy().hasHeightForWidth())
        self.picture_label_4.setSizePolicy(sizePolicy)
        self.picture_label_4.setText("")
        self.picture_label_4.setPixmap(QtGui.QPixmap("erdmaennchen.jpg"))
        self.picture_label_4.setScaledContents(True)
        self.picture_label_4.setObjectName("picture_label_4")
        self.verticalLayout_4.addWidget(self.picture_label_4)
        self.label_4 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setPointSize(20)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_4.addWidget(self.label_4)
        self.tabWidget.addTab(self.tab_4, "")
        self.horizontalLayout.addWidget(self.tabWidget)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1439, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuCommands = QtWidgets.QMenu(self.menubar)
        self.menuCommands.setObjectName("menuCommands")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionImport = QtWidgets.QAction(MainWindow)
        self.actionImport.setObjectName("actionImport")
        self.actionSave = QtWidgets.QAction(MainWindow)
        self.actionSave.setObjectName("actionSave")
        self.actionExport_Logs = QtWidgets.QAction(MainWindow)
        self.actionExport_Logs.setObjectName("actionExport_Logs")
        self.actionRun_Algorithm = QtWidgets.QAction(MainWindow)
        self.actionRun_Algorithm.setObjectName("actionRun_Algorithm")
        self.menuFile.addAction(self.actionImport)
        self.menuFile.addAction(self.actionSave)
        self.menuFile.addAction(self.actionExport_Logs)
        self.menuCommands.addAction(self.actionRun_Algorithm)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuCommands.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(3)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Select an Algorithm"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Deep Learning Algorithm"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Machine Learning Baseline"))
        self.label.setText(_translate("MainWindow", "Hey ! This is a placeholder for future information about the classification"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Sample 1"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Select an Algorithm"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Deep Learning Algorithm"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Machine Learning Baseline"))
        self.label_2.setText(_translate("MainWindow", "Hey ! This is a placeholder for future information about the classification"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Sample 2"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "Select an Algorithm"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "Deep Learning Algorithm"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "Machine Learning Baseline"))
        self.label_3.setText(_translate("MainWindow", "Hey ! This is a placeholder for future information about the classification"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "Sample 3"))
        self.comboBox_4.setItemText(0, _translate("MainWindow", "Select an Algorithm"))
        self.comboBox_4.setItemText(1, _translate("MainWindow", "Deep Learning Algorithm"))
        self.comboBox_4.setItemText(2, _translate("MainWindow", "Machine Learning Baseline"))
        self.label_4.setText(_translate("MainWindow", "Hey ! This is a placeholder for future information about the classification"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "Sample 4"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuCommands.setTitle(_translate("MainWindow", "Commands"))
        self.actionImport.setText(_translate("MainWindow", "Import"))
        self.actionSave.setText(_translate("MainWindow", "Save"))
        self.actionExport_Logs.setText(_translate("MainWindow", "Export Logs"))
        self.actionRun_Algorithm.setText(_translate("MainWindow", "Run Algorithm"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
