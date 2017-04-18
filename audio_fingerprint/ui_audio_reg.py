# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'audio_reg.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

import sys
from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_mainWindow(object):
    def setupUi(self, mainWindow):
        mainWindow.setObjectName("mainWindow")
        mainWindow.resize(529, 321)
        self.centralwidget = QtWidgets.QWidget(mainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.show_file = QtWidgets.QLineEdit(self.centralwidget)
        self.show_file.setGeometry(QtCore.QRect(160, 30, 211, 31))
        self.show_file.setObjectName("show_file")
        self.recognize = QtWidgets.QPushButton(self.centralwidget)
        self.recognize.setGeometry(QtCore.QRect(30, 140, 111, 71))
        self.recognize.setObjectName("recognize")
        self.input_file = QtWidgets.QPushButton(self.centralwidget)
        self.input_file.setGeometry(QtCore.QRect(30, 30, 111, 31))
        self.input_file.setObjectName("input_file")
        self.show_message = QtWidgets.QLabel(self.centralwidget)
        self.show_message.setGeometry(QtCore.QRect(160, 140, 331, 111))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(18)
        self.show_message.setFont(font)
        self.show_message.setStyleSheet("color: rgb(255, 0, 0);")
        self.show_message.setText("")
        self.show_message.setObjectName("show_message")
        self.addto_db = QtWidgets.QPushButton(self.centralwidget)
        self.addto_db.setGeometry(QtCore.QRect(30, 80, 111, 41))
        self.addto_db.setObjectName("addto_db")
        self.show_db_message = QtWidgets.QLabel(self.centralwidget)
        self.show_db_message.setGeometry(QtCore.QRect(160, 80, 211, 41))
        font = QtGui.QFont()
        font.setFamily("微软雅黑")
        font.setPointSize(18)
        self.show_db_message.setFont(font)
        self.show_db_message.setStyleSheet("color: rgb(0, 85, 255);")
        self.show_db_message.setText("")
        self.show_db_message.setObjectName("show_db_message")
        self.play_audio = QtWidgets.QPushButton(self.centralwidget)
        self.play_audio.setGeometry(QtCore.QRect(400, 30, 61, 31))
        self.play_audio.setObjectName("play_audio")
        mainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(mainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 529, 23))
        self.menubar.setObjectName("menubar")
        mainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(mainWindow)
        self.statusbar.setObjectName("statusbar")
        mainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(mainWindow)

        QtCore.QMetaObject.connectSlotsByName(mainWindow)
        mainWindow.setTabOrder(self.input_file, self.show_file)
        mainWindow.setTabOrder(self.show_file, self.addto_db)
        mainWindow.setTabOrder(self.addto_db, self.recognize)

    def retranslateUi(self, mainWindow):
        _translate = QtCore.QCoreApplication.translate
        mainWindow.setWindowTitle(_translate("mainWindow", "音频识别"))
        self.recognize.setText(_translate("mainWindow", "识别"))
        self.input_file.setText(_translate("mainWindow", "输入文件"))
        self.addto_db.setText(_translate("mainWindow", "添加到数据库"))
        self.play_audio.setText(_translate("mainWindow", "播放"))
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QMainWindow()
    ui = Ui_mainWindow()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())

