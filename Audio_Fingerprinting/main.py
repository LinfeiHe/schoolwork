# -*- coding: utf-8 -*
from ui_audio_reg import *
import os
import subprocess
import decoder
import fingerprint
from mysql import SQLDatabase


class Mythread(QtCore.QThread):
    signal = QtCore.pyqtSignal(dict)
    def __init__(self, filename):
        super(Mythread, self).__init__()
        self.filename = filename

    def run(self):
        db = SQLDatabase()
        channels, fs = decoder.read(self.filename)
        result = set()
        for i in range(len(channels)):
            print("Fingerprinting channel %d/%d for %s" % (i + 1, len(channels), self.filename))
            hashes = fingerprint.fingerprint(channels[i], fs=fs)
            print("Finished channel %d/%d for %s" % (i + 1, len(channels), self.filename))
            result |= set(hashes)
        # check
        matches = db.return_matches(result)
        song = db.align_matches(matches)
        print song
        self.signal.emit(song)


class MainWindow(QtWidgets.QMainWindow, Ui_mainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.recognize.clicked.connect(self.recognize_click)
        self.input_file.clicked.connect(self.input_file_click)
        self.addto_db.clicked.connect(self.addto_db_click)
        self.play_audio.clicked.connect(self.play_audio_click)

    def recognize_click(self):
        filename = self.show_file.text()
        self.show_message.setText('识别中...')
        self.thread = Mythread(filename)
        self.thread.signal.connect(self.show_match_result)
        self.thread.start()

    def show_match_result(self, song):
        if song['confidence'] <= 10:
            self.show_message.setText('没有找到歌曲')
        else:
            self.show_message.setText('%s' % song['song_name'])

    def input_file_click(self):
        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', '/home')
        if fname[0]:
            self.show_file.setText(fname[0])
            self.show_message.setText('')
            self.show_db_message.setText('')

    def addto_db_click(self):
        db = SQLDatabase()
        filename = self.show_file.text()
        song_name, extension = os.path.splitext(os.path.basename(filename))
        channels, fs = decoder.read(filename)
        result = set()
        for i in range(len(channels)):
            print("Fingerprinting channel %d/%d for %s" % (i + 1, len(channels), filename))
            hashes = fingerprint.fingerprint(channels[i], fs=fs)
            print("Finished channel %d/%d for %s" % (i + 1, len(channels), filename))
            result |= set(hashes)
        # add to db table songs and fingerprints
        sid = db.insert_song(song_name)
        db.insert_fingerprints(sid, result)
        self.show_db_message.setText('添加完成')

    def play_audio_click(self):
        command = 'ffplay "%s" ' % self.show_file.text()
        print command
        subprocess.Popen(command, shell=True)

if __name__ == "__main__":
    # ui
    app = QtWidgets.QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    sys.exit(app.exec_())

