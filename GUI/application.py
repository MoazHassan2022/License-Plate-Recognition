# Used PyQt5 to implement the GUI
from PyQt5.QtWidgets import QDialog, QApplication,QWidget, QVBoxLayout, QPushButton, QFileDialog , QLabel, QTextEdit
from PyQt5 import QtWidgets, uic
from functions.recognizePlate import recognizeCar, runDatabase
from functions.validation import Validation
from functions.printMessages import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt
import cv2 as cv
import traceback
import logging
import sys
from functions.func import readImage

class Window(QDialog):
    def __init__(self):
        super(Window, self).__init__()
        # Load the implemented UI window
        uic.loadUi('mainWindow.ui', self)
        runDatabase()
        self.selectButton.clicked.connect(self.getImage)
        self.selectedPhoto = ""
        self.showMaximized()
        self.recognizeButton.clicked.connect(self.start)
        self.setWindowTitle("Gate Access Controller")
        self.show()

    def convert_cv_qt(self, cv_img, displayWidth, displayHeight):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_RGBA2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(displayWidth, displayHeight, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def start(self):
        imageRead = self.selectedPhoto
        databaseCheck, characters, plate = recognizeCar(imageRead)
        if len(plate) == 0:
            return
        try:
            plate = self.convert_cv_qt(plate, 481, 191)
        except Exception as e:
            logging.error(traceback.format_exc())
        self.plateImage.setPixmap(plate)
        plateChars = ""
        for char in characters:
            plateChars += char + " "
        printInfo("Car Plate Characters: " + plateChars)
        if databaseCheck:
            printInfo("Car is welcome to enter!")
        else:
            printCritical("Car can't enter!")
        return

    def getImage(self):
        self.plateImage.clear()
        self.carImage.clear()
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:\'')
        selectedPath = fname[0] # image path
        validator = Validation(selectedPath)
        photo = []
        if validator.validate():
            try:
                self.selectedPhoto = readImage(selectedPath)
                photo = self.convert_cv_qt(self.selectedPhoto, 581, 371)
            except Exception as e:
                logging.error(traceback.format_exc())
                return
            self.carImage.setPixmap(photo)
        return


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    appWindow = Window()
    sys.exit(app.exec_())
