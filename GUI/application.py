# Used PyQt5 to implement the GUI
from PyQt5.QtWidgets import QDialog
from PyQt5 import QtWidgets, uic
import sys
from recognize import recognizeCar, runDatabase
from validation import Validation
from printMessages import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import cv2 as cv
import traceback
import logging

class Window(QDialog):
    def __init__(self):
        super(Window, self).__init__()
        # Load the implemented UI window
        uic.loadUi('mainWindow.ui', self)
        runDatabase()
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
        carImageName = self.carImageName.text()
        validator = Validation(carImageName)
        if validator.validate():
            databaseCheck, characters, plate, imageRead = recognizeCar(carImageName)
            try:
                imageRead = self.convert_cv_qt(imageRead, 581, 371)
                plate = self.convert_cv_qt(plate, 481, 191)
            except Exception as e:
                logging.error(traceback.format_exc())
            self.carImage.setPixmap(imageRead)
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


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    appWindow = Window()
    sys.exit(app.exec_())
