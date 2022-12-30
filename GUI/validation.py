import os
from printMessages import *

class Validation:
    inputString = ""

    def __init__(self, inputString):
        self.inputString = inputString

    def validate(self):
        # Some input validations
        if self.inputString == "" or self.inputString == " ":
            printCritical("Empty input for car image name!")
            return 0
        found = 0
        for i in os.listdir('cars/'):
            if i == self.inputString:
                found = 1
        if not found:
            printCritical("Image is not found in cars folder!")
            return 0
        return 1

