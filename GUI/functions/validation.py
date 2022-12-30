import os
from functions.printMessages import *

class Validation:
    inputString = ""

    def __init__(self, inputString):
        self.inputString = inputString

    def validate(self):
        # Some input validations
        if self.inputString == "" or self.inputString == " ":
            printCritical("Empty input for car image name!")
            return 0
        return 1

