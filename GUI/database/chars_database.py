import os
import numpy as np
import re
import cv2 as cv
width = 60
height = 60

class character:
    def __init__(self, char, template):
        self.char = char
        self.template = cv.resize(cv.imread(template,cv.IMREAD_GRAYSCALE), (60, 60), interpolation = cv.INTER_AREA)
        self.col_sum = np.zeros(shape=(height,width))
        self.corr = 0


chars_matching = {
    "alf":"أ",
    "ba'": "ب",
    "gem": "ج",
    "dal": "د",
    "ra'": "ر",
    "zay": "ز",
    "sen": "س",
    "sad": "ص",
    "taa": "ط",
    "ain":"ع",
    "fa'":"ف",
    "qaf":"ق",
    "lam":"ل",
    "mem": "م",
    "noon": "ن",
    "ha'":"ه",
    "waw":"و",
    "ya'":"ي",
    "0":"0",
    "1":"1",
    "2":"2",
    "3":"3",
    "4":"4",
    "5":"5",
    "6":"6",
    "7":"7",
    "8":"8",
    "9":"9"
}

def fill_chars_database(database_characters):
    for char in os.listdir('database/charImages/'):
        for key, val in chars_matching.items():
            if char.startswith(key):
                database_characters.append(character(val, "database/charImages/"+char))
    return database_characters




