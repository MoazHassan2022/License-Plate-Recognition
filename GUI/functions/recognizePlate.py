from functions.commonfunctions import *
from database.chars_database import *
from skimage.measure import find_contours
import cv2 as cv
from skimage import img_as_ubyte
import numpy as np
from skimage.morphology import binary_opening
from skimage.morphology import (rectangle)
from skimage.draw import rectangle
from functions.detectPlate import plateDetect
from database.check_plates_database import checkInDatabase
from functions.printMessages import *
import traceback
import logging

dim = (60, 60)

database_characters = []


def runDatabase():
    fill_chars_database(database_characters)


def recognizeCar(imageRead):
    plateRead = plateDetect(imageRead)
    plate = (rgb2gray(plateRead) * 255).astype("uint8")
    image = cv.threshold(plate, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    x, y = image.shape
    structuringEl = np.ones((2, 2))
    image = binary_opening(image, structuringEl)
    image = image[int(x // 2.2):x - int(x // 15), int(y // 15):]
    image = cropPlate(image)
    image = removeBlackColumns(image)
    charText = getChars(image)
    return checkInDatabase(charText, 'database/database.txt'), charText, plateRead


class characterContour:
    def __init__(self, char, template):
        self.char = char
        self.template = cv.resize(template, dim, interpolation=cv.INTER_AREA)
        self.col_sum = np.zeros(shape=(60, 60))
        self.corr = 0


def detectChar(unknownChar):
    minError = 500000000000
    currError = 500000000000
    for databaseChar in database_characters:
        currError = getError(unknownChar.template, databaseChar.template)
        if currError < minError:
            unknownChar.char = databaseChar.char
            minError = currError
    return unknownChar.char


def movePointBy90(hieght, width, i, j):
    iNew = -j + width - 1
    jNew = i
    return int(iNew), int(jNew)


def rotateMatrix90(mat):
    image2 = np.zeros([mat.shape[1], mat.shape[0]])
    for i in range(mat.shape[0] - 1):
        for j in range(mat.shape[1] - 1):
            newPoint = movePointBy90(mat.shape[0], mat.shape[1], i, j)
            image2[newPoint[0], newPoint[1]] = mat[i, j]
    return image2


def rotateMatrixNTimes(mat, n=1):
    for i in range(0, n):
        mat = rotateMatrix90(mat)
    return mat


def checkRange(width, Xmin, Xmax):
    minWidth, maxWidth = width * 0.42, width * 0.57
    return not ((minWidth < Xmin and Xmin < maxWidth) and (minWidth < Xmax and Xmax < maxWidth))


def checkValidContour(Xmin, Xmax, Ymin, Ymax):
    width = abs(Ymax - Ymin)
    height = abs(Xmax - Xmin)
    if (width == 0):
        width = 1
    heightOverWidth = height / width
    return (heightOverWidth >= 0.12 and heightOverWidth <= 2.9 and width > 11 and height > 3)


def detectDefect(char):
    charAvg = char.mean()
    if charAvg > 0.7:
        # many white pixels
        return True
    else:
        return False


def cropPlateFromRight(image):
    rows, cols = image.shape
    if (rows == 0):
        return image
    indexToCropFrom = -1
    for col in range(cols - 1, 0, -1):
        colAvg = sum(image[:, col]) / rows
        if (colAvg > 0.7):
            indexToCropFrom = col
            break
    image = image[:, :indexToCropFrom]
    return image


def cropPlateFromLeft(image):
    rows, cols = image.shape
    if (rows == 0):
        return image
    indexToCropFrom = -1
    for col in range(0, cols):
        colAvg = sum(image[:, col]) / rows
        if (colAvg > 0.7):
            indexToCropFrom = col
            break
    image = image[:, indexToCropFrom:cols]
    return image


def cropPlateFromTop(image):
    rows, cols = image.shape
    if (cols == 0):
        return image
    indexToCropFrom = -1
    for row in range(0, rows):
        rowAvg = sum(image[row]) / cols
        if (rowAvg > 0.7):
            indexToCropFrom = row
            break
    image = image[indexToCropFrom:rows, :]
    return image


def cropPlateFromBottom(image):
    rows, cols = image.shape
    if (cols == 0):
        return image
    indexToCropFrom = -1
    for row in range(rows - 1, 0, -1):
        rowAvg = sum(image[row]) / cols
        if (rowAvg > 0.7):
            indexToCropFrom = row
            break
    image = image[:indexToCropFrom, :]
    return image


def removeBlackColumns(image):
    rows, cols = image.shape
    if (rows == 0):
        return image
    indexToCropFrom = -1
    for col in range(0, cols):
        colAvg = sum(image[:, col]) / rows
        if (colAvg <= 0.15):
            image[:, col] = 1
    return image


def cropPlate(image):
    image = cropPlateFromTop(image)
    image = cropPlateFromLeft(image)
    image = cropPlateFromRight(image)
    image = cropPlateFromBottom(image)
    return image


def getChars(img):
    imgRows, imgCols = img.shape
    if (imgRows < 2 or imgCols < 2):
        return []
    charTexts = []
    contours = find_contours(img, 0.8)
    with_boxes = np.zeros(img.shape, dtype=float)
    bounding_boxes = []
    for contour in contours:
        Xmin = int(np.min(contour[:, 1]))
        Xmax = int(np.max(contour[:, 1]))
        Ymin = int(np.min(contour[:, 0]))
        Ymax = int(np.max(contour[:, 0]))
        bounding_boxes.append([Xmin, Xmax, Ymin, Ymax])

    bounding_boxes.sort(key=lambda x: x[0])
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        if (checkValidContour(Xmin, Xmax, Ymin, Ymax) and checkRange(imgCols, Xmin, Xmax)):
            rr, cc = rectangle(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=with_boxes.shape)
            with_boxes[rr, cc] = 1  # set color black
            char = img[rr, cc]
            char = rotateMatrixNTimes(char, 3)
            char = np.fliplr(char)
            rows, cols = char.shape
            char = char[1:rows - 1, 1:cols - 1]  # Cancel black borders (thickness is only 1 pixel)
            if (abs(Xmax - Xmin) <= 4 and detectDefect(char)):
                continue
            char = characterContour("Unknown", img_as_ubyte(char))
            textChar = detectChar(char)
            charTexts.append(textChar)
    charTexts.reverse()
    return charTexts


def getError(img1, img2):
    dim = (60, 60)
    img1 = cv.GaussianBlur(img1, (19, 19), 0)
    img2 = cv.GaussianBlur(img2, (19, 19), 0)
    img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)
    _, img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    _, img2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    error = img1 - img2
    error = error * error
    error = np.sum(error)
    error = np.sqrt(error)
    # sqrt(sum(square(img1 - img2)))
    return error
