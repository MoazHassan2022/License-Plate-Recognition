from commonfunctions import *
from skimage.exposure import histogram
from skimage.measure import find_contours
from skimage.filters import gaussian
import cv2 as cv
import os
from skimage import img_as_ubyte
import functools
from func import *
import imutils
import matplotlib.image as mpimg
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import binary_erosion, binary_dilation, black_tophat, binary_closing, binary_opening, \
    skeletonize, thin, area_closing, area_opening
from skimage.morphology import (square, rectangle, diamond, disk, cube,
                                octahedron, ball, octagon, star)
from skimage.draw import rectangle
from skimage.filters import threshold_otsu
from skimage.filters import sobel
from printMessages import *
import traceback
import logging

dim = (60, 60)

database_characters = []


def recognizeCar(image):
    try:
        imageRead = mpimg.imread("cars/" + image).astype(np.uint8)[:, :, :3]
        # show_images([imageRead], ["Read"])
        plateRead = plateDetect(imageRead)
        plate = (rgb2gray(plateRead) * 255).astype("uint8")
        # show_images([image], ["Original"])
        image = cv.threshold(plate, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
        # show_images([image], ["Binary"])
        x, y = image.shape
        structuringEl = np.ones((2, 2))
        image = binary_opening(image, structuringEl)
        # show_images([image], ["After Morphology"])
        image = image[int(x // 2.2):x - int(x // 15), int(y // 15):]
        # show_images([image], ["After cutting first time"])
        image = cropPlate(image)
        # show_images([image], ["After cutting second time"])
        image = removeBlackColumns(image)
        # show_images([image], ["After removing black columns"])
        charText = getChars(image)
        # print(charText)
        return checkInDatabase(charText, 'database.txt'), charText, plateRead, imageRead
    except Exception as e:
        logging.error(traceback.format_exc())
        printCritical("Error in processing the image")

class characterContour:
    def __init__(self, char, template):
        self.char = char
        self.template = cv.resize(template, dim, interpolation=cv.INTER_AREA)
        self.col_sum = np.zeros(shape=(60, 60))
        self.corr = 0


def detectChar(Unk_char):
    r = 500000000000
    curr_r = 500000000000
    for j in database_characters:
        curr_r = similarity = getSimilarity(Unk_char.template, j.template)
        if curr_r < r:
            Unk_char.char = j.char
            r = curr_r
    return Unk_char.char


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
        absWidthDifference = abs(Ymax - Ymin)
        if (absWidthDifference == 0):
            absWidthDifference = 1

        if (abs(Xmax - Xmin) / absWidthDifference >= 0.12 and abs(Xmax - Xmin) / absWidthDifference <= 2.9 and abs(
                Ymax - Ymin) > 11 and abs(Xmax - Xmin) > 3 and checkRange(imgCols, Xmin, Xmax)):
            rr, cc = rectangle(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=with_boxes.shape)
            with_boxes[rr, cc] = 1  # set color black
            char = img[rr, cc]
            char = rotateMatrixNTimes(char, 3)
            char = np.fliplr(char)
            rows, cols = char.shape
            char = char[1:rows - 1, 1:cols - 1]  # Cancel black borders (thickness is only 1 pixel)
            if (abs(Xmax - Xmin) <= 4 and detectDefect(char)):
                continue
            # show_images([char], ["Char before resizing"])
            char = characterContour("Unknown", img_as_ubyte(char))
            textChar = detectChar(char)
            charTexts.append(textChar)
            # show_images([char.template], ["Char"])
    charTexts.reverse()
    return charTexts


def checkInDatabase(charList, databaseName):
    if len(charList) <= 3:
        return False
    databaseFile = open(databaseName, "r", encoding='utf-8')
    charListLen = len(charList)
    for line in databaseFile:
        currLine = line.split()
        if functools.reduce(lambda x, y: x and y, map(lambda p, q: p == q, currLine, charList), True):
            return True
    return False


def getSimilarity(img1, img2):
    dim = (60, 60)
    img1 = cv.GaussianBlur(img1, (19, 19), 0)
    img2 = cv.GaussianBlur(img2, (19, 19), 0)
    img1 = cv.resize(img1, dim, interpolation=cv.INTER_AREA)
    img2 = cv.resize(img2, dim, interpolation=cv.INTER_AREA)
    ret2, img1 = cv.threshold(img1, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    ret2, img2 = cv.threshold(img2, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    sim = img1 - img2
    sim = sim * sim
    sim = np.sum(sim)
    sim = np.sqrt(sim)
    return sim


def plateDetect(img):
    originalImage = np.copy(img)
    img = imutils.resize(img, width=500)
    grayedimg = rgb2gray(img)
    img_after_orig_minus_closing = grayedimg - binary_erosion(binary_dilation(grayedimg))
    img_after_orig_minus_closing = black_tophat(grayedimg, np.ones((5, 3)))
    blackhat = black_hat_cv2(grayedimg)
    blackhat = img_as_ubyte(blackhat)
    img1 = binary_dilation(grayedimg, np.ones((3, 3)))
    img1 = binary_erosion(img1, np.ones((3, 3)))
    img_thresh = img1
    img_thresh[img1 < 50] = 0
    img_thresh[img_thresh >= 50] = 255

    light_regions = find_light_regions(grayedimg)

    sobelX_cv2 = sobel_x_cv2(blackhat)
    sobelX_skimage = sobel(img_after_orig_minus_closing)

    gau_img_sk = gaussian(sobelX_skimage, sigma=0, truncate=1 / 5)
    gau_img_cv = gaussian_cv(sobelX_cv2)
    otsu_thresh_cv = otsu_threshold(gau_img_cv)
    thresh = threshold_otsu(gau_img_sk)
    otsu_thresh_sk = gau_img_sk > thresh

    otsu_thresh_cv = cv2.erode(otsu_thresh_cv, None, iterations=2)
    otsu_thresh_cv = cv2.dilate(otsu_thresh_cv, None, iterations=3)

    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)
    otsu_thresh_sk = binary_dilation(otsu_thresh_sk)

    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)
    otsu_thresh_sk = binary_erosion(otsu_thresh_sk)

    otsu_thresh_cv = cv2.erode(otsu_thresh_cv, None, iterations=3)
    otsu_thresh_cv = cv2.dilate(otsu_thresh_cv, None, iterations=6)
    otsu_thresh_cv = cv2.erode(otsu_thresh_cv, None, iterations=8)
    otsu_thresh_cv = cv2.dilate(otsu_thresh_cv, None, iterations=3)

    otsu_thresh_cv = cv2.erode(otsu_thresh_cv, None, iterations=3)
    otsu_thresh_cv = cv2.dilate(otsu_thresh_cv, None, iterations=8)
    otsu_thresh_cv = cv2.erode(otsu_thresh_cv, None, iterations=3)

    otsu_thresh_cv = gaussian_cv(otsu_thresh_cv)
    otsu_thresh_sk = gau_img_sk = gaussian(otsu_thresh_sk, sigma=0, truncate=1 / 5)
    img_with_boxes = np.zeros(img.shape)
    contours = find_contours(otsu_thresh_cv, 0.8)
    bounding_boxes = []
    for contour in contours:
        box = [min(contour[:, 1]), max(contour[:, 1]), min(contour[:, 0]), max(contour[:, 0])]
        w = abs(box[0] - box[1])
        h = abs(box[2] - box[3])
        area = h * w
        ratio = w / h
        if ratio >= 1.5 and ratio <= 3.8:
            if w < 150 and w > 50 and h < 50 and h > 10 and box[2] - 10 > 0 and box[3] + 10 < img.shape[0]:
                bounding_boxes.append(box)
    cropped = img
    plate_list = []
    YminMapped = -1
    YmaxMapped = -1
    XminMapped = -1
    XmaxMapped = -1
    for box in bounding_boxes:
        [Xmin, Xmax, Ymin, Ymax] = box
        Xmin = int(Xmin)
        Xmax = int(Xmax)
        Ymin = int(Ymin)
        Ymax = int(Ymax)
        h = int(h)
        w = int(w)
        originalHeight = originalImage.shape[0]
        originalWidth = originalImage.shape[1]
        resizedHeight = img.shape[0]
        resizedWidth = img.shape[1]
        YminMapped = ((Ymin - 15) * originalHeight) // resizedHeight
        YmaxMapped = (Ymax * originalHeight) // resizedHeight
        XminMapped = (Xmin * originalWidth) // resizedWidth
        XmaxMapped = (Xmax * originalWidth) // resizedWidth
        img_cropped = originalImage[YminMapped:YmaxMapped, XminMapped:XmaxMapped]
        [B, G, R] = [np.sum(img_cropped[:, :, 0]), np.sum(img_cropped[:, :, 1]), np.sum(img_cropped[:, :, 2])]
        plate_list.append([img_cropped, Ymin, Ymax])
        rr, cc = rectangle(start=(Ymin, Xmin), end=(Ymax, Xmax), shape=img.shape)
        img_with_boxes[rr.astype(int), cc.astype(int)] = 1  # set color white
    choosed_plate_index = -1;
    Yhigh = 0
    x = 0;
    if len(plate_list) == 2:
        if (plate_list[0][2] + 10) > img.shape[0]:
            choosed_plate_index = 1;
        elif (plate_list[1][2] + 10) > img.shape[0]:
            choosed_plate_index = 0;
        else:
            for z, plate_instance in enumerate(plate_list):
                if plate_instance[1] > Yhigh:
                    Yhigh = plate_instance[1]
                    choosed_plate_index = x
                x = x + 1
    else:
        for z, plate_instance in enumerate(plate_list):
            if plate_instance[1] > Yhigh:
                Yhigh = plate_instance[1]
                choosed_plate_index = x
            x = x + 1
    plate_final = []
    if choosed_plate_index != -1:
        plate_final.append(plate_list[choosed_plate_index][0])
    else:
        for plate_instance in plate_list:
            plate_final.append(plate_instance[0])
    return plate_final[0]


class character:
    def __init__(self, char, template):
        self.char = char
        self.template = cv.resize(cv.imread(template, cv.IMREAD_GRAYSCALE), (60, 60), interpolation=cv.INTER_AREA)
        self.col_sum = np.zeros(shape=(60, 60))
        self.corr = 0


def runDatabase():
    width = 60
    height = 60

    alf = character("أ", '../charImages/alf.jpg')
    alf2 = character("أ", '../charImages/alf2.png')
    alf3 = character("أ", '../charImages/alf3.jpg')
    alf4 = character("أ", '../charImages/alf4.jpg')
    alf5 = character("أ", '../charImages/alf5.jpg')
    baa = character("ب", "../charImages/ba'.jpg")
    baa2 = character("ب", "../charImages/ba'2.jpg")
    baa3 = character("ب", "../charImages/ba'3.jpg")
    baa4 = character("ب", "../charImages/ba'4.jpg")
    gem = character("ج", '../charImages/gem.jpg')
    dal = character("د", '../charImages/dal.jpg')
    dal2 = character("د", '../charImages/dal2.jpg')
    raa = character("ر", "../charImages/ra'.jpg")
    raa2 = character("ر", "../charImages/ra'2.png")
    raa3 = character("ر", "../charImages/ra'3.jpg")
    raa4 = character("ر", "../charImages/ra'4.jpg")
    raa5 = character("ر", "../charImages/ra'5.jpg")
    zay = character("ز", "../charImages/zay.jpg")
    sen = character("س", '../charImages/sen.jpg')
    sen2 = character("س", '../charImages/sen2.jpg')
    sad = character("ص", '../charImages/sad.jpg')
    sad2 = character("ص", '../charImages/sad2.jpg')
    sad3 = character("ص", '../charImages/sad3.jpg')
    sad4 = character("ص", '../charImages/sad4.jpg')
    taa = character("ط", '../charImages/taa.jpg')
    taa2 = character("ط", '../charImages/taa2.jpg')
    taa3 = character("ط", '../charImages/taa3.jpg')
    ain = character("ع", '../charImages/ain.png')
    ain2 = character("ع", '../charImages/ain2.jpg')
    faa = character("ف", "../charImages/fa'.jpg")
    faa2 = character("ف", "../charImages/fa'2.png")
    faa3 = character("ف", "../charImages/fa'3.jpg")
    faa4 = character("ف", "../charImages/fa'4.jpg")
    faa5 = character("ف", "../charImages/fa'5.jpg")
    qaf = character("ق", '../charImages/qaf.jpg')
    qaf2 = character("ق", '../charImages/qaf2.jpg')
    qaf3 = character("ق", '../charImages/qaf3.jpg')
    qaf4 = character("ق", '../charImages/qaf4.jpg')
    qaf5 = character("ق", '../charImages/qaf5.jpg')
    qaf6 = character("ق", '../charImages/qaf6.jpg')
    lam = character("ل", '../charImages/lam.jpg')
    lam2 = character("ل", '../charImages/lam2.jpg')
    mem = character("م", '../charImages/mem.jpg')
    mem2 = character("م", '../charImages/mem2.jpg')
    mem3 = character("م", '../charImages/mem3.jpg')
    noon = character("ن", '../charImages/noon.png')
    noon2 = character("ن", '../charImages/noon2.png')
    noon3 = character("ن", '../charImages/noon3.jpg')
    noon4 = character("ن", '../charImages/noon4.jpg')
    haa = character("ه", "../charImages/ha'.jpg")
    haa2 = character("ه", "../charImages/ha'2.png")
    haa3 = character("ه", "../charImages/ha'3.png")
    haa4 = character("ه", "../charImages/ha'4.jpg")
    waw = character("و", '../charImages/waw.jpg')
    waw2 = character("و", '../charImages/waw2.jpg')
    yaa = character("ى", "../charImages/ya'.jpg")
    yaa2 = character("ى", "../charImages/ya'2.jpg")
    zero = character("0", '../charImages/0.jpg')
    one = character("1", '../charImages/1.jpg')
    one2 = character("1", '../charImages/11.jpg')
    one3 = character("1", '../charImages/111.jpg')
    two = character("2", '../charImages/2.jpg')
    two2 = character("2", '../charImages/22.jpg')
    three = character("3", '../charImages/3.jpg')
    three2 = character("3", '../charImages/33.jpg')
    four = character("4", '../charImages/4.jpg')
    four2 = character("4", '../charImages/44.jpg')
    five = character("5", '../charImages/5.jpg')
    five2 = character("5", '../charImages/55.jpg')
    five3 = character("5", '../charImages/555.jpg')
    six = character("6", '../charImages/6.jpg')
    seven = character("7", '../charImages/7.jpg')
    seven2 = character("7", '../charImages/77.jpg')
    seven3 = character("7", '../charImages/777.jpg')
    eight = character("8", '../charImages/8.jpg')
    eight2 = character("8", '../charImages/88.jpg')
    nine = character("9", '../charImages/9.jpg')

    database_characters.append(alf)
    database_characters.append(alf2)
    database_characters.append(alf3)
    database_characters.append(alf4)
    database_characters.append(alf5)
    database_characters.append(baa)
    database_characters.append(baa2)
    database_characters.append(baa3)
    database_characters.append(baa4)
    database_characters.append(gem)
    database_characters.append(dal)
    database_characters.append(dal2)
    database_characters.append(raa)
    database_characters.append(raa2)
    database_characters.append(raa3)
    database_characters.append(raa4)
    database_characters.append(raa5)
    database_characters.append(zay)
    database_characters.append(sen)
    database_characters.append(sen2)
    database_characters.append(sad)
    database_characters.append(sad2)
    database_characters.append(sad3)
    database_characters.append(sad4)
    database_characters.append(taa)
    database_characters.append(taa2)
    database_characters.append(taa3)
    database_characters.append(ain)
    database_characters.append(ain2)
    database_characters.append(faa)
    database_characters.append(faa2)
    database_characters.append(faa3)
    database_characters.append(faa4)
    database_characters.append(faa5)
    database_characters.append(qaf)
    database_characters.append(qaf2)
    database_characters.append(qaf3)
    database_characters.append(qaf4)
    database_characters.append(qaf5)
    database_characters.append(qaf6)
    database_characters.append(lam)
    database_characters.append(lam2)
    database_characters.append(mem)
    database_characters.append(mem2)
    database_characters.append(mem3)
    database_characters.append(noon)
    database_characters.append(noon2)
    database_characters.append(noon3)
    database_characters.append(noon4)
    database_characters.append(haa)
    database_characters.append(haa2)
    database_characters.append(haa3)
    database_characters.append(haa4)
    database_characters.append(waw)
    database_characters.append(waw2)
    database_characters.append(yaa)
    database_characters.append(yaa2)
    database_characters.append(one)
    database_characters.append(one2)
    database_characters.append(one3)
    database_characters.append(two)
    database_characters.append(two2)
    database_characters.append(three)
    database_characters.append(three2)
    database_characters.append(four)
    database_characters.append(four2)
    database_characters.append(five)
    database_characters.append(five2)
    database_characters.append(five3)
    database_characters.append(six)
    database_characters.append(seven)
    database_characters.append(seven2)
    database_characters.append(seven3)
    database_characters.append(eight)
    database_characters.append(eight2)
    database_characters.append(nine)
