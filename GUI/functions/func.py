import cv2
import numpy as np
import matplotlib.image as mpimg
from functions.printMessages import *
import traceback
import logging

def sobel_x_cv2(img):
    sobelX = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    sobelX = np.absolute(sobelX)
    # Scaling
    (minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
    sobelX = (255 * ((sobelX - minVal) / (maxVal - minVal))).astype("uint8")
    return sobelX


def black_hat_cv2(img):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)
    return blackhat


def gaussian_cv(img):
    img = cv2.GaussianBlur(img, (9, 9), 0)
    return img


def otsu_threshold(img):
    threshold_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold_img


def find_light_regions(img):
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(img, cv2.MORPH_CLOSE, squareKernel)
    light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
    return light


def bitwise_and(img1, img2):
    out = cv2.bitwise_and(img1, img1, mask=img2)
    return out

def readImage(filename):
    try:
        return mpimg.imread(filename).astype(np.uint8)[:, :, :3]
    except Exception as e:
        logging.error(traceback.format_exc())
        printCritical("Error in processing the image")
        return []