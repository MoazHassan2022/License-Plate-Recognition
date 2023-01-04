

import skimage.io as io
import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import bar
from skimage.color import rgb2gray,rgb2hsv
import cv2
# Convolution:
from scipy.signal import convolve2d
from scipy import fftpack
import math

from skimage.util import random_noise
from skimage.filters import median,gaussian
from skimage.feature import canny

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import imutils

from skimage.filters import sobel_h, sobel, sobel_v,roberts, prewitt

def sobel_x_cv2(img):
    sobelX = cv2.Sobel(img,ddepth=cv2.CV_32F,dx = 1, dy = 0, ksize = -1)
    sobelX = np.absolute(sobelX)
    #Scaling
    (minVal, maxVal) = (np.min(sobelX), np.max(sobelX))
    sobelX = (255 * ((sobelX - minVal) / (maxVal - minVal))).astype("uint8")
    return sobelX
def black_hat_cv2(img):
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, rectKernel)
    return blackhat

def gaussian_cv(img):
    img=cv2.GaussianBlur(img, (9, 9), 0)
    return img
def otsu_threshold(img):
    threshold_img=cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return threshold_img
def find_light_regions(img):
    squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    light = cv2.morphologyEx(img, cv2.MORPH_CLOSE, squareKernel)
    light = cv2.threshold(light, 50, 255, cv2.THRESH_BINARY)[1]
    return light
def bitwise_and(img1,img2):
    out = cv2.bitwise_and(img1, img1, mask=img2)
    return out

