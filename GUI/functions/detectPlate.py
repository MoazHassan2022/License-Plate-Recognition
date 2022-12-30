from functions.commonfunctions import *
from skimage.measure import find_contours
from skimage.filters import gaussian
import cv2
from skimage import img_as_ubyte
from functions.func import *
import imutils
import numpy as np
from skimage.morphology import binary_erosion, binary_dilation,black_tophat
from skimage.morphology import (rectangle)
from skimage.draw import rectangle
from skimage.filters import threshold_otsu, sobel


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

    for i in range(0, 9):
        otsu_thresh_sk = binary_dilation(otsu_thresh_sk)

    for i in range(0, 8):
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
        if (ratio >= 1.5 and ratio <= 3.8):
            if (w < 150 and w > 50 and h < 50 and h > 10 and box[2] - 10 > 0 and box[3] + 10 < img.shape[0]):
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
    if (len(plate_list) == 2):
        if ((plate_list[0][2] + 10) > img.shape[0]):
            choosed_plate_index = 1;
        elif ((plate_list[1][2] + 10) > img.shape[0]):
            choosed_plate_index = 0;
        else:
            for z, plate_instance in enumerate(plate_list):
                if (plate_instance[1] > Yhigh):
                    Yhigh = plate_instance[1]
                    choosed_plate_index = x
                x = x + 1
    else:
        for z, plate_instance in enumerate(plate_list):
            if (plate_instance[1] > Yhigh):
                Yhigh = plate_instance[1]
                choosed_plate_index = x
            x = x + 1
    plate_final = []
    if (choosed_plate_index != -1):
        plate_final.append(plate_list[choosed_plate_index][0])
    else:
        for plate_instance in plate_list:
            plate_final.append(plate_instance[0])
    return plate_final[0]