import cv2 as cv
import numpy as np


def redDetect(imgIn):
    lowerRange = np.array([160, 45, 0])
    upperRange = np.array([255, 255, 200])

    blur = cv.GaussianBlur(imgIn, (13, 13), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    range = cv.inRange(hsv, lowerRange, upperRange)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    erode = cv.erode(range, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    dilate = cv.dilate(erode, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (15, 15))
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel)

    return close