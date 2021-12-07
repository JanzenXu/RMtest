import cv2 as cv
import numpy as np
import os

os.chdir('/home/janzen/桌面/GitHub/RMtest/Primary')
print(os.getcwd())

def colorRecog(imgIn, color):
    colorDict = {
        'red': [np.array([0, 55, 100]), np.array([80, 255, 255])],
        'blue': [np.array([0, 60, 60]), np.array([255, 255, 255])]
    }

    blur = cv.GaussianBlur(imgIn, (9, 9), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    range = cv.inRange(hsv, colorDict[color][0], colorDict[color][1])

    open = cv.morphologyEx(range, cv.MORPH_OPEN, (5, 5))
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, (5, 5))
    erode = cv.erode(close, (1, 1))
    dilate = cv.dilate(erode, (3, 3))

    dilate[700:, :] = 0
    imgOut = dilate

    return imgOut
