import cv2 as cv
import numpy as np


def nothing(x):
    pass


def createTrackbar(color):
    if color == 'blue':
        cv.createTrackbar('minH', 'Adjust', 70, 255, nothing)
        cv.createTrackbar('maxH', 'Adjust', 100, 255, nothing)
        cv.createTrackbar('minS', 'Adjust', 50, 255, nothing)
        cv.createTrackbar('maxS', 'Adjust', 80, 255, nothing)
        cv.createTrackbar('minV', 'Adjust', 240, 255, nothing)
        cv.createTrackbar('maxV', 'Adjust', 255, 255, nothing)

    if color == 'red':
        cv.createTrackbar('minH', 'Adjust', 0, 255, nothing)
        cv.createTrackbar('maxH', 'Adjust', 55, 255, nothing)
        cv.createTrackbar('minS', 'Adjust', 150, 255, nothing)
        cv.createTrackbar('maxS', 'Adjust', 255, 255, nothing)
        cv.createTrackbar('minV', 'Adjust', 100, 255, nothing)
        cv.createTrackbar('maxV', 'Adjust', 255, 255, nothing)

    cv.createTrackbar('erode', 'Adjust', 1, 30, nothing)
    cv.createTrackbar('dilate', 'Adjust', 10, 30, nothing)
    cv.createTrackbar('open', 'Adjust', 10, 30, nothing)
    cv.createTrackbar('close', 'Adjust', 10, 30, nothing)


def hsvChange(imgIn):
    minH = cv.getTrackbarPos('minH', 'Adjust')
    maxH = cv.getTrackbarPos('maxH', 'Adjust')
    minS = cv.getTrackbarPos('minS', 'Adjust')
    maxS = cv.getTrackbarPos('maxS', 'Adjust')
    minV = cv.getTrackbarPos('minV', 'Adjust')
    maxV = cv.getTrackbarPos('maxV', 'Adjust')

    hsv = cv.cvtColor(imgIn, cv.COLOR_BGR2HSV)
    lowerRange = np.array([minH, minS, minV])
    upperRange = np.array([maxH, maxS, maxV])

    imgOut = cv.inRange(hsv, lowerRange, upperRange)

    return imgOut


def morphOpen(imgBinary, x, y):
    if x == 0:
        imgOut = imgBinary
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
        imgOut = cv.morphologyEx(imgBinary, cv.MORPH_OPEN, kernel)
    return imgOut


def morphClose(imgBinary, x, y):
    if x == 0:
        imgOut = imgBinary
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
        imgOut = cv.morphologyEx(imgBinary, cv.MORPH_CLOSE, kernel)
    return imgOut


def morphErode(imgBinary, x, y):
    if x == 0:
        imgOut = imgBinary
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
        imgOut = cv.erode(imgBinary, kernel)
    return imgOut


def morphDilate(imgBinary, x, y):
    if x == 0:
        imgOut = imgBinary
    else:
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (x, y))
        imgOut = cv.dilate(imgBinary, kernel)
    return imgOut


def morphology(imgBinary):
    open = cv.getTrackbarPos('open', 'Adjust')
    close = cv.getTrackbarPos('close', 'Adjust')
    erode = cv.getTrackbarPos('erode', 'Adjust')
    dilate = cv.getTrackbarPos('dilate', 'Adjust')

    imgErode = morphErode(imgBinary, erode, erode)
    imgDilate = morphDilate(imgErode, dilate, dilate)
    imgOpen = morphOpen(imgDilate, open, open)
    imgClose = morphClose(imgOpen, close, close)

    imgOut = imgClose
    return imgOut


img1 = cv.imread('RMimage/678.jpg', 1)
h, w, d = img1.shape

cv.namedWindow('Adjust', 0)
cv.resizeWindow('Adjust', w, h)
createTrackbar('blue')
img1 = cv.GaussianBlur(img1, (1, 1), 0)

img2 = img1.copy()
while True:
    imgMix = np.hstack((img1, img2))
    cv.imshow('Adjust', imgMix)
    if cv.waitKey(1) & 0xff == 27:
        break
    img2 = hsvChange(img1)
    img2 = morphology(img2)
    img2 = cv.cvtColor(img2, cv.COLOR_GRAY2BGR)

cv.destroyAllWindows()
