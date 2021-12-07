import cv2 as cv
import numpy as np
import os

from function import colorRecog

video = cv.VideoCapture('RMvideo/1.mp4')

cap, frame = video.read()
h, w, d = frame.shape

cv.namedWindow('res', 0)
cv.resizeWindow('res', (w-200, h-200))

while video.isOpened():
    cap, frame = video.read()
    if cap:
        frameRed=colorRecog(frame,'red')
