import cv2 as cv
import numpy as np

from function import redDetect

video = cv.VideoCapture('RMvideo/2.mp4')

while video.isOpened:
    cap, frame = video.read()
    if cap:
        binary = redDetect(frame)

        cnts, hier = cv.findContours(
            binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if cnts != []:
            bestCnt = max(cnts, key=cv.contourArea)
            cv.drawContours(frame, bestCnt, -1, (0, 255, 0), 5)
            
        cv.imshow('res', frame)

        if cv.waitKey(1) & 0xff == 27:
            break
    else:
        break

cv.destroyAllWindows()
video.release()
