import cv2 as cv
import numpy as np

from function import colorRecog, objectiveDetect, rectShift

cv.namedWindow('res', 0)
cv.resizeWindow('res', 1000, 700)

video = cv.VideoCapture('RMvideo/red1.mp4')

while video.isOpened:
    cap, frame = video.read()
    if cap:
        binary = colorRecog(frame, 'red')
        ret, pts, center = objectiveDetect(binary, 0)
        if ret:
            if center:
                pts = rectShift(pts, center, 20)
            cv.polylines(frame, [pts], True, (0, 255, 0), 5)

        cv.imshow('res', frame)

        if cv.waitKey(5) & 0xff == 27:
            break
    else:
        break

cv.destroyAllWindows()
video.release()
