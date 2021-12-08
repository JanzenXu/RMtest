import cv2 as cv
import numpy as np

from function import colorRecog, objectiveDetect

video = cv.VideoCapture('RMvideo/1.mp4')

cap, frame = video.read()
h, w, d = frame.shape

cv.namedWindow('res', 0)
cv.resizeWindow('res', (w-200, h-200))

while video.isOpened():
    cap, frame = video.read()
    if cap:
        frameBinary = colorRecog(frame, 'red')
        retval, rects = objectiveDetect(frameBinary)

        if retval:
            for rect in rects:
                rectX, rectY, rectW, rectH = rect
                cv.rectangle(frame, (rectX, rectY),
                             (rectX+rectW, rectY+rectH), (255, 155, 0), 3)

        cv.imshow('res', frame)
        print(retval)

        if cv.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

cv.destroyAllWindows()
video.release()
