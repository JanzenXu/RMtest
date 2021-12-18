import cv2 as cv
import numpy as np
from function import colorRecog, objectiveDetect


img1 = cv.imread('RMimage/272.jpg', 1)
img2 = colorRecog(img1, 'blue')

retval, rects = objectiveDetect(img2)

if retval:
    for rect in rects:
        rectX, rectY, rectW, rectH = rect
        cv.rectangle(img1, (rectX, rectY),
                     (rectX+rectW, rectY+rectH), (0, 255, 255), 3)


cv, cv.imshow('res', img1)
cv.waitKey()
cv.destroyAllWindows()
