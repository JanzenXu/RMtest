import cv2 as cv
import numpy as np

from function import colorRecog, objectiveDetect, rectShift

cv.namedWindow('res', 0)
cv.resizeWindow('res', 1000, 700)

video = cv.VideoCapture('RMvideo/blue1.mp4')

while video.isOpened:
    cap, frame = video.read()
    if cap:
        # 二值化函数color参数与视频文件名相同
        binary = colorRecog(frame, 'blue1')
        # 目标检测函数color参数，文件为red1和blue1时为0，其余为1
        ret, pts, center = objectiveDetect(binary, 0)
        if ret:
            if center:
                # 提前量角度可以根据实际情况计算得出
                angle = 30
                pts = rectShift(pts, center, angle)
            cv.polylines(frame, [pts], True, (0, 255, 0), 5)

        cv.imshow('res', frame)

        if cv.waitKey(5) & 0xff == 27:
            break
    else:
        break

cv.destroyAllWindows()
video.release()
