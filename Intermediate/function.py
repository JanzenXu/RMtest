import cv2 as cv
import numpy as np


def colorRecog(imgOri, color):  # 根据目标颜色进行二值化
    colorDict = {
        'red1': [np.array([0, 0, 240]), np.array([160, 160, 255])],
        'blue1': [np.array([0, 0, 240]), np.array([160, 160, 255])],
        'red2': [np.array([0, 60, 60]), np.array([20, 255, 255])],
        'blue2': [np.array([0, 150, 70]), np.array([160, 255, 255])]
    }

    blur = cv.GaussianBlur(imgOri, (13, 13), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    range = cv.inRange(hsv, colorDict[color][0], colorDict[color][1])

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    erode = cv.erode(range, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    dilate = cv.dilate(erode, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel)

    imgOut = close.copy()
    imgOut[:, :400] = 0
    if color == 'red1' or color == 'blue1':
        imgOut[950:, :] = 0
    else:
        imgOut[500:, :] = 0

    return imgOut


def objectiveDetect(binary, color):  # 检测目标矩形
    minCenterArea = [500, 300]
    maxCenterArea = [1500, 600]
    minCntArea = [5000, 750]
    maxCntArea = [40500, 20000]
    maxRectArea = [8000, 1000]
    minRectScale = [0.6, 0.4]
    maxRectScale = [1.6, 2.5]

    cnts, hier = cv.findContours(binary, cv.RETR_CCOMP, cv.CHAIN_APPROX_NONE)

    rects = []
    betterCnts = []
    objectiveRect = []
    center = []

    retval = cnts != []
    if retval:
        retval = False
        for i in range(len(cnts)):
            area = cv.contourArea(cnts[i])
            if area > minCenterArea[color]:
                # 利用面积查找能量开关的中心
                if area < maxCenterArea[color]:
                    x, y, w, h = cv.boundingRect(cnts[i])
                    center = (x+int(w/2), y+int(h/2))

                # 筛选面积合适的轮廓
                if minCntArea[color] < area < maxCntArea[color]:
                    rect = cv.minAreaRect(cnts[i])
                    rectCenter, rectShape, rectAngle = rect
                    coordinates = np.int0(cv.boxPoints(rect))

                    dic = dict()
                    dic['area'] = area
                    dic['rectC'] = rectCenter
                    dic['rectW'] = rectShape[0]
                    dic['rectH'] = rectShape[1]
                    dic['coordinates'] = coordinates
                    dic['index'] = hier[0][i]

                    betterCnts.append(dic)

        # 查找矩形
        for i in range(len(betterCnts)):
            rectW = betterCnts[i]['rectW']
            rectH = betterCnts[i]['rectH']
            rectArea = betterCnts[i]['area']
            rectIdx = betterCnts[i]['index']

            if minRectScale[color] < rectH/rectW < maxRectScale[color]:
                if rectArea < maxRectArea[color]:
                    # 对搜索到的矩形查找它的父轮廓
                    betterCnts[i]['upperCnt'] = cnts[rectIdx[3]]
                    rects.append(betterCnts[i])

        if rects != []:
            retval = True
            # 取父轮廓面积最小的矩形为目标
            rects = sorted(rects, key=lambda x: cv.contourArea(x['upperCnt']))
            objectiveRect = rects[0]['coordinates']

    return retval, objectiveRect, center


def rectShift(pts, center, angle=0):  # 根据提前量移动矩形框
    centerX, centerY = center
    rectOut = []
    for pt in pts:
        # 转换坐标系
        x = np.float32(pt[0])-centerX
        y = np.float32(pt[1])-centerY

        # 利用极坐标转换确定移动后矩形框坐标
        rho, theta = cv.cartToPolar(x, y, angleInDegrees=True)
        theta += angle

        x, y = cv.polarToCart(rho, theta, angleInDegrees=True)
        x = int((np.int0(x)+centerX)[0])
        y = int((np.int0(y)+centerY)[0])
        rectOut.append((x, y))

    rectOut = np.array(rectOut)

    return rectOut
