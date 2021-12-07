import cv2 as cv
import numpy as np
import os


def colorRecog(imgOri, color):  # 根据对方颜色进行二值化
    colorDict = {
        'red': [np.array([0, 55, 100]), np.array([80, 255, 255])],
        'blue': [np.array([0, 60, 60]), np.array([255, 255, 255])]
    }

    blur = cv.GaussianBlur(imgOri, (9, 9), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    range = cv.inRange(hsv, colorDict[color][0], colorDict[color][1])

    open = cv.morphologyEx(range, cv.MORPH_OPEN, (5, 5))
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, (5, 5))
    erode = cv.erode(close, (1, 1))
    dilate = cv.dilate(erode, (3, 3))

    dilate[700:, :] = 0
    imgOut = dilate

    return imgOut


def objectiveDetect(imgBinary, imgOri):  # 识别轮廓搜索装甲板并筛选
    h, w = imgBinary.shape

    # 初始化包含轮廓对的列表
    dataRaw = []
    dataCheck = []
    dataMatch = []
    dataMatchPro = []
    rectOut = []

    # 查找轮廓
    cnts, hier = cv.findContours(
        imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    retval = cnts is not None
    if retval:
        retval = False
        for cnt in cnts:
            dic = dict()
            if cv.contourArea(cnt) > 3396 or cv.contourArea(cnt) < 6:
                continue

            area = cv.contourArea(cnt)
            ellipse = cv.fitEllipse(cnt)
            ellCenter, ellShape, ellAngle = ellipse

            rect = cv.minAreaRect(cnt)
            coordinates = np.int0(cv.boxPoints(rect))
            # cv.polylines(img,[ coordinates], True, (255, 0, 0), 1)

            dic['area'] = area
            dic['ellCenter'] = ellCenter
            dic['ellShape'] = ellShape
            dic['ellAngle'] = ellAngle
            dic['coordinates'] = coordinates
            dataRaw.append(dic)

        for i in range(len(dataRaw)):
            height = dataRaw[i]['ellShape'][0]
            width = dataRaw[i]['ellShape'][1]
            area = dataRaw[i]['area']
            angle = dataRaw[i]['ellAngle']

            if 4 >= height/width >= 0.2 and area > 18:
                if angle < 45 or angle > 135:
                    dataCheck.append(dataRaw[i])

        dataCheck = sorted(
            dataCheck, key=lambda dataCheck: dataCheck['ellCenter'][1])

        for i in range(len(dataCheck)):
            coordinates = dataCheck[i]['coordinates']
            # cv.polylines(img, [coordinates], True, (0, 255, 255))
            # cv.putText(img, str(dataCheck[i]['ellAngle']),
            #            coordinates[0], 0, 0.5, (0, 0, 255))

            j = i+1
            while j < len(dataCheck):
                ellX1, ellY1 = dataCheck[i]['ellCenter']
                ellX2, ellY2 = dataCheck[j]['ellCenter']
                ellH1 = dataCheck[i]['ellShape'][0]
                ellH2 = dataCheck[j]['ellShape'][0]
                ellAngle1 = dataCheck[i]['ellAngle']
                ellAngle2 = dataCheck[j]['ellAngle']
                area1 = dataCheck[i]['area']
                area2 = dataCheck[j]['area']

                if abs(ellY1-ellY2) <= 2*(ellH1+ellH2):
                    if abs(ellH2-ellH1) <= max(ellH1, ellH2):
                        if abs(ellX1-ellX2) <= 6.8*(ellH2+ellH1):
                            if abs(ellAngle2-ellAngle1) < 30:
                                if abs(area2-area1) < 3*min(area1, area2):
                                    dataMatch.append(
                                        (dataCheck[i], dataCheck[j]))
                j += 1

        dataMatchPro = dataMatch.copy()
        for i in range(len(dataMatch)):
            center2 = dataMatch[i][1]['ellCenter']
            angle1 = dataMatch[i][0]['ellAngle']+0.000000001
            angle2 = dataMatch[i][1]['ellAngle']+0.000000001

            j = i+1
            if j < len(dataMatch):
                centerTest1 = dataMatch[j][0]['ellCenter']
                angleTest1 = dataMatch[j][0]['ellAngle']+0.000000001
                angleTest2 = dataMatch[j][1]['ellAngle']+0.000000001
                if centerTest1 == center2:
                    if abs(angleTest1/angleTest2-1) > abs(angle1/angle2-1):
                        dataMatchPro.pop(j)
                    else:
                        dataMatchPro.pop(i)

        if dataMatchPro is not None:
            retval = True
            for i in range(len(dataMatchPro)):
                coordinates1 = dataMatchPro[i][0]['coordinates']
                coordinates2 = dataMatchPro[i][1]['coordinates']

                # cv.polylines(img, [coordinates1], True, (255, 255, 0))
                # cv.polylines(img, [coordinates2], True, (255, 0, 255))

                matTemp = np.zeros((h, w), np.uint8)
                cv.polylines(matTemp, [coordinates1], True, 255)
                cv.polylines(matTemp, [coordinates2], True, 255)
                cv.line(matTemp, coordinates1[0], coordinates2[0], 255)
                cntTemp, hierTemp = cv.findContours(
                    matTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                rectX, rectY, rectW, rectH = cv.boundingRect(cntTemp[0])

                cv.rectangle(imgOri, (rectX, rectY),
                             (rectX+rectW, rectY+rectH), (255, 155, 0), 3)
                rectOut.append((rectX, rectY, rectW, rectH))

    return retval, rectOut
