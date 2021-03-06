import cv2 as cv
import numpy as np


def colorRecog(imgOri, color):  # 根据对方颜色进行二值化
    colorDict = {
        'red': [np.array([0, 100, 100]), np.array([55, 255, 255])],
        'blue': [np.array([70, 50, 240]), np.array([100, 80, 255])]
    }

    blur = cv.GaussianBlur(imgOri, (1, 1), 0)
    hsv = cv.cvtColor(blur, cv.COLOR_BGR2HSV)
    range = cv.inRange(hsv, colorDict[color][0], colorDict[color][1])

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, 1))
    erode = cv.erode(range, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    dilate = cv.dilate(erode, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    open = cv.morphologyEx(dilate, cv.MORPH_OPEN, kernel)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 10))
    close = cv.morphologyEx(open, cv.MORPH_CLOSE, kernel)

    imgOut = close

    return imgOut


def objectiveDetect(imgBinary):  # 识别轮廓搜索装甲板并筛选
    h, w = imgBinary.shape

    # 初始化包含轮廓对的列表
    dataRaw = []
    dataCheck = []
    dataMatch = []
    dataMatchPro = []
    rectOut = []

    # 查找轮廓
    cnts, hier = cv.findContours(
        imgBinary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    retval = cnts != ()
    if retval:
        retval = False  # 中途退出的返回值全为False
        for cnt in cnts:
            if cv.contourArea(cnt) > 50000 or cv.contourArea(cnt) < 6:
                continue

            dic = dict()
            area = cv.contourArea(cnt)
            ellipse = cv.fitEllipse(cnt)
            ellCenter, ellShape, ellAngle = ellipse

            rect = cv.minAreaRect(cnt)
            coordinates = np.int0(cv.boxPoints(rect))

            dic['rectArea'] = rect[1][0]*rect[1][1]
            dic['ellCenter'] = ellCenter
            dic['ellShape'] = sorted(ellShape)
            dic['ellAngle'] = ellAngle
            dic['coordinates'] = coordinates
            dataRaw.append(dic)

        # 筛选符合条件的轮廓
        for i in range(len(dataRaw)):
            height = dataRaw[i]['ellShape'][1]
            width = dataRaw[i]['ellShape'][0]
            area = dataRaw[i]['rectArea']
            angle = dataRaw[i]['ellAngle']

            if 10 >= height/width and area > 1000:
                if angle < 30 or angle > 150:
                    if height < 120 and width < 60:
                        dataCheck.append(dataRaw[i])

        # 对筛选出的轮廓按中心点高度排序
        dataCheck = sorted(
            dataCheck, key=lambda dataCheck: dataCheck['ellCenter'][1])

        # 查找符合条件的轮廓对
        for i in range(len(dataCheck)):
            j = i+1
            while j < len(dataCheck):
                ellX1, ellY1 = dataCheck[i]['ellCenter']
                ellX2, ellY2 = dataCheck[j]['ellCenter']
                ellH1 = dataCheck[i]['ellShape'][1]
                ellH2 = dataCheck[j]['ellShape'][1]
                ellAngle1 = dataCheck[i]['ellAngle']
                ellAngle2 = dataCheck[j]['ellAngle']
                area1 = dataCheck[i]['rectArea']
                area2 = dataCheck[j]['rectArea']

                if abs(ellY1-ellY2) <= max(ellH1, ellH2):
                    if abs(ellH2-ellH1) <= max(ellH1, ellH2):
                        if min(ellH2, ellH1) <= abs(ellX1-ellX2) <= 2*(ellH2+ellH1):
                            if abs(ellAngle2-ellAngle1) < 30:
                                if abs(area2-area1) <= min(area1, area2):
                                    dataMatch.append(
                                        (dataCheck[i], dataCheck[j]))
                j += 1

        # 对共享一个轮廓的两组轮廓对进行筛选
        dataMatchPro = dataMatch.copy()
        for i in range(len(dataMatch)):
            center2 = dataMatch[i][1]['ellCenter']
            angle1 = dataMatch[i][0]['ellAngle']+0.0000000001
            angle2 = dataMatch[i][1]['ellAngle']+0.0000000001

            j = i+1
            if j < len(dataMatch):
                centerTest1 = dataMatch[j][0]['ellCenter']
                angleTest1 = dataMatch[j][0]['ellAngle']+0.0000000001
                angleTest2 = dataMatch[j][1]['ellAngle']+0.0000000001
                if centerTest1 == center2:
                    if abs(angleTest1/angleTest2-1) > abs(angle1/angle2-1):
                        dataMatchPro[j] = 0
                    else:
                        dataMatchPro[i] = 0
        while 0 in dataMatchPro:
            dataMatchPro.remove(0)

        if dataMatchPro != []:
            for i in range(len(dataMatchPro)):
                coordinates1 = dataMatchPro[i][0]['coordinates']
                coordinates2 = dataMatchPro[i][1]['coordinates']

                # 绘制包含一个轮廓对的最小矩形
                matTemp = np.zeros((h, w), np.uint8)
                cv.polylines(matTemp, [coordinates1], True, 255)
                cv.polylines(matTemp, [coordinates2], True, 255)
                cv.line(matTemp, coordinates1[0], coordinates2[0], 255)
                cntTemp, hierTemp = cv.findContours(
                    matTemp, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                # 找到外接矩形
                rectX, rectY, rectW, rectH = cv.boundingRect(cntTemp[0])

                if rectH*1.2 < rectW < rectH*2.5:
                    retval = True
                    rectOut.append((rectX, rectY, rectW, rectH))

    return retval, rectOut
