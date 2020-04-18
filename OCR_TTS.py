import cv2
import argparse
from PIL import Image
import pytesseract
import numpy as np
from imutils.object_detection import non_max_suppression
import pyttsx3
#------------------------------------------------------
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

import time


start_time = time.time()

#////////////////////////////////////////////////

#file = open("data.txt", "w+")
#////////////////////////////////////////////////

def decode_predictions(scores, geometry):
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    for y in range(0, numRows):

        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        for x in range(0, numCols):

            if scoresData[x] < float(0.5):
                continue

            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]

            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)


#///////////////////////////////////////////////////////

engine.say("A4 mode press 1 and for medicine mode press 2")
engine.runAndWait()
#///////////////////////////////////////////////////////
ocr_mode = int(input('''Enter the mode of the OCR operation: A4 Papers: 1 Medicine: 2 '''))
if ocr_mode == 2:
    print("Medicine Mode activated")
    ###################################################################################
    image = cv2.imread('1.jpg', cv2.IMREAD_COLOR)
    print("Image loaded ")
    # ///////////////////////////////////////////////////////
    orig = image.copy()
    (origH, origW) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (int(320), int(320))
    rW = origW / float(newW)
    rH = origH / float(newH)

    # resize the image and grab the new image dimension2s
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet("E:/OCR_RPI_TE/frozen_east_text_detection.pb")

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)

    (scores, geometry) = net.forward(layerNames)

    # decode the predictions, then  apply non-maxima suppression to
    # suppress weak, overlapping bounding boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    final_list = []
    text_empty = ''
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scale the bounding box coordinates based on the respective
        # ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        dX = int((endX - startX) * float(0))
        dY = int((endY - startY) * float(0))
        startX = max(0, startX - dX)
        startY = max(0, startY - dY)
        endX = min(origW, endX + (dX * 2))
        endY = min(origH, endY + (dY * 2))
        roi = orig[startY:endY, startX:endX]
        ########################################################################

        text = pytesseract.image_to_string(
            roi, config="-l eng --oem 1 --psm 11")
        print("for:" + text)

        text_empty = text_empty +text + " "
    print(text_empty)
    engine.say(text_empty)
    engine.runAndWait()
    engine.stop()
    exit_loop = True
    while exit_loop:
        engine.say("repeat press 2 else press 1")
        engine.runAndWait()
        if "2" == input("choice: "):
            engine.say(text_empty)
            engine.runAndWait()
            engine.stop()
        else:
            exit_loop = False
            engine.stop()


    #############################################################################
    print("--- %s seconds ---" % (time.time() - start_time))

if ocr_mode == 1:
    print("A4 Mode")
    # ---------------------------Load Imagge---------------------------#
    img = cv2.imread('1.png', cv2.IMREAD_COLOR)
    # ---------------------------GreyScale Imagge---------------------------#
    # convert to grey to reduce detials
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # /////////////////////////////////////////////////////////////////
    # ---------------------------Filter1 Imagge---------------------------#
    gray = cv2.bilateralFilter(gray, 11, 17, 17)  # Blur to reduce noise
    # /////////////////////////////////////////////////////////////////
    # ---------------------------Thresholding Imagge---------------------------#
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # /////////////////////////////////////////////////////////////////
    # ---------------------------Result---------------------------#
    original = pytesseract.image_to_string(gray, config=' -l eng --oem 1 ')
    print(original)

    engine.say("words detected are "+original)
    engine.runAndWait()
    engine.stop()
    exit_loop = True
    while exit_loop:
        engine.say("repeat press 2 else press 1")
        engine.runAndWait()
        if "2" == input("choice: "):
            engine.say(original)
            engine.runAndWait()
            engine.stop()
        else:
            exit_loop = False
            engine.stop()
    print("--- %s seconds ---" % (time.time() - start_time))

