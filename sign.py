import numpy as np
import pandas as pd
import mediapipe as mp
import cvzone.HandTrackingModule as ht
HandDetector = ht.HandDetector
import cv2
import math
import time
import os

detector = HandDetector(maxHands=1)
cap = cv2.VideoCapture(0)

offset = 20
imgsize = 300

folder = r"C:\cv.project\Data\You"

os.makedirs(folder,exist_ok=True)
counter = 0

while True:
    success , img = cap.read()
    img = cv2.flip(img ,1)

    hands , img = detector.findHands(img , draw=True)

    if hands:
        hand = hands[0]
        x , y , w , h = hand['bbox']

        imgwhite = np.ones((imgsize,imgsize,3),np.uint8)*255
        imgcrop = img[y-offset:y+h+offset , x-offset:x+w+offset]

        imgcropshape = imgcrop.shape
        imgwhite[0:imgcropshape[0],0:imgcropshape[1]] = imgcrop

        aspectRatio = h/w

        if aspectRatio > 1:

            k = imgsize/h
            wcal = math.ceil(k*w)
            imgresize = cv2.resize(imgcrop,(wcal,imgsize))
            imgresizeshape = imgresize.shape
            wgap = math.ceil((imgsize-wcal)/2)
            imgwhite[:,wgap:wcal+wgap] = imgresize

        else:
            k = imgsize/w
            hcal = math.ceil(k*h)
            imgresize = cv2.resize(imgcrop,(imgsize,hcal))
            imgresizeshape = imgresize.shape
            hgap = math.ceil((imgsize-hcal)/2)
            imgwhite[hgap:hcal+hgap,:] = imgresize


        cv2.imshow("image", imgcrop)
        cv2.imshow("imagewhite", imgwhite)

    cv2.imshow("img", img)

    key = cv2.waitKey(1)
    if key == ord('s'):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgwhite)
        print(counter)

    if key == ord('q'):
        print("program closed")
        break

cap.release()
cv2.destroyAllWindows()