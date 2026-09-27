import cv2
import numpy as np
import cvzone.HandTrackingModule as ht
HandDetector = ht.HandDetector
from cvzone.ClassificationModule import Classifier
import math
import os

offset = 15
imgsize = 300

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
Classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

folder = r"C:\cv.project\Data"
os.makedirs(folder, exist_ok=True)

counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "I", "H", "J", "K", "I Love You", "Hello", "L", "M", "N", "O", "P", "Q", "R", "S", "No", "Okay", "Please", "Yes", "You", "T", "U", "V", "W", "X", "Y", "Z"]

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)

    imgoutput = img.copy()
    hands, img = detector.findHands(img, draw=True)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgwhite = np.ones((imgsize, imgsize, 3), np.uint8) * 255
        y1, y2 = max(0, y - offset), min(img.shape[0], y + h + offset)
        x1, x2 = max(0, x - offset), min(img.shape[1], x + w + offset)
        imgcrop = img[y1:y2, x1:x2]

        if imgcrop.size == 0:
            continue

        imgcropshape = imgcrop.shape
        imgwhite[0:imgcropshape[0], 0:imgcropshape[1]] = imgcrop

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgsize / h
            wcal = math.ceil(k * w)
            wcal = min(wcal, imgsize)
            imgresize = cv2.resize(imgcrop, (wcal, imgsize))
            wgap = math.ceil((imgsize - wcal) / 2)
            imgwhite[:, wgap:wgap + wcal] = imgresize
        else:
            k = imgsize / w
            hcal = math.ceil(k * h)
            hcal = min(hcal, imgsize)
            imgresize = cv2.resize(imgcrop, (imgsize, hcal))
            hgap = math.ceil((imgsize - hcal) / 2)
            imgwhite[hgap:hgap + hcal, :] = imgresize

        # Prediction ab dono cases ke baad common hai - yahi asli fix hai
        prediction, index = Classifier.getPrediction(imgwhite, draw=False)
        print(prediction, index)

        cv2.rectangle(imgoutput, (x - offset, y - offset - 50), (x - offset + 150, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)

        cv2.putText(imgoutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

        cv2.putText(imgoutput, labels[index], (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 2)

        cv2.imshow("Image", imgcrop)
        cv2.imshow("Imagewhite", imgwhite)

    cv2.imshow("Imagecrop", imgoutput)
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("program closed")
        break

cap.release()
cv2.destroyAllWindows()