import cv2
import mediapipe as mp
import time
import numpy as np
import sys
from tensorflow.keras.models import load_model
import tensorflow as tf
import math

model = load_model('F:\\Doaa Mamdouh\\project cv\\vgg16.h5')


mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

#cap = cv2.VideoCapture(0)
count=0

cap = cv2.VideoCapture('F:\\Doaa Mamdouh\\project cv\\Video7.avi')
pTime = 0
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(imgRGB)
    #print(results.pose_landmarks)
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
        for id, lm in enumerate(results.pose_landmarks.landmark):
            h, w,c = img.shape
            #print(id, lm)
            cx, cy = int(lm.x*w), int(lm.y*h)
            cv2.circle(img, (cx, cy), 5, (255,0,0), cv2.FILLED)


    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    train_images = []

    cv2.putText(img, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)
    cv2.imshow("Image", img)
    cv2.imwrite("F:\\Doaa Mamdouh\\project cv\\images\\"+"frame%d.jpg" % count , img)
    image=cv2.imread('F:\\Doaa Mamdouh\\project cv\\images\\'+"frame%d.jpg" % count)
    image = cv2.resize(image, (224, 224))
    train_images.append(image)

    count+=1

    #cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

    train_images = np.array(train_images)   
    predict=((model.predict(train_images)).astype('int64').flatten())
    print(predict)
    
    





