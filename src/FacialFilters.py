# Importing libraries
import numpy as np
import pandas as pd
import cv2
import os
import imutils as ut
from imutils import img_blend, full_eyes
import keyboard

# Directories

# Model Directory
mod_dir = "/".join(os.getcwd().split("/")[0:-1] + ['model/'])

# Image Directory
img_dir = "/".join(os.getcwd().split("/")[0:-1] + ['data/images/'])

# images
img_lips = cv2.imread(img_dir + "lips1.png")
img_beard = cv2.imread(img_dir + "beard.jpg")
img_spects = cv2.imread(img_dir +  "spectsclipart2.png")

# Importing models
face_cascade = cv2.CascadeClassifier(mod_dir+'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(mod_dir+'haarcascade_eye.xml')
mouth_casade = cv2.CascadeClassifier(mod_dir+'haarcascade_mcs_mouth.xml')

# Initiating the video capture
cap = cv2.VideoCapture(0)
while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame1 = frame.copy()
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame1,(x,y),(x+w,y+h),(255,0,0),3)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        roi_color1 = frame1[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        height = np.size(roi_color, 0)
        mouth = mouth_casade.detectMultiScale(roi_gray, 1.7, 11)

        for (ex,ey,ew,eh) in eyes:
            if ey + eh < height/2:
                cv2.rectangle(roi_color1,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

        if len(mouth) > 0:
            (lx,ly,lw,lh) = mouth[0]
            cv2.rectangle(roi_color1,(lx,ly),(lx+lw,ly+lh),(0,255,255),2)

        if len(eyes) == 2:
            X,Y,W,H = full_eyes(eyes)
            roi_rect = roi_color[Y:Y+H, X:X+W]
            roi_shape = roi_rect.shape
            img1_resize = ut.resize(img_spects, roi_shape[1], roi_shape[0])
            blend = img_blend(img1_resize, roi_rect)
            roi_color[Y:Y+H, X:X+W] = blend
            frame[y:y+h, x:x+w] = roi_color
            font = cv2.FONT_HERSHEY_SIMPLEX

        if len(mouth) > 0:
            x_m, y_m, w_m, h_m = mouth[0]
            roi_rect = roi_color[y_m:y_m+h_m, x_m:x_m+w_m]
            roi_shape = roi_rect.shape
            img3_resize = ut.resize(img_lips, roi_shape[1], roi_shape[0])
            blend = img_blend(img3_resize, roi_rect)
            roi_color[y_m:y_m+h_m, x_m:x_m+w_m] = blend
            frame[y:y+h, x:x+w] = roi_color

        '''
        if cv2.waitKey(33) == ord('l'):
            if len(mouth) > 0:
                x_m, y_m, w_m, h_m = mouth[0]
                roi_rect = roi_color[y_m:y_m+h_m, x_m:x_m+w_m]
                roi_shape = roi_rect.shape
                img3_resize = ut.resize(img_lips, roi_shape[1], roi_shape[0])
                blend = img_blend(img3_resize, roi_rect)
                roi_color[y_m:y_m+h_m, x_m:x_m+w_m] = blend
                frame[y:y+h, x:x+w] = roi_color
        '''
        if len(mouth) > 0:
            x_m, y_m, w_m, h_m = mouth[0]
            roi_rect = roi_color[y_m:y_m+h_m, x_m:x_m+w_m]
            roi_shape = roi_rect.shape
            img3_resize = ut.resize(img_lips, roi_shape[1], roi_shape[0])
            blend = img_blend(img3_resize, roi_rect)
            roi_color[y_m:y_m+h_m, x_m:x_m+w_m] = blend
            frame[y:y+h, x:x+w] = roi_color

        if cv2.waitKey(33) == ord('b'):
            b_x = x
            b_w = w
            b_y = int(2*(y+h)/3)
            b_h = h
            if b_y+b_h > 690:
                value = 690-(b_y+b_h)
                b_h = b_h - value

            beard = frame[b_y:b_y+b_h, b_x:b_x+b_w]
            beard_shape = beard.shape
            img4_resize = ut.resize(img_beard, beard_shape[1], beard_shape[0])
            blend = img_blend(img4_resize, beard)
            frame[b_y:b_y+b_h, b_x:b_x+b_w] = blend
            cv2.putText(frame,'Applying beard Filter',(300,650), font, 2,(255,255,255),2,cv2.LINE_AA)


    cv2.imshow('FilterFrame', cv2.resize(frame, (0,0), fx=0.8, fy=0.8))
    cv2.imshow('DetectionFrame', cv2.resize(frame1, (0,0), fx=0.3, fy=0.3))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
