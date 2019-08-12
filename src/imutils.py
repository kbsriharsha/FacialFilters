# Importing libraries
import numpy as np
import pandas as pd
import cv2
import os

## Resizing Images
def resize(img, width, height, interpolation=cv2.INTER_AREA):
    return cv2.resize(img, (width, height), interpolation)

## For viewing images
def imshow(x):
    cv2.imshow("image", x)
    cv2.waitKey(5000) # 5 sec delay before image window closes
    cv2.destroyWindow("image")

## For blending two same size images
def img_blend(foreground, background):
    _,alpha = cv2.threshold(foreground, 127, 255, cv2.THRESH_BINARY_INV) #Outside black
    alpha = alpha.astype(float)/255
    foreground = cv2.multiply(alpha, foreground/255)
    background = cv2.multiply(1.0 - alpha, background/255)
    outImage = cv2.add(foreground, background)
    return outImage*255

## For drawing the rectangular box around the the
def full_eyes(eyes):
    eye1 = eyes[0]
    eye2 = eyes[1]
    x1,y1,w1,h1 = eye1
    x2,y2,w2,h2 = eye2
    Y = np.min([y1,y2])-5
    X = np.min([x1,x2])-5
    H = np.max([h1,h2]) +5
    W = np.max([x1,x2]) - X + np.max([w1,w2]) + 5
    return X,Y,W,H

if __name__ == "__main__":
    mod_dir = "/".join(os.getcwd().split("/")[0:-1] + ['model/'])
    img_dir = "/".join(os.getcwd().split("/")[0:-1] + ['data/images/'])
    img1 = cv2.imread(img_dir +  "images.png")
    img2 = cv2.imread(img_dir + "face2.jpeg")
    img3 = cv2.imread(img_dir + "lips1.png")
    img1_resize = resize(img1,310,115)
    img2_roi = img2[168:168+105+10, 106:106+300+10]
    blend = img_blend(img1_resize,img2_roi)
    img2[168:168+105+10, 106:106+300+10, :] = blend*255
    #imshow(img2)
    #print(img2.shape)
    #print(img3.shape)
    #imshow(blend)
    # Import beard
    img4 = cv2.imread(img_dir + "beard.jpg")
    print(img4.shape)
    img2_roi = img2[168:168+105+10, 106:106+124]
    img4_resize = resize(img4,115,230)
    #imshow(img4_resize)
    blend = img_blend(img4_resize,img2_roi)
    imshow(blend)
