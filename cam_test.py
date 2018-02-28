import numpy as np
import cv2
import imutils
from BodyRect import BodyRect
import os
from toolbox import *

cam= cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FPS, 1)

while True:
    ret, img=cam.read()
    img=cv2.resize(img,(340,220))
    i2 = BodyRect(img)
    if i2.quality>0.2:
        im2=i2.retrieveRectDraw()
        cv2.imshow('Image ',im2)
    cv2.waitKey(10)
    
cam.release()


