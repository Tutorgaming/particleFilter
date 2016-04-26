__author__ = 'Theppasith N'

import numpy as np
import cv2
import math


cam = cv2.VideoCapture(0)
i = 0;
while True:
    s, im = cam.read() # captures image
    cv2.imshow("Test Picture", im) # displays captured image
    # cv2.imwrite("test.bmp",im) # writes image test.bmp to disk
    k = cv2.waitKey(1) & 0xFF
    if k == 27 or k == ord('q'):
        break
    if k == ord('a'):
        i = i + 1
        cv2.imwrite("left"+str(i)+".jpg", im)