# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 17:59:54 2020

@author: hp
"""


# check opencv version

import cv2

# print version number

print(cv2.__version__)

# Set working directory

import os
print("Current Working Directory " , os.getcwd())
os.chdir("F:")

# plot photo with detected faces using opencv cascade classifier

from cv2 import imread
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle

# load the photograph

pixels = imread('FD_test2.jpg')

# load the pre-trained model

classifier = CascadeClassifier('haarcascade_frontalface_default.xml')

# perform face detection

bboxes = classifier.detectMultiScale(pixels,1.05,8)

# print bounding box for each detected face

for box in bboxes:
	# extract
	x, y, width, height = box
	x2, y2 = x + width, y + height
	# draw a rectangle over the pixels
	rectangle(pixels, (x, y), (x2, y2), (0,0,255), 1)
    
# show the image

imshow('face detection', pixels)

# keep the window open until we press a key

waitKey(0)

# close the window

destroyAllWindows()
