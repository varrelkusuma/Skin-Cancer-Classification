# Built-in imports
import sys
import os
import json
from time import time

# third-party imports
import cv2
import numpy as np

#costum imports
import skinLesion as sl

path = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma'
imagepath = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma\ISIC_0025489.jpg'

#image = sl.skinLesion()
#image.init(path)
#image.preprocess()

raw = cv2.imread(imagepath)
img = cv2.imread(imagepath)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 32)
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
#thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = max(contours, key=cv2.contourArea)

#mask = np.zeros(img.shape,np.uint8)
#cv2.drawContours(mask,cnt,-1)
"""
if len(cnt) > 4:
	ellipse = cv2.fitEllipse(cnt)
	x,y,w,h = cv2.boundingRect(cnt)
	area = hsv[int(y+(0.3*h)):int(y+(0.8*h)),int((0.2*w)+x):int((0.7*w)+x)]
	ellipse_cnt = cv2.ellipse2Poly( (int(ellipse[0][0]),int(ellipse[0][1]) ) ,( int(ellipse[1][0]),int(ellipse[1][1]) ),int(ellipse[2]),0,360,1)
	comp = cv2.matchShapes(cnt,ellipse_cnt,1,0.0)
	variance = cv2.meanStdDev(area)
	print(comp)
	print(variance[1])
	cv2.ellipse(img, ellipse, (0,255,0), 2)
"""

x,y,w,h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
cv2.putText(img, 'skin_lesion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)

# cv2.imshow('raw', raw)
# cv2.imshow('blur', blur)
cv2.imshow('frame',img)
cv2.imshow('thresh',thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()