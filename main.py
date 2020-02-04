# Built-in imports
import sys
import os
import json
import uuid
from time import time

# third-party imports
import cv2
import numpy as np

#costum imports
import skinLesion as sl


melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma'
bcc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\bcc'
d = 1
e = 1

# Digital Hair Filtering


# Image Read, Threshold and Save
for filename in os.listdir(melanoma):
	if filename.endswith(".jpg"):
		tempfilename = melanoma+"/"+filename
		# raw = cv2.imread(tempfilename)
		img = cv2.imread(tempfilename)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (17, 17), 32)
		ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours, key=cv2.contourArea)

		# Draw Contours
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
		cv2.putText(img, 'skin_lesion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)

		# Image View
		# cv2.imshow('raw', raw)
		# cv2.imshow('blur', blur)
		# cv2.imshow('frame',img)
		# cv2.imshow('thresh',thresh)

		savepath = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\melanoma'
		filename = "file_%d.jpg"%d
		cv2.imwrite(os.path.join(savepath, filename), thresh)
		d = d + 1

for filename in os.listdir(bcc):
	if filename.endswith(".jpg"):
		tempfilename = bcc+"/"+filename
		# raw = cv2.imread(tempfilename)
		img = cv2.imread(tempfilename)
		hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		blur = cv2.GaussianBlur(gray, (17, 17), 32)
		ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
		contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		cnt = max(contours, key=cv2.contourArea)

		# Draw Contours
		x,y,w,h = cv2.boundingRect(cnt)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
		cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)
		cv2.putText(img, 'skin_lesion', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)

		# Image View
		# cv2.imshow('raw', raw)
		# cv2.imshow('blur', blur)
		# cv2.imshow('frame',img)
		# cv2.imshow('thresh',thresh)

		savepath = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\bcc'
		filename = "file_%d.jpg"%e
		cv2.imwrite(os.path.join(savepath, filename), thresh)
		e = e + 1

cv2.waitKey(0)
cv2.destroyAllWindows()