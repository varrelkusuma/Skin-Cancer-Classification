# Built-in imports
import sys
import os
import json
import uuid
from time import time

# third-party imports
import cv2
import numpy as np
import pandas as pd
from skimage import feature
from skimage.feature import greycomatrix, greycoprops
from skimage import io, color, img_as_ubyte, img_as_float

#costum imports
from feature_extraction.GrayLevelCooccurenceMatrix import GrayLevelCooccurenceMatrix
from feature_extraction.LocalBinaryPattern import LocalBinaryPatterns
from feature_extraction.ColorExtraction import ColorExtraction
from preprocessing.ImageProcess import ImageProcess
from preprocessing.ObjectRemoval import ObjectRemoval
from preprocessing.Segmentation import Segmentation

saveloc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images'
melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\melanoma\ISIC_0000142.jpg'
bcc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\bcc\ISIC_0024332.jpg'
scc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\scc\ISIC_0031659.jpg'

#thresh_img = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\melanoma\file_3.jpg'
#cropped = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\melanoma\file_3.jpg'
alpha = 1.4
beta = 10
count = 1

ip = ImageProcess()
obr = ObjectRemoval()
sg = Segmentation()
ce = ColorExtraction()
glcm = GrayLevelCooccurenceMatrix()
lbp = LocalBinaryPatterns(24, 8)
"""
image = cv2.imread(melanoma)
remove = obr.removeHair(image)
resize = ip.resize(remove, 80)
color_correction = ip.manualColorCorrection(resize, alpha, beta)
cropped = sg.cropRect(color_correction)

cv2.imshow("resize", resize)

#img = ip.resize(image, 100)

thresh = obr.toThresh(cropped)
thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)

masked = cv2.bitwise_and(cropped, thresh)
mean, stdev = cv2.meanStdDev(image)

h, w, c = cropped.shape

print(cropped[20][20])

red = []
green = []
blue = []
for i in range(h):
	for j in range(w):
		pixels = cropped[i][j]
		if ( (pixels[0] > 0) and (pixels[1] > 0) and (pixels[2] > 0) ):
			red.append(pixels[0])
			green.append(pixels[1])
			blue.append(pixels[2])

#print(len(red))
#print(green)
#print(blue)

#print(masked.shape)
#print(mean[1])
#print(stdev[1])


#print(cropped.shape)
#print(thresh.shape)
#print(thresh.dtype)
#print(thresh.dtype)

#cv2.imshow("thresh", thresh)
#cv2.imshow("cropped", cropped)
#masked = ip.color_mask(cropped, thresh)

#data = ce.color_extraction(img)
#comatrix = glcm.createMatrix(thresh_image)
#result = glcm.feature_extraction(comatrix)

#cv2.imshow("masked", masked)
#print(lbp_rep.shape)
#print(data)


#cv2.imshow("raw", img)
"""

image = cv2.imread(scc)
img = ip.resize(image, 60)
grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
kernel = cv2.getStructuringElement(1,(17,17))
blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)
ret, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
result = cv2.inpaint(img, thresh, 1, cv2.INPAINT_TELEA)

color_correction = ip.manualColorCorrection(result, alpha, beta)
cropped = sg.cropRect(color_correction)
if cropped.size < 1:
	count = count + 1
else:
	thresh_save = obr.toThresh(cropped)
	count = count + 1

gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray_crop, (17, 17), 32)

# Thresh for cropping
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
cv2.drawContours(img, cnt, -1, (0, 0, 255), 2)

thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2RGB)
#masked = ip.color_mask(cropped, thresh)

cv2.imshow("cropped", cropped)
cv2.imshow("gray", gray_crop)

obr.save(gray_crop, saveloc, "gray-03.jpg")

#cv2.imshow("thresh", thresh_save)
#cv2.imshow("masked", masked)


#cv2.putText(img, 'segmented', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#v2.imshow("segmented", img)

#cv2.imshow("blur", blur)
#cv2.imshow("color_correction", color_correction)
#cv2.imshow("thresh", thresh)
#cv2.imshow("result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()