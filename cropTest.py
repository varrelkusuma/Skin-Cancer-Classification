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

# Raw Picture Folder
melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\scc\ISIC_0029582.jpg'
bcc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\bcc'
scc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\raw\scc'

# Resize Picture Folder
mel_resize = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\melanoma'
bcc_resize = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\bcc'
scc_resize = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\resize\scc'

# Threshold Picture Folder
mel_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\melanoma'
bcc_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\bcc'
scc_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\scc'

# Create Object
ip = ImageProcess()
obr = ObjectRemoval()
ce = ColorExtraction()

def nothing(x):
    pass

#object
alpha = 1.4
beta = 10

image = cv2.imread(melanoma)
resize = ip.resize(image, 90)
# cv2.namedWindow('manual')
segment = ip.resize(image, 90)

# cv2.createTrackbar('alpha', 'manual', 0, 30, nothing)
# cv2.createTrackbar('beta', 'manual', 0, 100, nothing)

"""
while(1):

    # get current positions of two trackbars
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    alpha = cv2.getTrackbarPos('alpha','manual')
    beta = cv2.getTrackbarPos('beta','manual')

    correction = ip.manualColorCorrection(resize, alpha/10, beta)
    cv2.imshow('manual', correction)
"""

yen = ip.yenThreshold(segment)
manual = ip.manualColorCorrection(segment, alpha, beta)
gray = cv2.cv2.cvtColor(manual, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 32)

# Thresh for cropping
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

x, y, w, h = cv2.boundingRect(cnt)
cv2.rectangle(segment,(x,y),(x+w,y+h),(0,255,0),2)
cv2.drawContours(segment, cnt, -1, (0, 0, 255), 2)

crop_img = manual[y:y+h, x:x+w]

# Thresh after cropping
gray = cv2.cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (17, 17), 32)

ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
th2 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11,2)
ret, thresh2 = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY_INV)


contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key=cv2.contourArea)

stats = ce.color_extraction(crop_img)
print(stats)

#cv2.putText(resize, 'raw (resized)', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(segment, 'segmented', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(yen, 'yen_color', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(manual, 'manual_color', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(crop_img, 'cropped_image', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(gray, 'gray', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(blur, 'blur', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(thresh, 'otsu_threshold', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(thresh2, 'simple_threshold', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)
#cv2.putText(th2, 'adaptive_threshold', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36,255,12), 2)

#cv2.imshow('raw', resize)
cv2.imshow('segmented', segment)
#cv2.imshow('yen_threshold', yen)
cv2.imshow('manual_correction', manual)
cv2.imshow('cropped', crop_img)
#cv2.imshow('gray', gray)
#cv2.imshow('blur', blur)
#cv2.imshow('simple_threshold', thresh2)
cv2.imshow('otsu_threshold', thresh)
#cv2.imshow('adaptive_threshold', th2)

#cv2.imwrite('raw.jpg', resize)
#cv2.imwrite('segmented.jpg', segment)
#cv2.imwrite('yen_threshold.jpg', yen)
#cv2.imwrite('manual_correction.jpg', manual)
#cv2.imwrite('cropped.jpg', crop_img)
#cv2.imwrite('gray.jpg', gray)
#cv2.imwrite('blur.jpg', blur)
#cv2.imwrite('simple_threshold.jpg', thresh2)
#cv2.imwrite('otsu_threshold.jpg', thresh)
#cv2.imwrite('adaptive_threshold.jpg', th2)

cv2.waitKey(0)
cv2.destroyAllWindows()
