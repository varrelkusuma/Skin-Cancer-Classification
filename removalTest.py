# Built-in imports
import sys
import os
import json
import uuid
from time import time
import tkinter as tk
from tkinter import filedialog

# third-party imports
import cv2
import numpy as np

#costum imports
from feature_extraction.GrayLevelCooccurenceMatrix import GrayLevelCooccurenceMatrix
from feature_extraction.LocalBinaryPattern import LocalBinaryPatterns
from preprocessing.Segmentation import Segmentation
from preprocessing.ObjectRemoval import ObjectRemoval

melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma\ISIC_0024586.jpg'

def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]

        #HUE, SATURATION, AND VALUE (BRIGHTNESS) RANGES. TOLERANCE COULD BE ADJUSTED.
        upper =  np.array([pixel[0] + 10, pixel[1] + 10, pixel[2] + 40])
        lower =  np.array([pixel[0] - 10, pixel[1] - 10, pixel[2] - 40])
        print(lower, upper)

        #A MONOCHROME MASK FOR GETTING A BETTER VISION OVER THE COLORS 
        image_mask = cv2.inRange(image_hsv, lower, upper)
        cv2.imshow("Mask", image_mask)

# Create Object
# glcm = GrayLevelCooccurenceMatrix()
# lbp = LocalBinaryPatterns()
obr = ObjectRemoval()

"""
global image_hsv, pixel
#OPEN DIALOG FOR READING THE IMAGE FILE
root = tk.Tk()
root.withdraw() #HIDE THE TKINTER GUI
image_src = cv2.imread(melanoma)
cv2.imshow("BGR",image_src)
#CREATE THE HSV FROM THE BGR IMAGE
image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV",image_hsv)
#CALLBACK FUNCTION
cv2.setMouseCallback("HSV", pick_color)
"""


# Hair Removal
img = cv2.imread(melanoma)
remove = obr.removeMicroscope(img)
remove2 = obr.removeHair(remove)

# Microscope Removal

thresh = obr.toThresh(remove2)
# cv2.imshow('image', img)
# cv2.imshow('hair remove', remove)
cv2.imshow('remove', thresh)


cv2.waitKey(0)
cv2.destroyAllWindows()