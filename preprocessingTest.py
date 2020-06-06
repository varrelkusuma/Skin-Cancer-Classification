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

# Create Object
sg = Segmentation()

# Defining ROI
melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma\ISIC_0009992.jpg'
image = cv2.imread(melanoma)
roi = sg.defineROI(image)
cv2.imshow("ROI", roi)

cv2.waitKey(0)
cv2.destroyAllWindows()