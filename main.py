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
from preprocessing.Segmentation import Segmentation
from preprocessing.ObjectRemoval import ObjectRemoval

melanoma = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma'
bcc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\bcc'
scc = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\scc'
count = 1

# Create Object
glcm = GrayLevelCooccurenceMatrix()
lbp = LocalBinaryPatterns(24, 8)
obr = ObjectRemoval()

# Threshold Picture Folder
mel_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\melanoma'
bcc_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\bcc'
scc_thresh = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\thresh\scc'


"""
==========================================================================================
Object Removal Method (Output: Thresholded Image)
==========================================================================================
1. Hair Removal
2. Microscope Removal

"""
"""
print("Melanoma object removal...")
for filename in os.listdir(melanoma):
	if filename.endswith(".jpg"):
		tempfilename = melanoma+"/"+filename

		# Hair Removal
		img = cv2.imread(tempfilename)
		remove = obr.removeHair(img)
		thresh = obr.toThresh(remove)

		# Save Image
		filename = "file_%d.jpg"%count
		obr.save(thresh, mel_thresh, filename)
		count = count + 1

count = 1
print("BCC object removal...")
for filename in os.listdir(bcc):
	if filename.endswith(".jpg"):
		tempfilename = bcc+"/"+filename

		#Hair Removal
		img = cv2.imread(tempfilename)
		remove = obr.removeHair(img)
		thresh = obr.toThresh(remove)

		# Save Image
		filename = "file_%d.jpg"%count
		obr.save(thresh, bcc_thresh, filename)
		count = count + 1

count = 1
print("SCC object removal...")
for filename in os.listdir(scc):
	if filename.endswith(".jpg"):
		tempfilename = scc+"/"+filename

		#Hair Removal
		img = cv2.imread(tempfilename)
		remove = obr.removeHair(img)
		thresh = obr.toThresh(remove)

		# Save Image
		filename = "file_%d.jpg"%count
		obr.save(thresh, scc_thresh, filename)
		count = count + 1

"""
"""
==========================================================================================
Feature Extraction for Thresholded Image
==========================================================================================
1. GLCM
2. Local Binary Pattern

"""
mel_class = []
mel_glcm = []
mel_result = []
bcc_class = []
bcc_glcm = []
bcc_result = []
scc_class = []
scc_glcm = []
scc_result = []

# Gray Level Cooccurence Matrix (GLCM)
# Melanoma
print("Extracting GLCM from Melanoma...")
for filename in os.listdir(mel_thresh):
	if filename.endswith(".jpg"):
		tempfilename = mel_thresh+"/"+filename
		img = cv2.imread(tempfilename)
		matrix_cooccurence = glcm.createMatrix(img)
		result = glcm.feature_extraction(matrix_cooccurence)
		class_value = 1
		mel_glcm.append(result)
		mel_class.append(class_value)
# Create Dataframe & Concatenate GLCM Features & Class
mel_glcm_df = pd.DataFrame(np.concatenate(mel_glcm))
mel_class_series = pd.Series(mel_class)
mel_result = pd.concat([mel_glcm_df, mel_class_series], axis = 1)

# Basal Cell Carcinoma
print("Extracting GLCM from BCC...")
for filename in os.listdir(bcc_thresh):
	if filename.endswith(".jpg"):
		tempfilename = bcc_thresh+"/"+filename
		img = cv2.imread(tempfilename)
		matrix_cooccurence = glcm.createMatrix(img)
		result = glcm.feature_extraction(matrix_cooccurence)
		class_value = 2
		bcc_glcm.append(result)
		bcc_class.append(class_value)
# Create Dataframe & Concatenate GLCM Features & Class
bcc_glcm_df = pd.DataFrame(np.concatenate(bcc_glcm))
bcc_class_series = pd.Series(bcc_class)
bcc_result = pd.concat([bcc_glcm_df, bcc_class_series], axis = 1)

# Squamous Cell Carcinoma
print("Extracting GLCM from SCC...")
for filename in os.listdir(scc_thresh):
	if filename.endswith(".jpg"):
		tempfilename = scc_thresh+"/"+filename
		img = cv2.imread(tempfilename)
		matrix_cooccurence = glcm.createMatrix(img)
		result = glcm.feature_extraction(matrix_cooccurence)
		class_value = 3
		scc_glcm.append(result)
		scc_class.append(class_value)
# Create Dataframe & Concatenate GLCM Features & Class
scc_glcm_df = pd.DataFrame(np.concatenate(scc_glcm))
scc_class_series = pd.Series(scc_class)
scc_result = pd.concat([scc_glcm_df, scc_class_series], axis = 1)

# Exporting as csv file
out = pd.concat([mel_result, bcc_result, scc_result])
out.drop(out.columns[0], axis=1)
out.to_csv('glcm_out.csv', index=False, header=None)

cv2.waitKey(0)
cv2.destroyAllWindows()