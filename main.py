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
# Create Object
obr = ObjectRemoval()

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

1. GLCM
2. Local Binary Pattern

==========================================================================================


==========================================================================================

Gray-Level Cooccurence Matrix (GLCM) Implementation

Steps done in the process:
1. Create empty matrix to contain all data extracted using GLCM
2. Loop every images in folder defined for this project (Melanoma, BCC, SCC)
3. Append all the data in defined matrix (6 elements from GLCM)
4. Create class series to contain image and data identifier
5. Create supervector for all this data

==========================================================================================
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

# Create object
glcm = GrayLevelCooccurenceMatrix()

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

"""
==========================================================================================

Local Binary Pattern (LBP) Implementation

Steps done in the process:
1. Create empty matrix to contain all data extracted using LBP
2. Loop every images in folder defined for this project (Melanoma, BCC, SCC)
3. Append all data (histogram) in the defined matrix
4. Create supervector for all the data

==========================================================================================

"""

# Create Object & Variable
lbp = LocalBinaryPatterns(24, 8)
mel_data = []
bcc_data = []
scc_data = []
lbp_out = []

# Melanoma
print("Extracting LBP from Melanoma...")
for filename in os.listdir(mel_thresh):
	if filename.endswith(".jpg"):
		tempfilename = mel_thresh+"/"+filename
		img = cv2.imread(tempfilename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		hist = lbp.describe(gray)
		reshaped = np.reshape(hist, (1,26))
		mel_data.append(reshaped)

# Basal Cell Carcinoma
print("Extracting LBP from BCC...")
for filename in os.listdir(bcc_thresh):
	if filename.endswith(".jpg"):
		tempfilename = bcc_thresh+"/"+filename
		img = cv2.imread(tempfilename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		hist = lbp.describe(gray)
		reshaped = np.reshape(hist, (1,26))
		bcc_data.append(reshaped)

# Squamous Cell Carcinoma
print("Extracting LBP from SCC...")
for filename in os.listdir(scc_thresh):
	if filename.endswith(".jpg"):
		tempfilename = scc_thresh+"/"+filename
		img = cv2.imread(tempfilename)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		hist = lbp.describe(gray)
		reshaped = np.reshape(hist, (1,26))
		scc_data.append(reshaped)

mel_lbp_df = pd.DataFrame(np.concatenate(mel_data))
bcc_lbp_df = pd.DataFrame(np.concatenate(bcc_data))
scc_lbp_df = pd.DataFrame(np.concatenate(scc_data))


"""
==========================================================================================

Exporting all the defined data (GLCM, LBP, Color) to a single matrix
The order of the supervector is as defined
1. (8 column) Color
2. (X column) LBP
3. (24 column) GLCM

==========================================================================================
"""

# Exporting as csv file
glcm_out = pd.concat([mel_result, bcc_result, scc_result])
lbp_out = pd.concat([mel_lbp_df, bcc_lbp_df, scc_lbp_df])
# glcm_out.drop(glcm_out.columns[0], axis=1)

glcm_out.reset_index(drop=True, inplace=True)
lbp_out.reset_index(drop=True, inplace=True)
out = pd.concat([lbp_out, glcm_out], axis = 1)

glcm_out.to_csv('glcm.csv', index=False, header=None)
lbp_out.to_csv('lbp.csv', index=False, header=None)
out.to_csv('data.csv', index=False, header=None)


cv2.waitKey(0)
cv2.destroyAllWindows()