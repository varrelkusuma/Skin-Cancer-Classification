import os
import sys
import cv2
import numpy as np
from skimage.segmentation import active_contour

path = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma\ISIC_0025489.jpg'

img = cv2.imread(path)
if img is None:
    print("Image file not found or unable to read")
    sys.exit()
    
img = cv2.GaussianBlur(img, (11,11),0)

cv2.imshow("Gaussian Blur", img)
cv2.waitKey(5000)