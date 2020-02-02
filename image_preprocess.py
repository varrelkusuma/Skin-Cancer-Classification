import os
import sys
import cv2
import numpy as np
from skimage.segmentation import active_contour

path = r'C:\Users\ravee\Jupyter\Skin-Cancer-Classification\images\melanoma\ISIC_0025489.jpg'

class skinLesion:

def preprocess(self):
    """
    Validate the image and preprocess the image by applying smoothing
    filter and color transformation.
    :return: True if succeeded else None
    """
    try:
        if self.original_image is None:
            self.isImageValid = False
            return
        if self.original_image.shape[2] != 3:
            self.isImageValid = False
            return
        # morphological closing
        self.image = self.original_image.copy()
        # blur image
        self.image = cv2.GaussianBlur(self.image, (5, 5), 0)
        # Applying CLAHE to resolve uneven illumination
        # hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        # hsv[:, :, 2] = clahe.apply(hsv[:, :, 2])
        # self.image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # kernel = np.ones((11, 11), np.uint8)
        # for i in range(self.image.shape[-1]):
        #     self.image[:, :, i] = cv2.morphologyEx(
        #         self.image[:, :, i],
        #         cv2.MORPH_CLOSE, kernel)
        self.hsv_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.contour_image = np.copy(self.original_image)
        self.isImageValid = True
        # Mednode dataset related params
        # if "mednode" in self.file_path:
        #     self.real_diamter_pixels_mm = (104 * 7360) // (
        #                 330 * 24)  # pixels/mm
        if self.iterations in range(3):
            temp = self.iter_colors[self.iterations]
            self.iter_colors.remove(self.iter_colors[self.iterations])
            self.iter_colors.append(temp)
        return True
    except:
        print("error")
        self.isImageValid = False
        return

img = cv2.imread(path)
if img is None:
    print("Image file not found or unable to read")
    sys.exit()
    
img = cv2.GaussianBlur(img, (11,11),0)

cv2.imshow("Gaussian Blur", img)
cv2.waitKey(5000)