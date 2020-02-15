import cv2
import os

class HairRemoval:

    def __init__(self, image):
        self.image = image
    
    def remove(self, image):
        # Convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Create kernel & perform blackhat filtering
        kernel = cv2.getStructuringElement(1,(17,17))
        blackhat = cv2.morphologyEx(grayscale, cv2.MORPH_BLACKHAT, kernel)

        # Create contours & inpaint
        ret, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        result = cv2.inpaint(image, thresh, 1, cv2.INPAINT_TELEA)

        return result