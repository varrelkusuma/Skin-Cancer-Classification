import numpy as np
import pandas as pd
import cv2

# Class Definition
class ColorExtraction:

    def __init__(self):
        print("Initializing Color Extraction...")

    def color_extraction(self, image):
        
        # Attention: This algorithm processed RGB color space
        mean, stdev = cv2.meanStdDev(image)
        stats = np.concatenate([mean, stdev]).flatten()

        return stats
