import numpy as np
from skimage import feature
from skimage.feature import greycomatrix, greycoprops

# Class Definition
class GrayLevelCooccurenceMatrix:

    def __init__(self, matrix_coocurrence):
        self.matrix_coocurrence = matrix_coocurrence

    def contrast_feature(self, matrix_coocurrence):
    	contrast = greycoprops(matrix_coocurrence, 'contrast')
    	return contrast

    def dissimilarity_feature(self, matrix_coocurrence):
    	dissimilarity = greycoprops(matrix_coocurrence, 'dissimilarity')	
    	return dissimilarity

    def homogeneity_feature(self, matrix_coocurrence):
    	homogeneity = greycoprops(matrix_coocurrence, 'homogeneity')
    	return homogeneity

    def energy_feature(self, matrix_coocurrence):
    	energy = greycoprops(matrix_coocurrence, 'energy')
    	return energy

    def correlation_feature(self, matrix_coocurrence):
    	correlation = greycoprops(matrix_coocurrence, 'correlation')
    	return correlation

    def asm_feature(self, matrix_coocurrence):
    	asm = greycoprops(matrix_coocurrence, 'ASM')
    	return asm