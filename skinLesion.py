# Built-in imports
import sys
import os
import json
from time import time

# third-party imports
import cv2
import numpy as np

class skinLesion:

    def init(self, file_path, iterations=3):
        """
        Initiate the program by reading the lesion image from the file_path.
        :param file_path:
        :param iterations:
        """
        self.file_path = file_path
        self.base_file, _ = os.path.splitext(file_path)
        self.original_image = cv2.imread(file_path)
        self.image = None
        self.segmented_img = None
        self.hsv_image = None
        self.contour_binary = None
        self.contour_image = None
        self.contour_mask = None
        self.warp_img_segmented = None
        self.color_contour = None
        self.asymmetry_vertical = None
        self.asymmetry_horizontal = None
        self.results = None
        self.value_threshold = 150
        self.iterations = int(iterations)

        # dataset related params (PH2)
        self.real_diamter_pixels_mm = 72
        self.hsv_colors = {
            'Blue Gray': [np.array([15, 0, 0]),
                          np.array([179, 255, self.value_threshold]),
                          (0, 153, 0), 'BG'],  # Green
            'White': [np.array([0, 0, 145]),
                      np.array([15, 80, self.value_threshold]),
                      (255, 255, 0), 'W'],  # Cyan
            'Light Brown': [np.array([0, 80, self.value_threshold + 3]),
                            np.array([15, 255, 255]), (0, 255, 255), 'LB'],
            # Yellow
            'Dark Brown': [np.array([0, 80, 0]),
                           np.array([15, 255, self.value_threshold - 3]),
                           (0, 0, 204), 'DB'],  # Red
            'Black': [np.array([0, 0, 0]), np.array([15, 140, 90]),
                      (0, 0, 0), 'B'],  # Black
        }
        self.iter_colors = [
            [50, (0, 0, 255)],
            [100, (0, 153, 0)],
            [200, (255, 255, 0)],
            [400, (255, 0, 0)]
        ]
        self.borders = 2
        self.isImageValid = False
        self.contour = None
        self.max_area_pos = None
        self.contour_area = None
        self.feature_set = []
        self.performance_metric = []
        self.xmlfile = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "opencv_svm.xml")
        self.cols = ['A1', 'A2', 'B', 'C', 'A_B', 'A_BG', 'A_DB', 'A_LB',
                     'A_W', 'D1', 'D2']

        # Active contour params
        self.iter_list = [75, 25]
        self.gaussian_list = [7, 1.0]
        self.energy_list = [2, 1, 1, 1, 1]
        self.init_width = 0.65
        self.init_height = 0.65
        self.shape = 0  # 0 - ellipse, 1 - rectangle    

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

    def segment(self, iterations=9999, color=(255, 0, 0)):
        """
        This method performs segmentation using active contour model.
        :param iterations: Number of iterations for contour evolution
        :param color: Color of contour to be drawn on the image (scalar tuple)
        :return: None
        """
        # Active contour segmentation time
        if self.isImageValid:
            self.contour_mask = active_contour.run(self.image,
                                                   iterations,
                                                   self.shape,
                                                   self.init_width,
                                                   self.init_height,
                                                   self.iter_list,
                                                   self.energy_list,
                                                   self.gaussian_list)
            ret_val = extract_largest_contour(self.contour_mask)
            if len(ret_val) == 0:
                print("error")
                return
            else:
                mask_contours = ret_val[0]
                self.max_area_pos = ret_val[1]
                self.contour = mask_contours[self.max_area_pos]
                cnt = len(mask_contours)
                if cnt > 0:
                    cv2.drawContours(self.contour_image, mask_contours,
                                     self.max_area_pos,
                                     color,
                                     2)
                    self.contour_binary = np.zeros(self.image.shape[:2],
                                                   dtype=np.uint8)
                    cv2.drawContours(self.contour_binary, mask_contours,
                                     self.max_area_pos,
                                     255,
                                     2)
                    self.contour_area = cv2.contourArea(self.contour)
                    self.segmented_img = cv2.bitwise_and(
                        self.original_image, self.original_image,
                        mask=self.contour_mask)
                    self.segmented_img[self.segmented_img == 0] = 255
                else:
                    print("No contours found")
        return

    def loop_through_iterations(self):
        """
        This method shows the evolution process of the contour by performing
        different number of iterations and draw their respective contours on
        the original image.
        """
        for lst in self.iter_colors:
            start = time()
            print(lst)
            self.segment(lst[0], lst[1])
            end = time()
            self.performance_metric.append(end - start)

    def extract_features(self):
        """
        This method is used to extract features (A,B,D).
        """
        returnVars = features.extract(self.original_image,
                                      self.contour_mask,
                                      self.contour)
        if len(returnVars) == 0:
            self.feature_set = returnVars
        else:
            self.feature_set = returnVars[0]
            self.asymmetry_horizontal = returnVars[1]
            self.asymmetry_vertical = returnVars[2]
            self.warp_img_segmented = returnVars[3]

    def get_color_contours(self):
        """
        This method is used to extract color contours of different regions of
        the lesion.
        """
        tolerance = 30
        self.value_threshold = np.uint8(cv2.mean(self.hsv_image)[2]) \
                               - tolerance
        hsv = cv2.cvtColor(self.segmented_img, cv2.COLOR_BGR2HSV)
        no_of_colors = []
        # dist = []
        self.color_contour = np.copy(self.original_image)
        for color in self.hsv_colors:
            #            print color
            cnt = color_contour.extract(self.segmented_img, hsv,
                                        self.hsv_colors[color],
                                        self.contour_area)
            centroid = []
            color_attr = {}
            if len(cnt) > 0:
                for contour in cnt:
                    moments = cv2.moments(contour)
                    if moments['m00'] == 0:
                        continue
                    color_ctrd = [int(moments['m10'] / moments['m00']),
                                  int(moments['m01'] / moments['m00'])]

                    centroid.append(color_ctrd)
            if len(centroid) != 0:
                cv2.drawContours(self.color_contour, cnt, -1,
                                 self.hsv_colors[color][2],
                                 2)
                asym_color = np.mean(np.array(centroid), axis=0)
                dist = ((asym_color[0] -
                         self.feature_set['centroid'][
                             0]) ** 2 + (asym_color[1] -
                                         self.feature_set['centroid'][
                                             1]) ** 2) ** 0.5
                color_attr['color'] = color
                color_attr['centroids'] = centroid
                self.feature_set['A_' + self.hsv_colors[color][3]] = \
                    round(dist / self.feature_set['D1'], 4)
                no_of_colors.append(color_attr)
            else:
                self.feature_set['A_' + self.hsv_colors[color][3]] = 0
        self.feature_set['image'] = self.file_path
        self.feature_set['colors_attr'] = no_of_colors
        self.feature_set['C'] = len(no_of_colors)

    def classify_lesion(self):
        """
        This method performs classification of the lesion based on the features
        extracted and running them through a trained SVM classifier
        """
        svm = cv2.ml.SVM_load(self.xmlfile)
        feature_vector = np.array([self.feature_set[col] for col in self.cols],
                                  dtype=np.float32)
        print("feature vector ", feature_vector)
        res = svm.predict(feature_vector.reshape(-1, len(feature_vector)))
        if res[1] > 0:
            print("RESULT: Suspicious of Melanoma")
        else:
            print("RESULT: Benign")

    def save_images(self):
        """
        This method saves intermediate images for debugging purposes and can be
        toggled using the save flag.
        """
        cv2.imwrite(self.base_file + '.PNG', self.contour_image)
        cv2.imwrite(self.base_file + '_active_contour.PNG',
                    self.contour_binary)
        cv2.imwrite(self.base_file + '_colors.PNG',
                    self.color_contour)
        cv2.imwrite(self.base_file + '_mask.PNG',
                    self.contour_mask)
        cv2.imwrite(self.base_file + '_segmented.PNG',
                    self.segmented_img)
        cv2.imwrite(self.base_file + '_horizontal.PNG',
                    self.asymmetry_horizontal)
        cv2.imwrite(self.base_file + '_vertical.PNG',
                    self.asymmetry_vertical)
        cv2.imwrite(self.base_file + '_warped.PNG',
                    self.warp_img_segmented)

    def save_result(self):
        """
        Stores the features and the result in json format.
        """
        target = open(self.base_file + ".json", 'w')
        target.write(json.dumps(self.feature_set,
                                sort_keys=True, indent=2) + '\n')
        target.close()

    def extract_info(self, save=False):
        """
        This is the main method that performs all the required steps to
        classify a lesion by calling other methods in the class and also
        extract performance metrics.
        :param save: True saves the intermediate images and False does not
         (default: False)
        """
        # Preprocessing time
        start = time()
        if self.preprocess():
            end = time()
            self.performance_metric.append(end - start)
            self.loop_through_iterations()
            # A,B,D feature extraction time
            start = time()
            self.extract_features()
            end = time()
            self.performance_metric.append(end - start)
            # color feature extraction time
            start = time()
            self.get_color_contours()
            end = time()
            self.performance_metric.append(end - start)
            self.feature_set['D1'] = \
                int(round(float(self.feature_set['D1'])
                          / self.real_diamter_pixels_mm))
            self.feature_set['D2'] = \
                int(round(float(self.feature_set['D2'])
                          / self.real_diamter_pixels_mm))
            # Total feature extraction time
            self.performance_metric.append(self.performance_metric[-1] +
                                           self.performance_metric[-2])
            self.classify_lesion()
            if save and len(self.feature_set) != 0:
                self.save_images()
                self.save_result()
        else:
            print("Invalid image")