from os import path
import numpy as np
import tensorflow as tf
import cv2
import imutils
import numpy as np
from numpy import random
from imutils import contours
from os import path
import glob
from PIL import Image
from keras_unet.models import custom_unet
from keras_unet.metrics import iou, iou_thresholded
from matplotlib import pyplot as plt

class EdgeCalculation:
    def __init__(self, img_preds):
        self.img_preds = img_preds
    
    #predicts results on pretrained model and returns size distributions
    def loadmodel(self):
        #load model from current path which contains pretrained weights
        input_x, input_shape = self.prepare_input(self.img_preds)
        model = tf.keras.models.load_model('segm_model_v3.h5', custom_objects={'iou': iou, 'iou_thresholded':iou_thresholded})
        y_pred = model.predict(input_x)
        im_path = path.abspath(path.join(path.curdir,'notebooks','pred.jpg'))
        y_pred = y_pred[0]
        cv2.imwrite(im_path,y_pred)
        size_distributions = self.get_size_distributions(im_path)
        return size_distributions

    def prepare_input(self, input_img):
        imgs_list = []
        imgs_list.append(np.array(Image.open(input_img).resize((384,384))))
        imgs_np = np.asarray(imgs_list)
        x = np.asarray(imgs_np, dtype=np.float32)/255
        input_shape = x[0].shape
        return x, input_shape

    def auto_canny(self, image, sigma=0.33):
        # compute the median of the single channel pixel intensities
        v = np.median(image)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edged = cv2.Canny(image, 100, 150)
        # return the edged image
        return edged

    #gets edges, plots image returns size distribution dictionary
    def get_size_distributions(self,y_pred):
		#load image 
        labelled_image = cv2.imread(y_pred)
        gray = cv2.cvtColor(labelled_image, cv2.COLOR_BGR2GRAY)
        gray_blurred = cv2.GaussianBlur(gray, (7, 7), 0)

        #find edge detection, then perform a dilation and erosion to close gaps in between objects
        edge_detected = self.auto_canny(gray_blurred)
        edged_dilated = cv2.dilate(edge_detected, None, iterations=1)
        edged_eroded = cv2.erode(edged_dilated, None, iterations=1)

        #find contours
        cnts= cv2.findContours(gray.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
        cnts = imutils.grab_contours(cnts)

        image = labelled_image.copy()
        class_names = ['100','90','80','70','60','50','40','30','20','10']
        colors = [(143, 17, 3), (219, 83, 68), (140, 113, 6), (230, 198, 71),(28, 145, 1),
                    (28, 145, 1), (2, 122, 114), (78, 222, 212), (31, 5, 117),(134, 112, 204)]

        sorted_contours = sorted(cnts, key=cv2.contourArea,reverse=True)
        group_count = len(sorted_contours) // len(class_names)
        enumerate_from = 0

        distributions = {}
        for i in range(len(class_names)):
            enumerate_till = group_count * (i+1)
            colr = colors[i]
            cls_name = class_names[i]
            size_count = 0
            areas = []
            for idx, c in enumerate(sorted_contours):
                if idx >= enumerate_from and idx < enumerate_till:
                    size_count +=1
                    # if the contour is not sufficiently large, ignore it
                    countour_area = cv2.contourArea(c)
                    areas.append(countour_area)
                    # compute the center of the contour
                    M = cv2.moments(c)
                    if M["m01"] != 0 and M["m00"] != 0 and M["m10"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # draw the contour and center of the shape on the image
                        cv2.drawContours(image, [c], -1, colr, -1)
                        cv2.circle(image, (cX, cY), 7, (0, 0, 0), -1)
                        if i < 3:
                            cnt_string = cls_name + ':' + str(countour_area)
                        else:
                            cnt_string = ''
                        cv2.putText(image, cnt_string, (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                elif idx > enumerate_till:
                    break
                else:
                    continue
            distributions[cls_name] = areas
            enumerate_from = enumerate_till
        cv2.imwrite(path.join(path.curdir , 'predictions.jpg'), image)
        return distributions

im_path = path.abspath(path.join(path.curdir,'input\\whales\\0000e88ab.jpg'))
edge_detection = EdgeCalculation(im_path)

distribution_preds = edge_detection.loadmodel()
print(distribution_preds)
