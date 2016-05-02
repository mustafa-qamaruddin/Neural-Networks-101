from sklearn import svm
from cv2 import *


class StateOfNature:
    arr_training_set = []
    arr_key_points = []
    arr_descriptors = []
    str_label = ''

    def __init__(self, _label):
        self.str_label = _label
        return

    def applySIFT(self):
        for img in self.arr_training_set:
            imshow('image', img)
            waitKey(0)

            # Initiate SIFT detector
            orb = ORB_create()

            # find the keypoints and descriptors with SIFT
            kp, des = orb.detectAndCompute(img, None)

            self.arr_key_points.append(kp)
            self.arr_descriptors.append(des)

            out_img = None
            out_img = drawKeypoints(img, kp, out_img)
            imshow('image', out_img)
            waitKey(0)
        return

    def getArrKeyPoints(self):
        return self.arr_key_points

    def getLabel(self):
        return self.str_label

    def getLabelVector(self):
        return [self.str_label for x in range(len(self.arr_key_points))]
