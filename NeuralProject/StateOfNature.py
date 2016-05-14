from sklearn import svm
from cv2 import *
import numpy as numpy
import cv2

class StateOfNature:
    arr_training_set = []
    arr_samples = []
    arr_responses = []
    str_label = ''
    int_label = -1

    def __init__(self, _label, _index):
        self.str_label = _label
        self.int_label = _index
        return

    def applySIFT(self):
        for img in self.arr_training_set:
            samples, responses = self.mqSIFT(img)
            ## ??!! ##
            responses = numpy.tile(self.int_label, len(samples))
            self.arr_samples.append(samples)
            self.arr_responses.append(responses)
        return

    def getSamples(self):
        return self.arr_samples

    def getResponses(self):
        return self.arr_responses

    def mqSIFT(self, image):
        # Display Original Image
        #imshow('image', image)
        #waitKey(0)

        # Initiate SIFT detector
        detector = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp, des = detector.detectAndCompute(image, None)

        # Convert Objects 2 Arrays ?
        samples = numpy.array(des)
        responses = numpy.arange(len(kp), dtype=numpy.integer)

        out_img = None
        out_img = drawKeypoints(image, kp, out_img)

        # Display Image with SIFT Key Points
        #imshow('image', out_img)
        #waitKey(0)

        return samples, responses

    def getLabel(self):
        return self.str_label

    def getSIFTKP(self, input_image):
        # Initiate SIFT detector
        detector = cv2.xfeatures2d.SIFT_create()

        # find the keypoints and descriptors with SIFT
        return detector.detect(input_image, None)