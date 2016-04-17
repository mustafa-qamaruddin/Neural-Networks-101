import cv2

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
            cv2.imshow('image', img)
            cv2.waitKey(0)

            # Initiate SIFT detector
            orb = cv2.ORB_create()

            # find the keypoints and descriptors with SIFT
            kp, des = orb.detectAndCompute(img, None)

            self.arr_key_points.append(kp)
            self.arr_descriptors.append(des)

            out_img = None
            out_img = cv2.drawKeypoints(img, kp, out_img)
            cv2.imshow('image', out_img)
            cv2.waitKey(0)
        return