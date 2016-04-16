import cv2

class StateOfNature:
    arr_training_set = []
    arr_key_points = []
    str_label = ''

    def __init__(self, _label):
        self.str_label = _label
        return

    def applySIFT(self):

        for img in self.arr_training_set:
            print "hola," + self.str_label
            cv2.NamedWindow("opencv")
            cv2.ShowImage("opencv", img)
            cv2.WaitKey(0)

            sift = cv2.SIFT()
            sift_ocl = sift.SiftPlan(template=img, device=GPU)
            kp = sift_ocl.keypoints(img)
            kp.sort(order=["scale", "angle", "x", "y"])

            self.arr_key_points.insert(kp)
        return