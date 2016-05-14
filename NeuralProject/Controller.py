from View import View
from StateOfNature import StateOfNature
from PIL import Image
import glob
import cv2
from mqSVM import mqSVM
import numpy
import sys


class Controller:
    arr_objs_states_of_nature = []
    classifier = None

    def __init__(self):
        return

    def run(self):
        app = View(self)
        app.mainloop()
        return

    def importImagesFrmFolder(self, dir_name, class_name, class_index):
        obj_state_of_nature = StateOfNature(class_name, class_index)
        obj_state_of_nature.arr_training_set = []
        counter = 10;
        for filename in glob.glob(dir_name + '/*.jpg'):  # assuming gif
            if counter == 0:
                break
            im = cv2.imread(filename, 0)
            obj_state_of_nature.arr_training_set.append(im)
            counter -= 1
        self.arr_objs_states_of_nature.append(obj_state_of_nature)
        return

    def findKeyPoints(self):
        for obj in self.arr_objs_states_of_nature:
            obj.applySIFT()
        return

    def applySVM(self):
        self.classifier = mqSVM()
        for obj in self.arr_objs_states_of_nature:
            samples = obj.getSamples()
            responses = obj.getResponses()
            for i in range(len(samples)):
                self.classifier.train(samples[i], responses[i])
        return

    def predict(self, file_name):
        test_image = cv2.imread(file_name, 0)
        obj = StateOfNature('', -1)
        samples, responses = obj.mqSIFT(test_image)
        classification = self.classifier.predict(samples)
        index = self.countVotes(classification)
        obj = self.arr_objs_states_of_nature[int(index) - 1]

        # Find Rectangle Coordinates
        kp = obj.getSIFTKP(test_image)
        x_min, x_max, y_min, y_max = self.getMinMaxCoords(kp)

        # Show Image With Label
        self.showImageWithLabel(file_name, obj.getLabel(), x_min, x_max, y_min, y_max)

        return obj.getLabel()

    def countVotes(self, in_votes):
        votes = numpy.array(in_votes)
        counts = {}
        for i in range(len(votes)):
            counts[votes[i]] = counts.get(votes[i], 0) + 1
        sortedCounts = sorted(counts, reverse=True)
        return sortedCounts[0]

    def showImageWithLabel(self, file_name, label, x_min, x_max, y_min, y_max):
        # read file
        img = cv2.imread(file_name, 0)
        # draw rectangle
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
        # write text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, label, (x_min, y_min+20), font, 1, (255, 0, 0), 1, cv2.LINE_AA)
        # show image
        cv2.imshow('image', img)
        cv2.waitKey(0)
        return

    def getMinMaxCoords(self, key_points):
        min_x = sys.maxsize
        min_y = sys.maxsize
        max_x = -sys.maxsize - 1
        max_y = -sys.maxsize- 1
        for kp in key_points:
            x = int(kp.pt[0])
            y = int(kp.pt[1])
            if x <= min_x:
                min_x = x
            if x >= max_x:
                max_x = x

            if y <= min_y:
                min_y = y
            if y >= max_y:
                max_y = y
        return min_x, max_x, min_y, max_y

    def testSVM(self):
        counter = 0
        correct = 0
        for obj in self.arr_objs_states_of_nature:
            samples = obj.getSamples()
            for i in range(len(samples)):
                classification = self.classifier.predict(samples[i])
                index = self.countVotes(classification)
                tempo = self.arr_objs_states_of_nature[int(index) - 1]
                if tempo.getLabel() == obj.getLabel():
                    correct = correct + 1
                counter = counter + 1
        overall_accuracy = correct / counter * 100
        return overall_accuracy
