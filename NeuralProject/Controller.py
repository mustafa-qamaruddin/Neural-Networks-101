from View import View
from StateOfNature import StateOfNature
from PIL import Image
import glob

class Controller:
    arr_objs_states_of_nature = []

    def __init__(self):
        return

    def run(self):
        app = View(self)
        app.mainloop()
        return

    def importImagesFrmFolder(self, dir_name, class_name):
        obj_state_of_nature = StateOfNature(class_name)
        obj_state_of_nature.arr_training_set = []
        counter = 10;
        for filename in glob.glob(dir_name+'/*.jpg'):  # assuming gif
            if counter == 0:
                break
            im = Image.open(filename)
            obj_state_of_nature.arr_training_set.append(im)
            counter -= 1
        self.arr_objs_states_of_nature.append(obj_state_of_nature)
        return

    def findKeyPoints(self):
        for obj in self.arr_objs_states_of_nature:
            obj.applySIFT()
        return