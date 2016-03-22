from sklearn.datasets import load_iris
from sklearn import preprocessing

class Controller:
    # dataset
    data = []
    training = []
    testing = []

    # characteristics
    int_set_size = 0
    int_num_features = 0
    int_training_size = 30
    int_testing_size = 20
    int_num_per_class = 50

    # initializiation
    def __init__(self):
        iris = load_iris()
        self.prepareDataSet(iris)

    # normalize or scale data
    # divide data to training and testing
    def prepareDataSet(self, iris):
        self.int_set_size = iris.data.shape[1]
        self.int_num_features = iris.data.shape[0]
        ## normalize data
        self.data = preprocessing.normalize(iris.data)

        # load data in arrays
        for i in range(0, self.int_set_size):
            Y = iris.target[i]
            X = self.data[i]
            check_i = i % self.int_num_per_class
            if check_i < self.int_training_size:
                self.training.append([X, Y])
            else:
                self.testing.append([X, Y])

