from sklearn.datasets import load_iris
from Perceptron import Perceptron
from sklearn import preprocessing

iris = load_iris()

## Constants
b = 0.0001 ## Bias
ETA = 0.0001
EPOCHS = 1000
NUM_DATASET = 150
NUM_PER_CLASS = 50
NUM_TRAINING = 20
NUM_TESTING = 30

NUM_TOTAL_SAMPLES = iris.data.shape[0]
NUM_FEATURES = iris.data.shape[1]

Training_Data = []
Testing_Data = []
for i in range(0, NUM_TOTAL_SAMPLES):
    Y = iris.target[i]
    X = iris.data[i]
    check_i = i % NUM_PER_CLASS
    if check_i < NUM_TRAINING:
        Training_Data.append([X, Y])
    else:
        Testing_Data.append([X, Y])

objPerc = Perceptron(ETA, EPOCHS)

objPerc.train(Training_Data)

objPerc.test(Testing_Data)