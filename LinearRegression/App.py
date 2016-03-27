from sklearn.datasets import load_iris
from LReg import LinearRegression
from sklearn import preprocessing
import numpy
iris = load_iris()
b = 0.001 ## Bias
Err = 0
EPOCHS = 1
lamda = 0.1
NUM_DATASET = 150
NUM_PER_CLASS = 50
NUM_TRAINING = 30
NUM_TESTING = 20
Num_of_Samples = NUM_TRAINING + NUM_TESTING
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
        #print Training_Data
    else:
        Testing_Data.append([X, Y])
numpy.random.shuffle(Training_Data)
numpy.random.shuffle(Testing_Data)
objPerc = LinearRegression(Err, EPOCHS,lamda)
objPerc.train(Training_Data)
objPerc.test(Testing_Data)