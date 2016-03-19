from sklearn.datasets import load_iris
from Perceptron import Perceptron

iris = load_iris()

## Constants
b = 0.001 ## Bias
ETA = 0.03
EPOCHS = 100
NUM_TRAINING = 20
NUM_TESTING = 30

objPerc = Perceptron(ETA, EPOCHS)
objPerc.train(iris.data, iris.target)