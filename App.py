from sklearn.datasets import load_iris
from Perceptron import Perceptron

iris = load_iris()

## Constants
b = 0.001 ## Bias
ETA = 0.03
EPOCHS = 100
NUM_DATASET = 150
NUM_PER_CLASS = 50
NUM_TRAINING = 20
NUM_TESTING = 30

l0 = iris.data[0:NUM_TRAINING]
l1 = iris.data[NUM_PER_CLASS:NUM_PER_CLASS+NUM_TRAINING]

objPerc = Perceptron(ETA, EPOCHS)
objPerc.train(l0, 0)
objPerc.train(l1, 1)

t0 = iris.data[NUM_TRAINING:NUM_PER_CLASS]
t1 = iris.data[NUM_PER_CLASS+NUM_TRAINING:NUM_PER_CLASS+NUM_PER_CLASS]

objPerc.test(t0, 0)
objPerc.test(t1, 1)