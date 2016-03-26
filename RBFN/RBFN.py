import numpy
from math import exp
from matplotlib import pyplot
import random


class RBFN:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 0.001
    dbl_eta = 0.0001
    dbl_bias = 0.0002
    dbl_w0 = 0.0002

    arr_num_neurons_in_hidden_layers = []

    # weights
    wo = []
    wh = []

    # mse
    arr_mse = []

    # centroids
    arr_centroids = []

    # pos_vector = zeros(1,num_tr); % a vector to denote which point belongs to which center
    arr_pos_vector = []

    # variance of each hidden neuron
    arr_sigma = []

    ## initializations
    def __init__(self, _int_num_input_neurons, _int_num_output_neurons, _int_num_hidden_layers, _int_num_epochs,
                 _int_num_hidden_neurons, _dbl_eta):
        ## initialize properties
        self.int_num_input_neurons = _int_num_input_neurons
        self.int_num_output_neurons = _int_num_output_neurons
        self.int_num_hidden_layers = _int_num_hidden_layers
        self.int_num_epochs = _int_num_epochs
        self.int_num_hidden_neurons = _int_num_hidden_neurons
        self.dbl_eta = _dbl_eta

        ## initialize weights arrays
        self.wo = [[self.dbl_w0 for x in range(self.int_num_hidden_neurons + 1)] for y in
                   range(self.int_num_output_neurons)]  ## bias +1
        self.wh = [[self.dbl_w0 for x in range(self.int_num_input_neurons + 1)] for y in
                   range(self.int_num_hidden_neurons)]  ## bias +1

        return

    ## Train
    def train(self, training_set):
        self.initCentroids(training_set)
        self.kMeans(training_set)
        self.kMeansVariance(training_set)
        self.plotSamples(training_set)
        self.plotKCentroids()
        self.LMS(training_set)
        return

    ## Test
    def test(self):
        return

    ## plot mse
    def plotMSE(self):
        pyplot.xlabel('Number of Epochs')
        pyplot.ylabel('MSE (Mean Square Error)')
        print  self.arr_mse
        pyplot.plot(self.arr_mse.tolist())
        pyplot.show()

    # plot samples
    def plotSamples(self, training_set):
        return

    # plot k centroids
    def plotKCentroids(self):
        return

    ## Random Centroids
    def initCentroids(self, training_set):
        for h in range(0, self.int_num_hidden_neurons):
            sample = random.choice(training_set)
            self.arr_centroids.append(sample[0])
        return

    ## K-Means
    def kMeans(self, training_set):
        ## loop training set
        self.arr_pos_vector = [0 for x in range(0, len(training_set))]
        for t in range(0, len(training_set)):
            # inputs
            x = training_set[t][0]

            # response
            d = training_set[t][1]

            ## loop hidden neurons & find nearest prototype
            eu_dist = []
            for h in range(0, self.int_num_hidden_neurons):
                eu_dist.append(self.calcDistance(x, self.arr_centroids[h]))

            value_min_distance = min(eu_dist)
            index_min_distance = eu_dist.index(value_min_distance)

            ## update nearest prototype
            self.arr_centroids[index_min_distance] = self.arr_centroids[index_min_distance] + value_min_distance
            ## assign sample to prototype
            self.arr_pos_vector[t] = index_min_distance

        return

    ## Claculate Euclidean
    def calcDistance(self, vec_a, vec_b):
        sum = 0
        for i in range(0, len(vec_a)):
            sum = sum + numpy.square(vec_a[i] - vec_b[i])
        return numpy.sqrt(sum)

    ## K-Means Variance
    def kMeansVariance(self, training_set):
        sum = numpy.zeros([self.int_num_hidden_neurons])
        count = numpy.zeros([self.int_num_hidden_neurons])
        ## loop hidden neurons
        for h in range(0, self.int_num_hidden_neurons):
            ## loop positions vector
            for p in range(0, len(self.arr_pos_vector)):
                if h == self.arr_pos_vector[p]:
                    sum[h] = sum[h] + self.calcDistance(training_set[p][0], self.arr_centroids[h])
                    count[h] = count[h] + 1

        ## loop hidden neurons and divide by count
        self.arr_sigma = []
        for h in range(0, self.int_num_hidden_neurons):
            self.arr_sigma.append(sum[h] / count[h])

        return

    ## Traiing with LMS
    def LMS(self, training_set):
        ## loop epochs
        mse = []
        for e in range(0, self.int_num_epochs):
            ## loop training set
            errors = []
            for t in range(0, len(training_set)):
                # inputs
                x = training_set[t][0]

                # response
                d = training_set[t][1]

                # Gaussian
                g = [1 for y in range(0, self.int_num_hidden_neurons + 1)]  ## bias
                for h in range(0, self.int_num_hidden_neurons):
                    g[h] = self.calcGaussian(x, self.arr_centroids[h], self.arr_sigma[h])

                # output? revise dimensions?
                o = numpy.inner(self.wo, g)

                # error
                error = d - o
                errors.append(error)

                # weight correction rule
                delta_w = numpy.zeros([self.int_num_hidden_neurons + 1, self.int_num_output_neurons])
                for h in g:
                    numpy.append(delta_w, self.dbl_eta * error * h)

                self.wo = self.wo + delta_w.transpose()
            ## end loop samples
            mse = numpy.mean(numpy.sum(numpy.square(errors)))
            self.arr_mse = numpy.append(self.arr_mse, mse)
            if mse < self.dbl_mse_threshold:
                break
            print 'MSE'
            print e
            print mse
        ## end loop epochs
        return

    ## Gaussian e^[-1 * (x - c[h]) ^ 2 ]
    def calcGaussian(self, vec_a, vec_b, sigma):
        denominator = -2 * numpy.square(sigma)
        return exp(self.calcDistance(vec_a, vec_b) / denominator)
