import numpy
from math import exp
from matplotlib import pyplot
import random
from pprint import pprint


class RBFN:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 3
    dbl_eta = 0.0001
    dbl_bias = 0.0002
    dbl_w0 = 0.0002
    dbl_tol = 0.0001

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

    # clusters
    arr_clusters = []

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

        ## initialize k clusters
        self.arr_clusters = [[] for x in range(0, self.int_num_hidden_neurons)]

        ## reset arrays
        self.arr_centroids = []
        self.arr_sigma = []
        self.arr_pos_vector = []

        return

    ## Run K-Means Once
    def preTrain(self, training_set):
        self.initCentroids(training_set)
        self.performKMeans(training_set)
        self.kMeansVariance(training_set)
        return

    ## Runk Iterative LMS
    def train(self, training_set):
        self.initCentroids(training_set)
        self.wrongKMeans(training_set)
        self.wrongKMeansVariance(training_set)
        self.LMS(training_set)
        return

    ## Test
    def test(self):
        return

    ## plot mse
    def plotMSE(self):
        pyplot.xlabel('Number of Epochs')
        pyplot.ylabel('MSE (Mean Square Error)')
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

    ## Refine Centroids Using K-Means
    def performKMeans(self, training_set):
        ## while no more centroid reallocation
        count = 0
        while True:
            ## loop training set
            for t in range(0, len(training_set)):
                # inputs
                x = training_set[t][0]

                # response
                d = training_set[t][1]

                ## loop centroids and calculate distance
                min_distance = numpy.inf
                min_index = -1
                for c in range(0, len(self.arr_centroids)):
                    try:
                        distance = self.calcDistance(x, self.arr_centroids[c])
                        if distance < min_distance:
                            min_distance = distance
                            min_index = c
                    except:
                        print 'Cluster with zeros Samples'


                self.arr_clusters[min_index].append(t)

                # end loop centroids
            # end loop training set

            # recalculate centroids
            arr_new_centroids = [[] for dummy in range(0, self.int_num_hidden_neurons)]
            for c in range(0, self.int_num_hidden_neurons):
                if len(self.arr_clusters[c]) == 0:
                    continue
                sum = [0 for dummy in range(0, self.int_num_input_neurons + 1)]
                for t in range(0, len(self.arr_clusters[c])):
                    the_index = self.arr_clusters[c][t]
                    sum = sum + training_set[the_index][0]
                arr_new_centroids[c] = [dummy / len(self.arr_clusters[c]) for dummy in sum]

            # compare centroids with old centroids
            if self.has_converged(self.arr_centroids, arr_new_centroids):
                return
            else:
                self.arr_centroids = arr_new_centroids
            count = count + 1
            print 'K-Means Iteration ', count
        return

    ## K-Means Variance
    def kMeansVariance(self, training_set):
        sum = numpy.zeros([self.int_num_hidden_neurons])
        count = numpy.zeros([self.int_num_hidden_neurons])
        ## loop hidden neurons
        for h in range(0, self.int_num_hidden_neurons):
            ## loop clusters array
            for t in range(0, len(self.arr_clusters[h])):
                the_index = self.arr_clusters[h][t]
                sum[h] = sum[h] + self.calcDistance(training_set[the_index][0], self.arr_centroids[h])
                count[h] = count[h] + 1

        ## loop hidden neurons and divide by count
        self.arr_sigma = []
        for h in range(0, self.int_num_hidden_neurons):
            self.arr_sigma.append(sum[h] / count[h])

        return

    ## compare two lists of lists in O(n^2)
    def has_converged(self, mu, oldmu):
        for i in range(0, len(mu)):
            for j in range(0, len(mu[i])):
                if abs(mu[i][j] - oldmu[i][j]) > self.dbl_tol:
                    return False
        return True

    ## K-Means
    def wrongKMeans(self, training_set):
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
            self.arr_centroids[index_min_distance] = self.calcHalfDistanceNewCentroid(self.arr_centroids[index_min_distance], x)

            ## assign sample to prototype
            self.arr_pos_vector[t] = index_min_distance

        return

    ## Claculate (x - c) . 2
    def calcHalfDistanceNewCentroid(self, vec_a, vec_b):
        return numpy.subtract(vec_a, vec_b) / 2


    ## Claculate Euclidean
    def calcDistance(self, vec_a, vec_b):
        sum = 0
        for i in range(0, len(vec_a)):
            sum = sum + numpy.square(vec_a[i] - vec_b[i])
        return numpy.sqrt(sum)

    ## K-Means Variance
    def wrongKMeansVariance(self, training_set):
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
                    try:
                        g[h] = self.calcGaussian(x, self.arr_centroids[h], self.arr_sigma[h])
                    except:
                        print 'Cluster With Zero Samples'

                # output? revise dimensions?
                o = numpy.dot(self.wo, g)
                ##o = o.tolist()
                ##classifier = o.index(max(o))
                # error
                error = d - o  ##o

                half_error_squared = self.halfErrorSquared(error)
                errors.append(half_error_squared)

                # weight correction rule
                scalar_value = self.dbl_eta * half_error_squared
                delta_w = numpy.multiply(scalar_value, g)
                ##self.wo = self.wo + delta_w.transpose()
                wo_cnt = 0
                for each_weight in self.wo:
                    self.wo[wo_cnt] = each_weight + delta_w
            ## end loop samples
            mse = self.calcMSE(errors)
            print mse
            self.arr_mse = numpy.append(self.arr_mse, mse)
            if mse < self.dbl_mse_threshold:
                break
        ## end loop epochs
        return

    ## calculate 1/2 e ^ 2
    def halfErrorSquared(self, vector):
        sum = 0
        for e in vector:
            sum += numpy.square(e)
        ret = sum / 2
        return ret;

    ## Gaussian e^[-1 * (x - c[h]) ^ 2 ]
    def calcGaussian(self, vec_a, vec_b, sigma):
        denominator = -2 * sigma
        return exp(self.calcDistance(vec_a, vec_b) / denominator)

    ## Calculate Mean Square Error
    def calcMSE(self, list_input):
        mse = 0.0
        for ele in list_input:
            mse = mse + ele
        cnt = len(list_input)
        return mse / cnt
