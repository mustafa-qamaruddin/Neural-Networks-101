import numpy
from math import exp
# from matplotlib import pyplot
import pprint


class mqMLP:
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
    overallaccuracy = 0.0
    arr_num_neurons_in_hidden_layers = []
    CofusionMat = numpy.zeros((3, 3))
    # weights
    wo = []
    wh = []

    # mse
    arr_mse = []

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
        self.CofusionMat = numpy.zeros((_int_num_output_neurons, _int_num_output_neurons))
        self.overallaccuracy = 0.0
        ## initialize weights arrays
        self.wo = [[self.dbl_w0 for x in range(self.int_num_hidden_neurons + 1)] for y in
                   range(self.int_num_output_neurons)]  ## bias +1
        self.wh = [[self.dbl_w0 for x in range(self.int_num_input_neurons + 1)] for y in
                   range(self.int_num_hidden_neurons)]  ## bias +1
        return

    ## back-propagation-algorithm
    def train(self, samples, responses):
        ## loop epochs
        mse = []
        for e in range(0, self.int_num_epochs):
            ## loop training set
            errors = []
            for t in range(0, len(samples)):
                # inputs
                x = numpy.array(samples[t])
                x = numpy.append(x, 1)

                # response
                d = responses[t]

                # forward path
                actual_hidden_output = self.hyberb(numpy.inner(self.wh, x))

                actual_hidden_output_plus_rshp = numpy.reshape(actual_hidden_output, (self.int_num_hidden_neurons))
                actual_hidden_output_plus_bias = numpy.append(actual_hidden_output_plus_rshp, self.dbl_bias)

                actual_output = self.hyberb(numpy.inner(self.wo, actual_hidden_output_plus_bias))
                actual_output = actual_output.tolist()
                actual_output = actual_output.index(max(actual_output))
                ## Question?! Why substract actual_output[3x1] Vector from scalar d = {0, 1, 2}
                error = d - actual_output

                ## erros will be used for mse
                errors = numpy.append(errors, error)

                ## backward path
                error_signal_output = []
                for i in range(len(self.wo)):
                    vec_initial_weights = self.wo[i]
                    temp_vec = numpy.inner(vec_initial_weights, actual_hidden_output_plus_bias)
                    error_signal_output.append(error * self.derivhyberb(temp_vec))

                ####### note there is no input weights to bias node            #######
                ####### add dumpy column for bias weights to avoid numpy error #######
                error_signal_output_dump_bias = numpy.append(error_signal_output, self.dbl_bias)

                derivative_temp_vec_1 = []
                for i in range(len(self.wh)):
                    temp_vec_1 = numpy.inner(self.wh[i], x)
                    derivative_temp_vec_1.append(self.derivhyberb(temp_vec_1))

                error_signal_hidden = []
                for i in range(len(self.wo)):
                    temp_initial_weights = self.wo[i]
                    temp_error_signal_output = error_signal_output_dump_bias[i]
                    temp_vec_2 = numpy.multiply(temp_initial_weights, temp_error_signal_output)
                    error_signal_hidden.append(derivative_temp_vec_1[i] * temp_vec_2)

                ###dimensions of error_signal_hidden = (number_of_hidden_neurons x 1)
                ## update weights hidden
                tmp_wh = numpy.transpose(self.wh)
                counter = 0
                for x_ele in x:
                    delta_wh = self.dbl_eta * error_signal_hidden[0] * x_ele
                    tmp_wh[counter] = delta_wh[0:len(tmp_wh[counter])] + tmp_wh[counter]  ## update weight
                    counter = counter + 1
                self.wh = tmp_wh.transpose()  ## weights updated
                ## end for x

                ## update weights output
                delta_wo = self.dbl_eta * error_signal_output[0] * actual_hidden_output
                for i in range(len(self.wo)):
                    counter = 0
                    for delta_wo_ele in delta_wo:
                        self.wo[i][counter] = self.wo[i][counter] + delta_wo_ele
                        counter = counter + 1
                        ## out weights updated
            ## end loop training set
            self.arr_mse = numpy.append(self.arr_mse, numpy.mean(numpy.sum(numpy.square(errors))))
        ## end loop epochs
        return

    ##############################################Testing Algorithm###############################################
    def test(self, testing_set):
        for t in range(0, len(testing_set)):
            x = numpy.array(testing_set[t])
            x = numpy.append(x, 1)

            # forward path
            actual_hidden_output = self.hyberb(numpy.inner(self.wh, x))

            actual_hidden_output_plus_rshp = numpy.reshape(actual_hidden_output, (self.int_num_hidden_neurons))
            actual_hidden_output_plus_bias = numpy.append(actual_hidden_output_plus_rshp, self.dbl_bias)

            actual_output = self.hyberb(numpy.inner(self.wo, actual_hidden_output_plus_bias))
            actual_output = actual_output.tolist()
            actual_output = actual_output.index(max(actual_output))
        return actual_output

    ###############################################################################################
    ## hyber-bolic function
    def hyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        # PHI.arange().reshape
        return (numpy.exp(V * 2) - 1) / (numpy.exp(V * 2) + 1)

    ## derivation of hyper-bolic function
    def derivhyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        # PHI.arange().reshape
        return 4 * numpy.exp(V * 2) / numpy.square(1 + numpy.exp(V * 2))

    # mysign
    def mysign(self, y):
        if y >= 0.0:
            return 0
        else:
            return 1

    ## plot mse
    def plotMSE(self):
        # pyplot.xlabel('Number of Epochs')
        # pyplot.ylabel('MSE (Mean Square Error)')
        # pyplot.plot(self.arr_mse)
        # pyplot.show()
        return
