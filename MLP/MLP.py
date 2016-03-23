import numpy
<<<<<<< HEAD
=======
from math import exp
>>>>>>> e8475a9a821012de41fafe00351b028d1759a41c

class MLP:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 0.001
    dbl_eta = 0.0001
<<<<<<< HEAD
=======
    dbl_bias = 0.0002
    dbl_w0 = 0.0002
>>>>>>> e8475a9a821012de41fafe00351b028d1759a41c

    arr_num_neurons_in_hidden_layers = []

    # weights
    wo = []
    wh = []

    ## initializations
    def __init__(self, _int_num_input_neurons, _int_num_output_neurons, _int_num_hidden_layers, _int_num_epochs, _int_num_hidden_neurons, _dbl_eta):
        ## initialize properties
        self.int_num_input_neurons = _int_num_input_neurons
        self.int_num_output_neurons = _int_num_output_neurons
        self.int_num_hidden_layers = _int_num_hidden_layers
        self.int_num_epochs = _int_num_epochs
        self.int_num_hidden_neurons = _int_num_hidden_neurons
        self.dbl_eta = _dbl_eta

        ## initialize weights arrays
<<<<<<< HEAD
        self.wo = numpy.zeros(self.int_num_output_neurons, self.int_num_hidden_neurons + 1) ## bias +1
        self.wh = numpy.zeros(self.int_num_hidden_neurons, self.int_num_input_neurons + 1) ## bias +1

=======
        self.wo = [[self.dbl_w0 for x in range(self.int_num_hidden_neurons + 1)] for y in range(self.int_num_output_neurons)] ## bias +1
        self.wh = [[self.dbl_w0 for x in range(self.int_num_input_neurons + 1)] for y in range(self.int_num_hidden_neurons)] ## bias +1
>>>>>>> e8475a9a821012de41fafe00351b028d1759a41c
        return

    ## back-propagation-algorithm
    def train(self, training_set):
        ## loop epochs
        for e in range(0, self.int_num_epochs):
            ## loop training set
<<<<<<< HEAD
            for t in range(0, len(training_set)):
                ## forward
                return
                ## backward
                ## update weights
            ## end loop training set
        ## end loop epochs
        return
=======
            errors = []
            for t in range(0, len(training_set)):
                # inputs
                x = training_set[t][0]

                # response
                d = training_set[t][1]

                # forward path
                hidden_output = self.hyberb(numpy.inner(self.wh, x))
                temp = numpy.reshape(hidden_output, (self.int_num_hidden_neurons))
                temp_temp = numpy.append(temp, self.dbl_bias)
                output_output = self.hyberb(numpy.inner(self.wo, temp_temp))
                error = d - output_output
                errors = numpy.append(errors,  error)

                ## backward path
                error_signal_output = error * self.derivhyberb(numpy.inner(self.wo, temp_temp))
                temp = numpy.reshape(error_signal_output, (self.int_num_hidden_neurons))
                temp_temp = numpy.append(temp, self.dbl_bias)
                partial = numpy.inner(self.wo, temp_temp)
                error_signal_hidden = self.derivhyberb(numpy.inner(self.wh, x)) * partial

                ## update weights
                delta_wh = self.dbl_eta * numpy.inner(error_signal_hidden , x.transpose())
                delta_wo = self.dbl_eta * numpy.inner(error_signal_output , hidden_output)

                print delta_wh
                print delta_wo

            ## end loop training set
        ## end loop epochs
        return

    ## hyber-bolic function
    def hyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        #PHI.arange().reshape
        return (numpy.exp(V * 2) - 1) / (numpy.exp(V * 2) + 1)

    ## derivation of hyper-bolic function
    def derivhyberb(self, V):
        """
        :rtype: VECTOR SAME DIMENSIONS AS V
        """
        #PHI.arange().reshape
        return 4 * numpy.exp(V * 2) / numpy.square(1 + numpy.exp(V * 2))
>>>>>>> e8475a9a821012de41fafe00351b028d1759a41c
