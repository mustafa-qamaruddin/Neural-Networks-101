import numpy

class MLP:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 0.001
    dbl_eta = 0.0001

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
        self.wo = numpy.zeros(self.int_num_output_neurons, self.int_num_hidden_neurons + 1) ## bias +1
        self.wh = numpy.zeros(self.int_num_hidden_neurons, self.int_num_input_neurons + 1) ## bias +1

        return

    ## back-propagation-algorithm
    def train(self, training_set):
        ## loop epochs
        for e in range(0, self.int_num_epochs):
            ## loop training set
            for t in range(0, len(training_set)):
                ## forward
                return
                ## backward
                ## update weights
            ## end loop training set
        ## end loop epochs
        return
