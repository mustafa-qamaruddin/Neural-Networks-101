import numpy

class MLP:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0
    int_num_hidden_neurons = 0
    dbl_mse_threshold = 0.001

    arr_num_neurons_in_hidden_layers = []

    # weights
    wo = []
    wh = []

    ## initializations
    def __init__(self, _int_num_input_neurons, _int_num_output_neurons, _int_num_hidden_layers, _int_num_epochs, _int_num_hidden_neurons):
        ## initialize properties
        self.int_num_input_neurons = _int_num_input_neurons
        self.int_num_output_neurons = _int_num_output_neurons
        self.int_num_hidden_layers = _int_num_hidden_layers
        self.int_num_epochs = _int_num_epochs
        self.int_num_hidden_neurons = _int_num_hidden_neurons

        ## initialize weights arrays
        self.wo = numpy.zeros(self.int_num_output_neurons, self.int_num_hidden_neurons + 1) ## bias +1
        self.wh = numpy.zeros(self.int_num_hidden_neurons, self.int_num_input_neurons + 1) ## bias +1

        return

    ## back-propagation-algorithm
    def train(self):
        return
