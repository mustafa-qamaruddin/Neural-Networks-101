class MLP:
    # properties
    int_num_epochs = 1000
    int_num_input_neurons = 0
    int_num_output_neurons = 0
    int_num_hidden_layers = 0

    arr_num_neurons_in_hidden_layers = []

    def __init__(self, _int_num_input_neurons, _int_num_output_neurons, _int_num_hidden_layers, _int_num_epochs):
        self.int_num_input_neurons = _int_num_input_neurons
        self.int_num_output_neurons = _int_num_output_neurons
        self.int_num_hidden_layers = _int_num_hidden_layers
        self.int_num_epochs = _int_num_epochs

        ## number of neurons in hidden layers ?

