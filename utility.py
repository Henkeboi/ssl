class ConvConfig:
    def __init__(self, input_shape, filters, kernel_size, af):
        self.input_shape = input_shape
        self.filters = filters
        self.kernel_size = kernel_size
        self.af = af


class DenseConfig:
    def __init__(self, neurons, af):
        self.neurons = neurons
        self.af = af
