import numpy as np
from statlearn.network import NeuralNetwork

class AutoEncoder(NeuralNetwork):
    def __init__(self, shape, encoder_type, decoder_type, parameters=None):
        super(Autoencoder, self).__init__(shape)
        encoder = Layer( (shape[0], shape[1]), encoder_type, parameters)
        decoder = Layer( (shape[1], shape[2]), decoder_type, parameters)
