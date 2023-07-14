from components.decorators.database_decorators import *
from components import Component
import tensorflow as tf
import numpy as np
import inspect


class NeuralNetwork(Component):

    def __init__(self, settings):
        super().__init__(settings)

        """ dictionary containing attributes for all neural networks """
        self.neural_networks = {}
        self.callbacks = []

    def get_neural_networks(self):
        """ return dictionary with all neural networks """
        return self.neural_networks

    def set_model_weights(self, weights, neural_network_index=0):
        """ return weights and bias from neural network with index = neural_network_index """
        self.neural_networks[f"{neural_network_index}"].set_weights(weights)

    def get_model_weights(self, neural_network_index=0):
        """ set weights and bias to the neural network with index = neural_network_index """
        return self.neural_networks[f"{neural_network_index}"].get_weights()

    @keras_neural_network_database
    def add_keras_neural_network(self, neural_network_function, name=None, seed=None):
        """ add a new neural network to the component """
        if not bool(self.neural_networks):
            self.settings["models"] = {}
        neural_network_index = len(self.neural_networks)
        if inspect.isfunction(neural_network_function):
            neural_network_name = neural_network_function.__name__ if name is None else name
            tf_random_seed = np.random.randint(1048576) if seed is None else seed
            tf.random.set_seed(tf_random_seed)
            method_name = neural_network_function.__name__
            neural_network_function = neural_network_function(self.settings["classes"], self.settings["number_of_samples"])
            return self.__set_neural_network_attributes(neural_network_index, neural_network_name, method_name, tf_random_seed,
                                                        neural_network_function)
        else:
            return self.__set_neural_network_attributes(neural_network_index, neural_network_function.name if name is None else name,
                                                        neural_network_function.name, seed, neural_network_function)

    @tensorflow_neural_network_database
    def add_tensorflow_neural_network(self, neural_network, name=None, seed=None):
        # TODO
        pass

    @torch_neural_network_database
    def add_torch_neural_network(self, neural_network, name=None, seed=None):
        # TODO
        pass

    def __set_neural_network_attributes(self, neural_network_index, neural_network_name, method_name, seed, neural_network):
        """ set neural network attributes to the neural network dictionary """

        self.neural_networks[f"{neural_network_index}"] = {}
        self.neural_networks[f"{neural_network_index}"]["neural_network_name"] = neural_network_name
        self.neural_networks[f"{neural_network_index}"]["method_name"] = method_name
        self.neural_networks[f"{neural_network_index}"]["seed"] = seed
        self.neural_networks[f"{neural_network_index}"]["neural_network"] = neural_network
        self.neural_networks[f"{neural_network_index}"]["index"] = neural_network_index

        self.settings["neural_networks"][f"{neural_network_index}"] = {}
        self.settings["neural_networks"][f"{neural_network_index}"]["neural_network_name"] = neural_network_name
        self.settings["neural_networks"][f"{neural_network_index}"]["method_name"] = method_name
        self.settings["neural_networks"][f"{neural_network_index}"]["seed"] = seed
        self.settings["neural_networks"][f"{neural_network_index}"]["index"] = neural_network_index

        return method_name, seed, neural_network

    def add_callback(self, callback):
        self.callbacks.append(callback)
