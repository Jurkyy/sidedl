from components.decorators.database_decorators import *
from packages.sidechannel.metrics.sca_metrics import SCAMetrics
from packages.sidechannel.profiling.deep_learning import Profiling
from packages.deeplearning.neural_networks.multi_layer_perceptron import MultiLayerPerceptron
from packages.deeplearning.neural_networks.convolutional_neural_network import ConvolutionalNeuralNetwork
import random
import enum


class NeuralNetworksEnum(enum.Enum):
    MULTILAYER_PERCEPTRON = "mlp"
    CONVOLUTIONAL_NEURAL_NETWORK = "cnn"
    RECURRENT_NEURAL_NETWORK = "rnn"


neural_network_type_dict = {
    "mlp": MultiLayerPerceptron,
    "cnn": ConvolutionalNeuralNetwork
}


class OptimizationSearch:

    def __init__(self, settings, search_space):
        self.settings = settings
        self.search_space = search_space
        self.model = None
        self.labels_key_guess = None
        self.search_id = random.getrandbits(128)  # TODO pass this variable to callbacks
        self.model_structure = search_space["model_structure"]

    @keras_neural_network_database
    def generate_neural_network(self, hyperparameters=None, loss_function=None, metrics=None, hyperparameters_func=None):
        if hyperparameters_func is not None:
            hyperparameters = hyperparameters_func
        nn_function = neural_network_type_dict[hyperparameters["neural_network"]]
        model_name, seed, self.model = nn_function(self.settings, hyperparameters, self.model_structure,
                                                   loss_function=loss_function, metrics=metrics).create()
        return model_name, seed, self.model

    @metrics_database
    def train_neural_network(self, dataset, callbacks):
        profiling = Profiling(self.settings, dataset)
        profiling.train_model(self.model, callbacks)

    @sca_metrics_database
    def compute_sca_metrics(self, traces, nt_metrics, correct_key, db_label=""):
        sca_metrics_obj = SCAMetrics(self.model, traces, nt_metrics, self.labels_key_guess, correct_key)
        return sca_metrics_obj.run(self.settings["key_rank_executions"])

    @update_best_model_database
    def check_best_model(self):
        pass
