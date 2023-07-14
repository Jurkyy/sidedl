from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
import numpy as np
import importlib
from packages.deeplearning.neural_networks.hyperparamaters import HyperparametersEnum as hp
from packages.deeplearning.neural_networks.hyperparamaters import *


class MultiLayerPerceptron:

    def __init__(self, settings, hyperparameters, model_structure, loss_function=None, metrics=None):
        self.settings = settings
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.metrics = metrics
        self.model_structure = model_structure

    def __get_random_seed(self):
        random_seed = self.hyperparameters[hp.SEED] if hp.SEED in self.hyperparameters else np.random.randint(1048576)
        tf.random.set_seed(random_seed)
        return random_seed

    def __create_pooling_layers(self, x):
        if self.model_structure["use_pooling_before_first_dense"]:
            if all(e in self.hyperparameters for e in [f"{hp.POOL_SIZE}_0", f"{hp.POOL_STRIDES}_0", f"{hp.POOL_TYPE}_0"]):
                pooling_sizes_0 = self.hyperparameters[f"{hp.POOL_SIZE}_0"]
                pooling_stride_0 = self.hyperparameters[f"{hp.POOL_STRIDES}_0"]
                pooling_type_0 = self.hyperparameters[f"{hp.POOL_TYPE}_0"]
                x = pooling_layer_type[pooling_type_0](pool_size=pooling_sizes_0, strides=pooling_stride_0,
                                                       input_shape=(self.settings["number_of_samples"], 1))(x)
                return Flatten()(x)
            return x
        else:
            return x

    def __create_dense_layer(self):
        hp_dict = {}
        for hp_name in dense_layer_hyperparameters:
            hp_dict[hp_name] = self.hyperparameters[hp_name]
        return Dense(self.hyperparameters[hp.NEURONS], activation=self.hyperparameters[hp.ACTIVATION], **hp_dict)

    def __create_fully_connected_layers(self, x):
        for layer_index in range(self.hyperparameters[hp.DENSE_LAYERS]):
            if self.model_structure["use_dropout_before_dense_layer"]:
                x = Dropout(self.hyperparameters[hp.DROPOUT_RATE])(x)
                x = self.__create_dense_layer()(x)
            elif self.model_structure["use_dropout_after_dense_layer"]:
                x = self.__create_dense_layer()(x)
                x = Dropout(self.hyperparameters[hp.DROPOUT_RATE])(x)
        return x

    def __create_output_layer(self, x):
        if self.model_structure["use_dropout_before_dense_layer"]:
            x = Dropout(self.hyperparameters[hp.DROPOUT_RATE])(x)
        return Dense(self.settings["classes"], activation='softmax')(x)

    def __create_optimizer(self, optimizer, learning_rate):
        if optimizer == "SGD":
            return optimizer_type[optimizer](lr=learning_rate, momentum=0.9, nesterov=True)
        return optimizer_type[optimizer](lr=learning_rate)

    def __create_loss_function(self):
        if self.loss_function is not None:
            class_inverted = self.loss_function["class"][::-1]
            module_name = class_inverted.partition(".")[2][::-1]
            loss_function_name = class_inverted.partition(".")[0][::-1]
            module_name = importlib.import_module(module_name)
            custom_loss_class = getattr(module_name, loss_function_name)
            return custom_loss_class(self.settings, parameters=self.loss_function['parameters'])
        else:
            return 'categorical_crossentropy'

    def create(self):

        random_seed = self.__get_random_seed()

        input_shape = (self.settings["number_of_samples"])
        input_features = Input(shape=input_shape)
        x = self.__create_pooling_layers(input_features)
        x = self.__create_fully_connected_layers(x)
        output = self.__create_output_layer(x)

        model = Model(input_features, output, name='mlp_name')  # TODO: add model name
        model.compile(loss=self.__create_loss_function(),
                      optimizer=self.__create_optimizer(self.hyperparameters[hp.OPTIMIZER], self.hyperparameters[hp.LEARNING_RATE]),
                      metrics=['accuracy'] if self.metrics is None else self.metrics)

        model_name = ""
        return model_name, random_seed, model
