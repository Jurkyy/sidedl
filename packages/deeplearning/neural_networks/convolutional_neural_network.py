from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
import tensorflow as tf
import numpy as np
import importlib
from packages.deeplearning.neural_networks.hyperparamaters import HyperparametersEnum as hp
from packages.deeplearning.neural_networks.hyperparamaters import *


class ConvolutionalNeuralNetwork:

    def __init__(self, settings, hyperparameters, model_structure, loss_function=None, metrics=None):
        self.settings = settings
        self.hyperparameters = hyperparameters
        self.loss_function = loss_function
        self.metrics = metrics
        self.model_structure = model_structure

    def get_random_seed(self):
        random_seed = self.hyperparameters[hp.SEED] if hp.SEED in self.hyperparameters else np.random.randint(1048576)
        tf.random.set_seed(random_seed)
        return random_seed

    def get_convolution_hyperparameter(self, conv_layer, hp_list, hp_type):

        label = f"{hp_type}_{conv_layer + 1}"

        if conv_layer > 0:
            if self.hyperparameters[label] == "equal_from_previous_convolution":
                new_hp = hp_list[conv_layer - 1]
            elif self.hyperparameters[label] == "double_from_previous_convolution":
                new_hp = hp_list[conv_layer - 1] * 2
            elif self.hyperparameters[label] == "half_from_previous_convolution":
                new_hp = int(hp_list[conv_layer - 1] / 2) if hp_list[conv_layer - 1] > 1 else 1
            else:
                new_hp = self.hyperparameters[label]
        else:
            new_hp = self.hyperparameters[label]
        hp_list.append(new_hp)
        return hp_list

    def get_pooling_hyperparameter(self, conv_layer, hp_list, hp_type):

        label = f"{hp_type}_{conv_layer + 1}"

        if conv_layer > 0:
            if self.hyperparameters[label] == "equal_from_previous_pooling":
                value = hp_list[conv_layer - 1]
            elif self.hyperparameters[label] == "double_from_previous_pooling":
                value = hp_list[conv_layer - 1] * 2
            elif self.hyperparameters[label] == "half_from_previous_pooling":
                value = int(hp_list[conv_layer - 1] / 2) if hp_list[conv_layer - 1] > 1 else 1
            else:
                value = self.hyperparameters[label]
        else:
            value = self.hyperparameters[label]
        hp_list.append(value)
        return hp_list

    def get_pooling_type(self, conv_layer, hp_list, hp_type):

        label = f"{hp_type}_{conv_layer + 1}"

        if conv_layer > 0:
            if self.hyperparameters[label] == "equal_from_previous_pooling":
                hp_list.append(hp_list[conv_layer + 1])
            else:
                hp_list.append(self.hyperparameters[label])
        else:
            hp_list.append(self.hyperparameters[label])
        return hp_list

    def create_convolution_and_pooling_layers(self, x):
        kernels = []
        strides = []
        filters = []
        pooling_types = []
        pooling_sizes = []
        pooling_strides = []
        for conv_layer_index in range(self.hyperparameters[hp.CONV_LAYERS]):
            kernels = self.get_convolution_hyperparameter(conv_layer_index, kernels, hp.KERNEL_SIZE)
            strides = self.get_convolution_hyperparameter(conv_layer_index, strides, hp.STRIDES)
            filters = self.get_convolution_hyperparameter(conv_layer_index, filters, hp.FILTERS)
            pooling_sizes = self.get_pooling_hyperparameter(conv_layer_index, pooling_sizes, hp.POOL_SIZE)
            pooling_strides = self.get_pooling_hyperparameter(conv_layer_index, pooling_strides, hp.POOL_STRIDES)
            pooling_types = self.get_pooling_type(conv_layer_index, pooling_types, hp.POOL_TYPE)

        for conv_layer_index in range(self.hyperparameters[hp.CONV_LAYERS]):
            if conv_layer_index == 0:
                if self.model_structure["use_pooling_before_first_dense"]:
                    if all(e in self.hyperparameters for e in [f"{hp.POOL_SIZE}_0", f"{hp.POOL_STRIDES}_0", f"{hp.POOL_TYPE}_0"]):
                        pooling_sizes_0 = self.hyperparameters[f"{hp.POOL_SIZE}_0"]
                        pooling_stride_0 = self.hyperparameters[f"{hp.POOL_STRIDES}_0"]
                        pooling_type_0 = self.hyperparameters[f"{hp.POOL_TYPE}_0"]
                        x = pooling_layer_type[pooling_type_0](pool_size=pooling_sizes_0, strides=pooling_stride_0, padding="same", )(x)
            x = Conv1D(kernel_size=kernels[conv_layer_index], strides=strides[conv_layer_index], filters=filters[conv_layer_index],
                       activation=self.hyperparameters[hp.ACTIVATION], padding="same")(x)

            if self.model_structure["use_batch_norm_after_convolution"] or (
                    self.model_structure["use_batch_norm_before_pooling"] and self.model_structure["use_pooling_after_convolution"]):
                x = BatchNormalization()(x)

            if self.model_structure["use_pooling_after_convolution"]:
                x = pooling_layer_type[pooling_types[conv_layer_index]](pool_size=pooling_sizes[conv_layer_index],
                                                                        strides=pooling_strides[conv_layer_index], padding="same", )(x)
            if self.model_structure["use_batch_norm_after_pooling"]:
                x = BatchNormalization()(x)
        return Flatten()(x)

    def create_dense_layer(self):
        hp_dict = {}
        for hp_name in dense_layer_hyperparameters:
            hp_dict[hp_name] = self.hyperparameters[hp_name]
        return Dense(self.hyperparameters[hp.NEURONS], activation=self.hyperparameters[hp.ACTIVATION], **hp_dict)

    def create_fully_connected_layers(self, x):
        for layer_index in range(self.hyperparameters[hp.DENSE_LAYERS]):
            if self.model_structure["use_dropout_before_dense_layer"]:
                x = Dropout(self.hyperparameters[hp.DROPOUT_RATE])(x)
                x = self.create_dense_layer()(x)
            elif self.model_structure["use_dropout_after_dense_layer"]:
                x = self.create_dense_layer()(x)
                x = Dropout(self.hyperparameters[hp.DROPOUT_RATE])(x)
        return x

    def create_output_layer(self, x):
        if self.model_structure["use_dropout_before_dense_layer"]:
            x = Dropout(self.hyperparameters[hp.DROPOUT_RATE])(x)
        return Dense(self.settings["classes"], activation='softmax')(x)

    def create_optimizer(self, optimizer, learning_rate):
        if optimizer == "SGD":
            return optimizer_type[optimizer](lr=learning_rate, momentum=0.9, nesterov=True)
        return optimizer_type[optimizer](lr=learning_rate)

    def create_loss_function(self):
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

        random_seed = self.get_random_seed()

        input_shape = (self.settings["number_of_samples"], 1)
        input_features = Input(shape=input_shape)

        x = self.create_convolution_and_pooling_layers(input_features)
        x = self.create_fully_connected_layers(x)
        output = self.create_output_layer(x)

        model = Model(input_features, output, name='cnn_name')  # TODO: add model name
        model.compile(loss=self.create_loss_function(),
                      optimizer=self.create_optimizer(self.hyperparameters[hp.OPTIMIZER], self.hyperparameters[hp.LEARNING_RATE]),
                      metrics=['accuracy'] if self.metrics is None else self.metrics)

        model_name = ""
        return model_name, random_seed, model
