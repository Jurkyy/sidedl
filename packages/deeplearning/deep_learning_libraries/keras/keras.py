import tensorflow.keras.backend as backend
import os
import json


def convert_keras_layer_name(layer_name):
    # TODO: add all keras layers to this list
    if "Dense" in layer_name:
        return "Dense"
    elif "Conv" in layer_name:
        return "Convolution"
    elif "Pool" in layer_name:
        return "Pooling"
    elif "Flatten" in layer_name:
        return "Flatten"
    elif "BatchNorm" in layer_name:
        return "BatchNormalization"
    elif "Dropout" in layer_name:
        return "Dropout"
    elif "GaussianNoise" in layer_name:
        return "GaussianNoise"
    elif "Activation" in layer_name:
        return "Activation"


def get_layer_units(layer):
    units = {}
    if "units" in layer:
        units["neurons"] = layer["units"]
    if "filters" in layer:
        units["filters"] = layer["filters"]
    if "kernel_size" in layer:
        units["kernel_size"] = layer["kernel_size"]
    if "strides" in layer:
        units["strides"] = layer["strides"]
    if "pool_size" in layer:
        units["pool_size"] = layer["strides"]
    if "padding" in layer:
        units["padding"] = layer["padding"]
    return units


class Keras:

    def __init__(self, model):
        self.model = model
        self.keras_model_dict = None
        self.side_dl_dict = None

    def load_model_h5(self, model_filename):
        """ Load model weights from .h5 keras model file """
        self.model.load_weights(model_filename)

    def save_model_h5(self, model_filename):
        """ Save model weights into .h5 file """
        self.model.save_weights(model_filename)

    def delete_model_h5(self, model_filename):
        """ Delete .h5 file from directory """
        os.remove(model_filename)

    def get_learning_rate(self):
        return backend.eval(self.model.optimizer.lr)

    def get_optimizer(self):
        return backend.eval(self.model.optimizer.__class__.__name__)

    def parse_keras_json_to_dict(self):
        """ Parse a keras json object into a Python dictionary with more simplification """
        model_json = json.loads(self.model.to_json())

        self.keras_model_dict = {"layers": {}}

        model_layers = model_json["config"]["layers"]
        model_dict_layers = []

        for layer in model_layers:

            """ Go through all the model layers """
            layer_config = layer["config"]
            model_dict_layer = {"class_name": layer["class_name"]}

            for config_name, config_value in layer_config.items():
                " add layer attributes to layer in dictionary "
                model_dict_layer[config_name] = config_value

                if isinstance(config_value, dict):
                    " in case an attribute is also a dictionary "
                    model_dict_layer[config_name] = {}
                    layer_attribute_config = config_value["config"]
                    for layer_attribute_key, layer_attribute_value in layer_attribute_config.items():
                        model_dict_layer[config_name][layer_attribute_key] = layer_attribute_value

            model_dict_layers.append(model_dict_layer)

        self.keras_model_dict["layers"] = model_dict_layers

    def convert_keras_to_sidedl_format(self):
        """ Convert keras model from dictionary into side_dl format """
        pass

    def convert_keras_to_hyperparameters_for_database(self):
        """ Convert keras model from dictionary into a dictionary for database format """
        pass
