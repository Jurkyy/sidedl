from tensorflow.keras.optimizers import *
from tensorflow.keras.layers import *
import enum


class HyperparametersEnum(enum.Enum):
    """ multi-layer perceptrons """
    NEURONS = "neurons",
    DENSE_LAYERS = "dense_layers",
    """ convolution_neural_network """
    CONV_LAYERS = "conv_layers",
    FILTERS = "filters",
    KERNEL_SIZE = "kernel_size",
    STRIDES = "strides",
    POOL_SIZE = "pooling_size",
    POOL_STRIDES = "pooling_strides",
    POOL_TYPE = "pooling_type"
    """ generic """
    ACTIVATION = "activation",
    KERNEL_INITIALIZER = "kernel_initializer",
    BIAS_INITIALIZER = "bias_initializer",
    KERNEL_REGULARIZER = "kernel_regularizer",
    BIAS_REGULARIZER = "bias_regularizer",
    ACTIVITY_REGULARIZER = "activity_regularizer",
    KERNEL_CONSTRAINT = "kernel_constraint",
    BIAS_CONSTRAINT = "bias_constraint",
    LEARNING_RATE = "learning_rate",
    OPTIMIZER = "optimizer",
    DROPOUT_RATE = "dropout_rate"
    SEED = "seed"


dense_layer_hyperparameters = [
    HyperparametersEnum.KERNEL_INITIALIZER, HyperparametersEnum.BIAS_INITIALIZER, HyperparametersEnum.KERNEL_REGULARIZER,
    HyperparametersEnum.BIAS_REGULARIZER, HyperparametersEnum.ACTIVITY_REGULARIZER, HyperparametersEnum.KERNEL_CONSTRAINT,
    HyperparametersEnum.BIAS_CONSTRAINT
]

pooling_layer_type = {
    "average": AveragePooling1D,
    "max": MaxPool1D
}

optimizer_type = {
    "Adam": Adam,
    "Adamax": Adamax,
    "Nadam": Nadam,
    "RMSprop": RMSprop,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "SGD": SGD
}
