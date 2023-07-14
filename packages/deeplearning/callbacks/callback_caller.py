from tensorflow.keras.callbacks import *
from packages.deeplearning.callbacks.callback_visualization import *
import enum


class CallbackEnum(enum.Enum):
    KERAS_CSV_LOGGER = "keras_csv_logger"
    KERAS_MODEL_CHECK_POINT = "keras_model_check_point"
    KERAS_EARLY_STOPPING = "keras_early_stopping"
    KERAS_LEARNING_RATE_SCHEDULER = "keras_learning_rate_scheduler"
    KERAS_REDUCE_LEARNING_RATE_ON_PLATEAU = "keras_reduce_learning_rate_on_plateau"
    VISUALIZATION = "visualization"
    EARLY_STOPPING = "early_stopping"
    METRIC_MONITORING = "metric_monitoring"
    NEURON_ACTIVITY_RATE = "neuron_activity_rate"
    NEURON_ACTIVATIONS = "neuron_activations"
    WEIGHTS_AND_BIAS = "weights_and_biases"
    CONFUSION_MATRIX = "confusion_matrix"
    CUSTOM = "custom"


""" Keras builtin callbacks """


def create_keras_csv_logger_callback(**kwargs):
    filename = None
    separator = ','
    append = False
    if "filename" in kwargs:
        filename = kwargs["filename"]
    if "separator" in kwargs:
        separator = kwargs["separator"]
    if "append" in kwargs:
        append = kwargs["append"]

    return CSVLogger(filename, separator=separator, append=append)


def create_keras_model_check_point_callback(**kwargs):
    filepath = None
    monitor = "val_loss"
    verbose = 0
    save_best_only = False
    save_weights_only = False
    mode = "auto"
    save_freq = "epoch"

    if "filepath" in kwargs:
        filepath = kwargs["filepath"]
    if "monitor" in kwargs:
        monitor = kwargs["monitor"]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    if "save_best_only" in kwargs:
        save_best_only = kwargs["save_best_only"]
    if "save_weights_only" in kwargs:
        save_weights_only = kwargs["save_weights_only"]
    if "mode" in kwargs:
        mode = kwargs["mode"]
    if "save_freq" in kwargs:
        save_freq = kwargs["save_freq"]

    return ModelCheckpoint(filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only,
                           save_weights_only=save_weights_only, mode=mode, save_freq=save_freq)


def create_keras_early_stopping_callback(**kwargs):
    monitor = "val_loss"
    min_delta = 0
    patience = 0
    verbose = 0
    mode = "auto"
    baseline = None
    restore_best_weights = False

    if "monitor" in kwargs:
        monitor = kwargs["monitor"]
    if "min_delta" in kwargs:
        min_delta = kwargs["min_delta"]
    if "patience" in kwargs:
        patience = kwargs["patience"]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    if "mode" in kwargs:
        mode = kwargs["mode"]
    if "baseline" in kwargs:
        baseline = kwargs["baseline"]
    if "restore_best_weights" in kwargs:
        restore_best_weights = kwargs["restore_best_weights"]
    return EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline,
                         restore_best_weights=restore_best_weights)


def create_keras_learning_rate_scheduler_callback(**kwargs):
    schedule = None
    verbose = 0
    if "schedule" in kwargs:
        schedule = kwargs["schedule"]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    return LearningRateScheduler(schedule=schedule, verbose=verbose)


def create_keras_reduce_learning_rate_on_plateau_callback(**kwargs):
    monitor = "val_loss"
    factor = 0.1
    patience = 10
    verbose = 0
    mode = "auto"
    min_delta = 0.0001
    cooldown = 0
    min_lr = 0
    if "monitor" in kwargs:
        monitor = kwargs["monitor"]
    if "factor" in kwargs:
        factor = kwargs["factor"]
    if "patience" in kwargs:
        patience = kwargs["patience"]
    if "verbose" in kwargs:
        verbose = kwargs["verbose"]
    if "mode" in kwargs:
        mode = kwargs["mode"]
    if "min_delta" in kwargs:
        min_delta = kwargs["min_delta"]
    if "cooldown" in kwargs:
        cooldown = kwargs["cooldown"]
    if "min_lr" in kwargs:
        min_lr = kwargs["min_lr"]
    return ReduceLROnPlateau(monitor=monitor, factor=factor, patience=patience, verbose=verbose, mode=mode, min_delta=min_delta,
                             cooldown=cooldown, min_lr=min_lr)


""" SideDL callbacks """


def create_visualization_callback(settings, **kwargs):
    dataset = None
    if "dataset" in kwargs:
        dataset = kwargs["dataset"]
    return CallbackVisualization(settings, dataset=dataset)


def create_early_stopping_callback(settings):
    pass


def create_metric_monitoring_callback(settings):
    pass


def create_neuron_activity_rate_callback(settings):
    pass


def create_neuron_activations_callback(settings):
    pass


def create_weights_and_biases_callbacks(settings):
    pass


def create_confusion_matrix_callbacks(settings):
    pass


""" Custom callbacks """


def create_custom_callback(settings):
    pass


callback_functions = {
    "keras_csv_logger": create_keras_csv_logger_callback,
    "keras_model_check_point": create_keras_model_check_point_callback,
    "keras_early_stopping": create_keras_early_stopping_callback,
    "keras_learning_rate_scheduler": create_keras_learning_rate_scheduler_callback,
    "keras_reduce_learning_rate_on_plateau": create_keras_reduce_learning_rate_on_plateau_callback,
    "visualization": create_visualization_callback,
    "early_stopping": create_early_stopping_callback,
    "metric_monitoring": create_metric_monitoring_callback,
    "neuron_activity_rate": create_neuron_activity_rate_callback,
    "neuron_activations": create_neuron_activations_callback,
    "weights_and_biases": create_weights_and_biases_callbacks,
    "confusion_matrix": create_confusion_matrix_callbacks,
    "custom": create_custom_callback
}
