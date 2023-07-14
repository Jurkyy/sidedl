import tensorflow.keras as tk
import tensorflow as tf
import numpy as np


class AccuracyWrapper:

    def __init__(self, settings, n_profiling_traces, n_validation_traces):
        self.settings = settings
        self.n_profiling_traces = n_profiling_traces
        self.n_validation_traces = n_validation_traces
        self.history = []
        self.objective_history = {}
        self.epoch_profiling = 0
        self.epoch_validation = 0
        self.batch = 0
        self.accuracy_sum = 0
        self.y_true_acc = []
        self.y_pred_acc = []
        self.number_of_batches_training = np.ceil(self.n_profiling_traces / self.settings["batch_size"])
        self.number_of_batches_validation = np.ceil(self.n_validation_traces / self.settings["batch_size"])

    def __initialize_objective_history(self):
        """ for a new model training, initialize history lists """
        if self.batch == 0 and self.epoch_profiling == 0 and self.epoch_validation == 0:
            self.objective_history["accuracy"] = []
            self.objective_history["val_accuracy"] = []

    def __append_prediction(self, y_true, y_pred):
        """ append predicted values from processed batch into the list """
        y_pred_batch = np.asarray(y_pred)
        y_true_batch = np.asarray(y_true)
        for i, y in enumerate(y_pred_batch):
            self.y_pred_acc.append(y_pred_batch[i])
            self.y_true_acc.append(y_true_batch[i])

    def __compute_accuracy(self):
        """ compute accuracy for the processed epoch """
        categorical_accuracy = tk.metrics.CategoricalAccuracy()
        categorical_accuracy.update_state(np.array(self.y_true_acc), np.array(self.y_pred_acc))
        accuracy = categorical_accuracy.result().numpy()
        self.accuracy_sum += accuracy.copy()
        return accuracy

    def __add_accuracy_to_history(self, validation=False):
        number_of_batches = self.number_of_batches_validation if validation else self.number_of_batches_training
        if self.batch == number_of_batches:

            """ if all batches are processed """
            accuracy = self.__compute_accuracy()

            """ append accuracy from the processed epoch into history """
            if validation:
                self.objective_history["val_accuracy"].append(self.accuracy_sum / self.batch)
            else:
                self.objective_history["accuracy"].append(self.accuracy_sum / self.batch)

            """ if training is done, append history list for all models """
            self.epoch_profiling += 1
            if self.epoch_profiling == self.settings["epochs"]:
                self.epoch_profiling = 0
                if validation:
                    self.history.append(self.objective_history)

            """ reset values as training is finished """
            self.__reset_values()

            return accuracy
        else:
            return 0

    def __reset_values(self):
        """ reset values as training is finished """
        if self.epoch_profiling == self.settings["epochs"]:
            self.epoch_profiling = 0
        self.batch = 0
        self.accuracy_sum = 0
        self.y_pred_acc = []
        self.y_true_acc = []

    def calculate_accuracy(self, y_true, y_pred):
        """ this function is called for each processed batch (training and validation sets). """
        self.__initialize_objective_history()
        self.__append_prediction(y_true, y_pred)
        self.batch += 1
        if y_true[:, self.settings["classes"]:][0][0] == 1:
            """ if this is the validation set"""
            accuracy = self.__add_accuracy_to_history(validation=True)
        else:
            """ if this is the training set"""
            accuracy = self.__add_accuracy_to_history()

        return np.float32(np.array([accuracy]))

    @tf.function
    def tf_calculate_accuracy(self, y_true, y_pred):
        _ret = tf.numpy_function(self.calculate_accuracy, [y_true, y_pred], tf.float32)
        return _ret
