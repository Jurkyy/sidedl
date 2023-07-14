import tensorflow as tf
import numpy as np


class GuessingEntropyWrapper:

    """ Class to compute validation guessing entropy during training """

    def __init__(self, settings, labels_key_hypothesis, n_validation_traces):
        self.settings = settings
        self.labels_key_hypothesis = labels_key_hypothesis
        self.n_validation_traces = n_validation_traces
        self.sk_acc = []
        self.history = []
        self.objective_history = {}
        self.epoch_validation = 0

    def __initialize_objective_history(self):
        """ for a new model training, initialize history lists """
        if self.epoch_validation == 0:
            self.objective_history["val_guessing_entropy"] = []

    def __append_prediction(self, y_pred):
        """ append predicted values from processed batch into the list """
        y_pred_batch = np.asarray(np.log(y_pred + 1e-36))
        for i in y_pred_batch:
            self.sk_acc.append(i)

    def __compute_guessing_entropy(self):
        """ compute accuracy for the processed epoch """
        self.sk_acc = np.asarray(self.sk_acc)
        key_ranking_sum = 0
        probabilities_kg_all_traces = np.zeros((self.n_validation_traces, 256))
        for index in range(self.n_validation_traces):
            probabilities_kg_all_traces[index] = self.sk_acc[index][
                np.asarray([int(leakage[index]) for leakage in self.labels_key_hypothesis[:]])
            ]

        for run in range(self.settings["key_rank_executions"]):
            r = np.random.choice(range(self.n_validation_traces), self.settings["key_rank_attack_traces"], replace=False)
            probabilities_kg_all_traces_shuffled = probabilities_kg_all_traces[r]
            key_probabilities = np.sum(probabilities_kg_all_traces_shuffled[:self.settings["key_rank_attack_traces"]], axis=0)
            key_probabilities_sorted = np.argsort(key_probabilities)[::-1]
            key_ranking_sum += list(key_probabilities_sorted).index(self.settings["good_key"]) + 1

        return key_ranking_sum / self.settings["key_rank_executions"]

    def __add_guessing_entropy_to_history(self, validation=False):
        if len(self.sk_acc) == self.n_validation_traces and validation:

            """ if all batches are processed """
            guessing_entropy = self.__compute_guessing_entropy()

            """ append accuracy from the processed epoch into history """
            self.objective_history["val_guessing_entropy"].append(guessing_entropy)

            """ if training is done, append history list for all models """
            self.epoch_validation += 1
            if self.epoch_validation == self.settings["epochs"]:
                self.history.append(self.objective_history)
                self.epoch_validation = 0

            return guessing_entropy
        else:
            return 0

    def calculate_guessing_entropy(self, y_true, y_pred):
        """ this function is called for each processed batch (training and validation sets). """
        self.__initialize_objective_history()
        if y_true[:, self.settings["classes"]:][0][0] == 1:
            """ if this is the validation set"""
            self.__append_prediction(y_pred)
            guessing_entropy = self.__add_guessing_entropy_to_history(validation=True)
        else:
            """ if this is the training set"""
            guessing_entropy = self.__add_guessing_entropy_to_history()

        return np.float32(np.array([guessing_entropy]))

    @tf.function
    def tf_calculate_guessing_entropy(self, y_true, y_pred):
        _ret = tf.numpy_function(self.calculate_guessing_entropy, [y_true, y_pred], tf.float32)
        return _ret
