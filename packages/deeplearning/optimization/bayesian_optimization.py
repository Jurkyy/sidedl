from packages.deeplearning.metrics.guessing_entropy_wrapper import GuessingEntropyWrapper
from packages.deeplearning.metrics.guessing_entropy import GuessingEntropy
from packages.deeplearning.metrics.accuracy_wrapper import AccuracyWrapper
from packages.deeplearning.metrics.accuracy import Accuracy
from packages.deeplearning.metrics.cross_entropy_wrapper import CrossEntropyWrapper
from packages.deeplearning.metrics.cross_entropy import CrossEntropy
from packages.deeplearning.optimization.optimization_search import OptimizationSearch
from keras_tuner.tuners import BayesianOptimization as BO
from tensorflow.keras.callbacks import Callback
import keras_tuner as kt
import numpy as np
from datetime import datetime

direction_dict = {
    "guessing_entropy": "min",
    "accuracy": "max",
    "loss": "min"
}

objective_dict = {
    "guessing_entropy": "val_guessing_entropy",
    "accuracy": "val_accuracy",
    "loss": "val_loss"
}


class BayesianOptimization(OptimizationSearch):

    def __init__(self, settings, search_space, max_trials=None):
        super().__init__(settings, search_space)
        self.number_of_models = None
        self.max_trials = max_trials

    def __generate_random_hyperparameters(self, hp_bo):
        """ generate combinations for grid search """

        hp = {}
        for hp_name, hp_options in self.search_space.items():
            if isinstance(hp_options, dict):
                if "min" in list(hp_options.keys()):
                    hp[hp_name] = hp_bo.Int("neurons", min_value=self.search_space[hp_name]["min"],
                                            max_value=self.search_space[hp_name]["max"], step=self.search_space[hp_name]["step"])
            elif isinstance(hp_options, list):
                hp[hp_name] = hp_bo.Choice(hp_name, self.search_space[hp_name])
            elif isinstance(hp_options, str):
                hp[hp_name] = hp_options

        return hp

    def __get_wrapper(self, labels_key_guess, n_profiling_traces, n_validation_traces):
        wrapper_dict = {
            "guessing_entropy": GuessingEntropyWrapper(self.settings, labels_key_guess, n_validation_traces),
            "accuracy": AccuracyWrapper(self.settings, n_profiling_traces, n_validation_traces),
            "loss": CrossEntropyWrapper(self.settings, n_profiling_traces, n_validation_traces)
        }
        return wrapper_dict[self.settings["optimization"]["metric"]]

    def __get_validation_metric(self, wrapper):
        validation_metric_dict = {
            "guessing_entropy": ["accuracy", GuessingEntropy(wrapper, None)],
            "accuracy": [Accuracy(wrapper)],
            "loss": ["accuracy", CrossEntropy(wrapper)]
        }
        return validation_metric_dict[self.settings["optimization"]["metric"]]

    def __get_validation_data(self, dataset):
        """ Concatenate categorical labels with additional element to differentiate between profiling and validation set """
        if self.settings["split_test_set"] or self.settings["split_training_set"]:
            y_validation = np.concatenate((dataset.y_validation, np.ones((len(dataset.y_validation), 1))), axis=1)
            validation_set = (dataset.x_validation, y_validation)
        else:
            y_attack = np.concatenate((dataset.y_attack, np.ones((len(dataset.y_attack), 1))), axis=1)
            validation_set = (dataset.x_attack, y_attack)
        return validation_set

    def run(self, dataset, db_label="", callbacks=None, loss_function=None):
        """ Run the main grid search process """

        objective = objective_dict[self.settings["optimization"]["metric"]]
        direction = direction_dict[self.settings["optimization"]["metric"]]

        now = datetime.now()
        now_str = now.strftime("%d_%m_%Y_%H_%M_%S")

        # TODO: dataset details must come from container
        wrapper = self.__get_wrapper(self.labels_key_guess, len(dataset.x_profiling), len(dataset.x_validation))
        metrics = self.__get_validation_metric(wrapper)
        model = self.generate_neural_network(hyperparameters_func=self.__generate_random_hyperparameters, loss_function=loss_function,
                                             metrics=metrics)
        tuner = BO(model,
                   objective=kt.Objective(objective, direction=direction),
                   max_trials=self.max_trials,
                   executions_per_trial=1,
                   directory=f"BO_{now_str}",
                   project_name=f"{self.settings['resources_root_folder']}models\\bayesian_optimization",
                   overwrite=True)

        save_metrics_callback = SaveMetrics(self.settings, dataset)

        tuner.search_space_summary()
        tuner.search(x=dataset.x_profiling,
                     y=np.concatenate((dataset.y_profiling, np.zeros((len(dataset.y_profiling), 1))), axis=1),
                     epochs=self.settings["epochs"],
                     batch_size=self.settings["batch_size"],
                     validation_data=self.__get_validation_data(dataset),
                     verbose=2,
                     callbacks=[save_metrics_callback])
        tuner.results_summary()

        self.check_best_model()


class SaveMetrics(Callback):
    def __init__(self, settings, dataset):
        super().__init__()
        self.settings = settings
        self.dataset = dataset

    def on_train_end(self, logs=None):
        db_label = ""
        OptimizationSearch.compute_sca_metrics(self.dataset.x_validation, self.dataset.nt_sca_metrics_validation,
                                               self.dataset.get_validation_key_byte(), db_label=db_label)
        db_label = ""
        OptimizationSearch.compute_sca_metrics(self.dataset.x_attack, self.dataset.nt_sca_metrics_attack,
                                               self.dataset.get_attack_key_byte(), db_label=db_label)
