from packages.deeplearning.optimization.optimization_search import OptimizationSearch
import itertools
import sys
import glob


class GridSearch(OptimizationSearch):

    def __init__(self, settings, search_space, max_trials=None):
        super().__init__(settings, search_space)
        self.number_of_models = None
        self.max_trials = max_trials

    def __generate_hyperparameters_combinations(self):
        """ generate combinations for grid search """

        hp_to_search = {}
        hp_to_adjust = {}
        for hp, hp_value in self.search_space.items():
            if type(hp_value) is str:
                """ in case the hyperparameter option is a string """
                hp_to_adjust[hp] = hp_value
            else:
                hp_to_search[hp] = hp_value
        keys, values = zip(*hp_to_search.items())
        if self.max_trials is not None:
            self.__check_search_space_size(values)

        search_hp_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        for idx in range(len(search_hp_combinations)):
            for hp, hp_value in hp_to_adjust.items():
                search_hp_combinations[idx][hp] = hp_value
        self.number_of_models = len(search_hp_combinations)
        return search_hp_combinations

    def __check_search_space_size(self, values):
        """ Check is number of models to search is higher than the given search space size """

        search_space_size = 1
        for v in values:
            search_space_size *= len(v)
            if search_space_size > self.max_trials:
                print("Search space is too large.")
                sys.exit()

    def __check_for_additional_models(self):
        """ Check for additional models to be accessed for SCA metrics during search """

        filepath = "models_folder/"
        return glob.glob(f"{filepath}/*_{self.search_id}.h5")

    def run(self, dataset, db_label="", callbacks=None, loss_function=None):
        """ Run the main grid search process """

        search_hp_combinations = self.__generate_hyperparameters_combinations()

        for model_index in range(self.number_of_models):
            self.generate_neural_network(hyperparameters=search_hp_combinations[model_index], loss_function=loss_function)
            self.train_neural_network(dataset, callbacks)
            self.compute_sca_metrics(dataset.x_validation, dataset.nt_sca_metrics_validation, dataset.get_validation_key_byte(),
                                     db_label=db_label)
            self.compute_sca_metrics(dataset.x_attack, dataset.nt_sca_metrics_attack, dataset.get_attack_key_byte(), db_label=db_label)

            """ In case the callbacks generate new models to be accessed (e.g., through early-stopping) """
            model_filenames = self.__check_for_additional_models()
            for model_filename in model_filenames:
                # TODO: implement a way to add db_label
                self.model.load_weights(model_filename)
                self.compute_sca_metrics(dataset.x_validation, dataset.nt_sca_metrics_validation, dataset.get_validation_key_byte(),
                                         db_label=db_label)
            self.check_best_model()
