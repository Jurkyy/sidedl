from packages.deeplearning.optimization.optimization_search import OptimizationSearch
import glob
import random


class RandomSearch(OptimizationSearch):

    def __init__(self, settings, search_space, max_trials=None):
        super().__init__(settings, search_space)
        self.number_of_models = None
        self.max_trials = max_trials

    def __generate_random_hyperparameters(self):
        """ generate combinations for grid search """

        hp = {}
        for hp_name, hp_options in self.search_space.items():
            if isinstance(hp_options, dict):
                if "min" in list(hp_options.keys()):
                    hp[hp_name] = random.randrange(hp_options['min'], hp_options['max'] + hp_options['step'], hp_options['step'])
            elif isinstance(hp_options, list):
                hp[hp_name] = random.choice(hp_options)
            elif isinstance(hp_options, str):
                hp[hp_name] = hp_options

        return hp

    def __check_for_additional_models(self):
        """
        Check for additional models to be accessed for SCA metrics during search. This are models generated, e.g., during early stopping.
        """

        filepath = "models_folder/"
        return glob.glob(f"{filepath}/*_{self.search_id}.h5")

    def run(self, dataset, db_label="", callbacks=None, loss_function=None):
        """ Run the main grid search process """

        for model_index in range(self.number_of_models):
            search_combination = self.__generate_random_hyperparameters()
            self.generate_neural_network(hyperparameters=search_combination, loss_function=loss_function)
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
