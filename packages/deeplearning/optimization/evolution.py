from packages.deeplearning.optimization.optimization_search import OptimizationSearch


class Evolution(OptimizationSearch):

    def __init__(self, settings, search_space):
        super().__init__(settings)
        self.search_space = search_space

    def generate_neural_network(self):
        pass

    def train_neural_network(self):
        pass

    def compute_sca_metrics(self):
        pass

    def check_best_model(self):
        pass

    def run(self):
        self.generate_neural_network()
        self.train_neural_network()
        self.compute_sca_metrics()
        self.check_best_model()
