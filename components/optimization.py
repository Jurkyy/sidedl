from components import Component
from packages.deeplearning.optimization.grid_search import GridSearch


class Optimization(Component):

    def __init__(self, settings):
        super().__init__(settings)

    def create_grid_search(self, search_space, dataset, callbacks, loss_function):
        # TODO: dataset, callbacks and loss function should be taken from the container
        grid_search = GridSearch(self.settings, search_space)
        grid_search.run(dataset, callbacks=callbacks, loss_function=loss_function)

    def create_random_search(self, search_space):
        pass

    def create_bayesian_optimization_search(self, search_space):
        pass

    def create_evolution_search(self, search_space):
        pass

    def create_reinforcement_learning_search(self, search_space):
        pass
