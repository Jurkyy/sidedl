from components import Component
from components.decorators.database_decorators import dataset_database
from packages.datasets.h5.h5 import read_h5


class Dataset(Component):

    def __init__(self, settings):
        super().__init__(settings)
        self.datasets_dict = {}

    @dataset_database
    def add_dataset(self, dataset_filepath, dataset_name):
        self.datasets_dict["name"] = dataset_name
        self.datasets_dict["filepath"] = dataset_filepath
