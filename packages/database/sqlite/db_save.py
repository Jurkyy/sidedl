import pandas as pd
from sqlalchemy.orm import sessionmaker
import packages.database.sqlite.db_create_table as tables
from packages.database.sqlite.db_create_table import *


class DBSave:

    def __init__(self, database_path):
        self.db_save = None
        self.analysis_id = None
        self.engine = create_engine('sqlite:///{}'.format(database_path), echo=False)
        self.metadata = MetaData(self.engine)
        self.session = sessionmaker(bind=self.engine)()
        tables.base().metadata.create_all(self.engine)

    def insert(self, row):
        """
        :param row: object to be inserted in the database table
        :return: id of the last inserted row
        """
        self.session.add(row)
        self.session.commit()
        return row.id

    def insert_if_not_exists(self, row, table, **kwargs):
        """
        :param row: object to be inserted in the database table
        :return: id of the last inserted row
        """

        existing_row = self.session.query(table).filter_by(**kwargs).first()
        if existing_row is None:
            self.session.add(row)
            self.session.commit()
            return row.id
        else:
            return existing_row.id

    def save_analysis(self, db_filename, dataset, settings, elapsed_time=0):
        """
        :param db_filename: string with database name
        :param dataset: string with dataset file name
        :param settings: dictionary containing analysis settings
        :param elapsed_time: elapsed time for analysis
        :return: id
        """
        new_insert = Analysis(db_filename=db_filename, dataset=dataset, settings=settings, elapsed_time=elapsed_time, deleted=False)
        self.analysis_id = self.insert(new_insert)
        return self.analysis_id

    def save_neural_network(self, name, seed):
        """
        :param name: neural network name
        :param seed: neural network seed for weights and biases initialization
        :return: id
        """
        new_insert = NeuralNetwork(name=name, seed=seed, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_layer(self, index, name, units, layer_type_id, activation_id, weight_initializer_id, bias_initializer_id, neural_network_id):
        """
        :param index: layer index
        :param name: layer name
        :param units: dictionary containing layer units.
        For dense layers: units = {"neurons": 100}
        For convolution layers: units = {"filters": 32, "kernel_size": 10, "strides": 1}
        For pooling layers: units = {"pool_size": 2, "strides": 2}
        :param layer_type_id: layer type id
        :param activation_id: activation_id
        :param weight_initializer_id: weight_initializer_id
        :param bias_initializer_id: bias_initializer_id
        :param neural_network_id: layer neural_network_id
        :return: id
        """
        new_insert = Layer(index=index, name=name, units=units, layer_type_id=layer_type_id, activation_id=activation_id,
                           weight_initializer_id=weight_initializer_id, bias_initializer_id=bias_initializer_id,
                           neural_network_id=neural_network_id)
        return self.insert(new_insert)

    def save_layer_type(self, layer_type):
        """
        :param layer_type: layer type
        :return: id
        """
        new_insert = LayerType(type=layer_type)
        return self.insert_if_not_exists(new_insert, LayerType, type=layer_type)

    def save_activation(self, activation):
        """
        :param activation: layer activation
        :return: id
        """
        new_insert = Activation(activation=activation)
        return self.insert_if_not_exists(new_insert, Activation, activation=activation)

    def save_weight_initializer(self, weight_initializer):
        """
        :param weight_initializer: weight_initializer
        :return: id
        """
        new_insert = WeightInitializer(weight_initializer=weight_initializer)
        return self.insert_if_not_exists(new_insert, WeightInitializer, weight_initializer=weight_initializer)

    def save_bias_initializer(self, bias_initializer):
        """
        :param bias_initializer: bias_initializer
        :return: id
        """
        new_insert = BiasInitializer(bias_initializer=bias_initializer)
        return self.insert_if_not_exists(new_insert, BiasInitializer, bias_initializer=bias_initializer)

    def save_hyperparameters(self, hyperparameters):
        """
        :param hyperparameters: dictionary of hyper-parameters
        :return: id
        """
        new_insert = HyperParameter(hyperparameters=hyperparameters, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_leakage_model(self, leakage_model):
        """
        :param leakage_model: dictionary containing leakage model parameters
        :return: id
        """
        new_insert = LeakageModel(leakage_model=leakage_model, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_dataset(self, name, filepath):
        """
        :param name: dataset name
        :param filepath: dataset filepath
        :return: id
        """
        new_insert = Dataset(name=name, filepath=filepath, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_callback(self, name):
        """
        :param name: callback name
        :return: id
        """
        new_insert = Callback(name=name, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_metric(self, metric, metric_label):
        """
        :param metric: list of metric values
        :param metric_label: string containing metric label
        :return: id
        """
        new_insert = Metric(values=pd.Series(metric).to_json(), label=metric_label, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_sca_metric(self, guessing_entropy, success_rate, label, report_interval):
        """
        :param guessing_entropy: guessing entropy vs number of attack/validation traces
        :param success_rate: success rate vs number of attack/validation traces
        :param label: string containing the label for key rank
        :param report_interval: key rank report interval
        :return: id
        """
        new_insert = ScaMetric(guessing_entropy=pd.Series(guessing_entropy).to_json(),
                               success_rate=pd.Series(success_rate).to_json(),
                               report_interval=report_interval, label=label, analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_visualization(self, input_gradients, label, hyperparameters_id):
        """
        :param hyperparameters_id:
        :param input_gradients: 2D array: [[input_gradients_epoch1], [input_gradients_epoch2], ..., [input_gradients_epochN]]
        :param label: label for visualization entry
        :return: id
        """
        epochs = len(input_gradients)
        for epoch in range(epochs):
            new_insert = Visualization(values=pd.Series(input_gradients[epoch]).to_json(), epoch=epoch, label=label,
                                       hyperparameters_id=hyperparameters_id, analysis_id=self.analysis_id)
            self.insert(new_insert)

    def save_confusion_matrix(self, confusion_matrix, hyperparameters_id):
        """
        :param hyperparameters_id:
        :param confusion_matrix: 2D array containing confusion matrix
        :return: id's of inserted rows
        """
        row_ids = []
        for y_true, y_pred in enumerate(confusion_matrix):
            new_insert = ConfusionMatrix(y_pred=pd.Series(y_pred).to_json(), y_true=y_true, hyperparameters_id=hyperparameters_id,
                                         analysis_id=self.analysis_id)
            row_ids.append(self.insert(new_insert))
        return row_ids

    def save_random_state_hyperparameter(self, label, index, random_states):
        """
        :param label: label
        :param index: index
        :param random_states: random states
        :return: id
        """
        new_insert = RandomStates(random_states=pd.Series(random_states).to_json(), label=label, index=index,
                                  analysis_id=self.analysis_id)
        return self.insert(new_insert)

    def save_hyperparameter_metric(self, hyperparameter_id, metric_id):
        """
        :param hyperparameter_id: hyperparameter_id
        :param metric_id: metric_id
        """
        new_insert = HyperParameterMetric(hyperparameter_id=hyperparameter_id, metric_id=metric_id)
        self.session.add(new_insert)
        self.session.commit()

    def save_hyperparameter_sca_metric(self, hyperparameter_id, sca_metric_id):
        """
        :param hyperparameter_id: hyperparameter_id
        :param sca_metric_id: sca_metric_id
        """
        new_insert = HyperParameterScaMetric(hyperparameter_id=hyperparameter_id, sca_metric_id=sca_metric_id)
        self.session.add(new_insert)
        self.session.commit()

    def get_db_save(self):
        """
        :return: db_save object
        """
        return self.db_save

    def get_db_session(self):
        """
        :return: db session object
        """
        return self.session
