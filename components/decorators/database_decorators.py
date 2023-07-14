from packages.database.sqlite.db_save import DBSave
from packages.database.sqlite.db_update import DBUpdate
from packages.deeplearning.deep_learning_libraries.keras.keras import *


def keras_neural_network_database(add_keras_neural_network):
    """ decorator to execute add_keras_neural_network and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        method_name, seed, neural_network = add_keras_neural_network(*args)

        """ Parse neural network """
        keras_obj = Keras(neural_network)
        neural_network_dict = keras_obj.parse_keras_json_to_dict()

        """ save to database """
        db_save_object = DBSave(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_save_object.analysis_id = settings["analysis_id"]
        neural_network_id = db_save_object.save_neural_network(method_name, seed)
        for layer_index, nn_layer in enumerate(neural_network_dict["layers"]):
            layer_name = convert_keras_layer_name(nn_layer["name"])
            layer_type = convert_keras_layer_name(nn_layer["class_name"])
            layer_units = get_layer_units(nn_layer)
            layer_activation = nn_layer["activation"]
            layer_weight_initializer = nn_layer["kernel_initializer"]
            layer_bias_initializer = nn_layer["bias_initializer"]
            layer_type_id = db_save_object.save_layer_type(layer_type)
            layer_activation_id = db_save_object.save_activation(layer_activation)
            layer_weight_initializer_id = db_save_object.save_weight_initializer(layer_weight_initializer)
            layer_bias_initializer_id = db_save_object.save_bias_initializer(layer_bias_initializer)
            db_save_object.save_layer(layer_index, layer_name, layer_units, layer_type_id, layer_activation_id,
                                      layer_weight_initializer_id, layer_bias_initializer_id, neural_network_id)

    return execute_and_save


def tensorflow_neural_network_database(add_tensorflow_neural_network):
    """ decorator to execute add_tensorflow_neural_network and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        _, _, method_name, seed, neural_network = add_tensorflow_neural_network(*args)

        # TODO

    return execute_and_save


def torch_neural_network_database(add_torch_neural_network):
    """ decorator to execute add_torch_neural_network and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        _, _, method_name, seed, neural_network = add_torch_neural_network(*args)

        # TODO

    return execute_and_save


def update_best_model_database(check_best_model):
    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        check_best_model(*args)

        """ get variables from dictionary arguments """
        db_label_old = kwargs["db_label_old"]
        db_label_new = kwargs["db_label_new"]

        """ save to database """
        db_update_object = DBUpdate(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_update_object.analysis_id = settings["analysis_id"]
        db_update_object.db_update_sca_metric_label(db_label_old, db_label_new)

    return execute_and_save


def metrics_database(metrics_function):
    """ decorator to execute sca_metrics and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        history = metrics_function(*args)

        """ save to database """
        db_save_object = DBSave(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_save_object.analysis_id = settings["analysis_id"]
        for metric_name, metric_values in history.history.items():
            db_save_object.save_metric(metric_values, metric_name)

    return execute_and_save


def sca_metrics_database(sca_metrics_function):
    """ decorator to execute sca_metrics and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        guessing_entropy, success_rate = sca_metrics_function(*args)

        """ get variables from dictionary arguments """
        db_label = kwargs["db_label"]

        """ save to database """
        db_save_object = DBSave(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_save_object.analysis_id = settings["analysis_id"]
        db_save_object.save_sca_metric(guessing_entropy, success_rate, db_label, settings["sca_metrics_report_interval"])

    return execute_and_save


def dataset_database(add_dataset):
    """ decorator to execute sca_metrics and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        add_dataset(*args)

        """ save to database """
        db_save_object = DBSave(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_save_object.analysis_id = settings["analysis_id"]
        db_save_object.save_dataset(*args)

    return execute_and_save


def leakage_model_database(add_leakage_model):
    """ decorator to execute sca_metrics and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        add_leakage_model(*kwargs)

        """ save to database """
        db_save_object = DBSave(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_save_object.analysis_id = settings["analysis_id"]
        db_save_object.save_leakage_model(*kwargs)

    return execute_and_save


def callback_database(add_callback):
    """ decorator to execute sca_metrics and save to database """

    def execute_and_save(*args, **kwargs):
        """
        Execute sca_metrics and save results to database
        - args[0] is the ScaMetric component object itself
        """
        settings = args[0].settings

        """ execute SCAMetrics(...).run(...) function """
        callback_function_name = add_callback(*args)

        """ save to database """
        db_save_object = DBSave(f'{settings["database_root_path"]}{settings["database_filename"]}')
        db_save_object.analysis_id = settings["analysis_id"]
        db_save_object.save_callback(callback_function_name)

    return execute_and_save
