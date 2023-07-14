class Profiling:
    """
    Class to run profiling phase on the dataset and model
    model: keras model already compiled
    settings: analysis settings
    dataset: dataset structure containing traces, labels and metadata (plaintext, ciphertext, keys)
    search_index: for hyper-parameter search purposes
    train_best_model: for hyper-parameter search purposes
    """

    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset
        self.builtin_callbacks = None
        self.custom_callbacks = None

    def train_model(self, model, callbacks):

        """
        Train the model (profiling phase).
        1. Configure dataset
        2. Train model
        """

        """ reshape traces if needed """
        input_layer_shape = model.get_layer(index=0).input_shape

        """ Check if model is created with Sequential or Model class from Keras """
        if len(input_layer_shape) == 1:
            """ Model was created with Model class """
            input_layer_shape = input_layer_shape[0]

        """ Check if neural network is a cnn or mlp """
        if len(input_layer_shape) == 3:
            self.dataset.reshape_for_cnn()
        else:
            self.dataset.reshape_for_mlp()

        """ Create callbacks """
        # callback_controls = CallbackControls(self.dataset, self.settings)
        # callback_controls.create_callbacks()
        # callbacks = callback_controls.get_callbacks()
        # self.builtin_callbacks = callback_controls.get_builtin_callbacks()
        # self.custom_callbacks = callback_controls.get_custom_callbacks()
        model_callbacks = [] if callbacks is None else callbacks

        if self.settings["split_test_set"] or self.settings["split_training_set"]:
            validation_set = (self.dataset.x_validation, self.dataset.y_validation)
        else:
            validation_set = (self.dataset.x_attack, self.dataset.y_attack)

        history = model.fit(
            x=self.dataset.x_profiling,
            y=self.dataset.y_profiling,
            batch_size=self.settings["batch_size"],
            verbose=2,
            epochs=self.settings["epochs"],
            shuffle=True,
            validation_data=validation_set,
            callbacks=model_callbacks)

        # TODO add callback metrics into history dictionary
        return history

    def get_builtin_callbacks(self):
        return self.builtin_callbacks

    def get_custom_callbacks(self):
        return self.custom_callbacks
