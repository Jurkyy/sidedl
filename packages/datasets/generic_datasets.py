class GenericDataset:

    def __init__(self):
        """ features """
        self.x_training = None
        self.x_validation = None
        self.x_test = None

        """ categorical labels """
        self.y_training = None
        self.y_validation = None
        self.y_test = None
