from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as backend
import tensorflow as tf
import numpy as np


class CallbackVisualization(Callback):
    def __init__(self, settings, dataset):
        super().__init__()
        self.current_epoch = 0
        self.x = dataset.x_profiling[:settings["visualization"]]
        self.y = dataset.y_profiling[:settings["visualization"]]
        self.number_of_epochs = settings["epochs"]
        self.number_of_samples = settings["number_of_samples"]
        self.gradients = np.zeros((settings["epochs"], settings["number_of_samples"]))
        self.gradients_sum = np.zeros(settings["number_of_samples"])

    def on_epoch_end(self, epoch, logs=None):
        input_trace = tf.Variable(self.x)

        with tf.GradientTape() as tape:
            tape.watch(input_trace)
            pred = self.model(input_trace)
            loss = tf.keras.losses.categorical_crossentropy(self.y, pred)

        grad = tape.gradient(loss, input_trace)

        input_gradients = np.zeros(self.number_of_samples)
        for i in range(len(self.x)):
            input_gradients += grad[i].numpy().reshape(self.number_of_samples)

        self.gradients[epoch] = input_gradients
        if np.max(self.gradients[epoch]) != 0:
            self.gradients_sum += np.abs(self.gradients[epoch] / np.max(self.gradients[epoch]))
        else:
            self.gradients_sum += np.abs(self.gradients[epoch])

        backend.clear_session()

    def input_gradients_epochs(self):
        for e in range(self.number_of_epochs):
            if np.max(self.gradients[e]) != 0:
                self.gradients[e] = np.abs(self.gradients[e] / np.max(self.gradients[e]))
            else:
                self.gradients[e] = np.abs(self.gradients[e])
        return self.gradients
