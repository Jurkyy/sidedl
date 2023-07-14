import tensorflow as tf
import tensorflow.keras as tk
import tensorflow.keras.backend as backend


class GuessingEntropy(tk.metrics.Metric):
    def __init__(self, wrapper, name="guessing_entropy", **kwargs):
        super(GuessingEntropy, self).__init__(name=name, **kwargs)
        self.sum = self.add_weight(name='sum', shape=1, initializer='zeros')
        """ The 'wrapper' allows us to obtain the metric history of training/validation sets (workaround for keras-tuner). It can ignored
        if not used for keras-tuner algorithms """
        self.wrapper = wrapper

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.sum.assign_add(self.wrapper.tf_calculate_guessing_entropy(y_true, y_pred))

    def result(self):
        return tf.cast(self.sum, tf.float32)

    def reset_states(self):
        self.sum.assign(backend.zeros(1))

    def get_wrapper(self):
        return self.wrapper
