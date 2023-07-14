import tensorflow as tf
import tensorflow.keras as tk


class CrossEntropy(tk.metrics.Metric):
    def __init__(self, wrapper, name="loss", **kwargs):
        super(CrossEntropy, self).__init__(name=name, **kwargs)
        """ Creates a 'Mean' instance """
        self.mean = tf.keras.metrics.Mean()
        """ The 'wrapper' allows us to obtain the metric history of training/validation sets (workaround for keras-tuner). It can ignored
        if not used for keras-tuner algorithms """
        self.wrapper = wrapper

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mean.update_state(self.wrapper.tf_calculate_cross_entropy(y_true, y_pred))

    def result(self):
        return self.mean.result()

    def reset_states(self):
        self.mean.reset_states()

    def get_wrapper(self):
        return self.wrapper
