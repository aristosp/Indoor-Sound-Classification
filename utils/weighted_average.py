import tensorflow as tf


class WeightedAverageLayer(tf.keras.layers.Layer):
    """
    This custom layer calculates the weighted average of its inputs (instead of simple average).
    This method can be expanded to n weighted averaged input by changing the number of w_i in __init__
    and what the layer returns accordingly.
    :param w1: weight of first input
    :param w2: weight of second input
    """
    def __init__(self, w1, w2, **kwargs):
        super(WeightedAverageLayer, self).__init__(**kwargs)
        self.w1 = w1
        self.w2 = w2

    def call(self, inputs):
        return self.w1 * inputs[0] + self.w2 * inputs[1]