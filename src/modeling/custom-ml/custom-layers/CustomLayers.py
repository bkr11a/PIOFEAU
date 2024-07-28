__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight(initializer = tf.keras.initializers.Zeros, trainable = True)

        @tf.function
        def call(self, X):
            return X + self.bias