__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

from src.modeling.customML.customLayers.DataTermLayer import DataTermLayer as dtl
from src.modeling.customML.customLayers.RegularisationTermLayer import RegularisationTermLayer as rtl

class UnrolledOFModel(tf.keras.Model):
    def __init__(self, num_iterations,  **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        self.alphas = [tf.Variable(tf.random.normal(shape=[1], mean=0.5, stddev=1), trainable=True, constraint=self.constraint_fn, dtype=tf.float32) for i in range(self.num_iterations)]
        self.data_term_layers = [dtl(alpha=self.alphas[i], name = f"DataTermLayer_{i+1}") for i in range(self.num_iterations)]
        self.regularisation_term_layers = [rtl(alpha=self.alphas[i], name = f"RegularisationTermLayer_{i+1}") for i in range(self.num_iterations)]

    def constraint_fn(self, x):
        return tf.maximum(x, 0) + 1e-6

    @tf.function
    def call(self, X):
        I1 = X[:, 0, :, :, :]
        I2 = X[:, 1, :, :, :]
        batch, height, width, channels = I1.shape
        flowshape = (batch, height, width, 2)
        flow = tf.zeros(shape=flowshape, dtype=tf.float32)

        for i in range(self.num_iterations):
            flow = self.data_term_layers[i]([I1, I2, flow])
            flow = self.regularisation_term_layers[i](flow)

        return flow