__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

from src.modeling.customML.customLayers.DataTermLayer import DataTermLayer as dtl
from src.modeling.customML.customLayers.RegularisationTermLayer import RegularisationTermLayer as rtl

class UnrolledOF_Model(tf.keras.Model):
    def __init__(self, num_iterations,  **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        self.alphas = [tf.Variable(tf.random.normal(mean=0.5, stddev=1), trainable=True, constraint=self.contraint_fn, dtype=tf.float32) for i in range(self.num_iterations)]
        self.data_term_layers = [dtl(alpha=self.alphas[i]) for i in range(self.num_iterations)]
        self.regularisation_term_layers = [rtl(alpha=self.alphas[i]) for i in range(self.num_iterations)]

    def contraint_fn(self, x):
        return tf.maximum(x) + 1e-6

    @tf.function
    def call(self, X):
        I1, I2 = X
        u = tf.zeros_like(I1)
        v = tf.zeros_like(I1)

        for i in range(self.num_iterations):
            u, v = self.data_term_layers[i]([I1, I2, u, v])
            u, v = self.regularisation_term_layers[i]([u, v])

        return u, v