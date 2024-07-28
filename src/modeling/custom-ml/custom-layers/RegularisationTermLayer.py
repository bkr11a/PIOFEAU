__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class RegularisationTermLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @tf.function
    def call(self, Inputs):
        u, v = Inputs

        u_grad_x, u_grad_y = tf.image.image_gradients(u)
        v_grad_x, v_grad_y = tf.image.image_gradients(v)

        u_next = u - self.alpha * (u_grad_x + u_grad_y)
        v_next = v - self.alpha * (v_grad_x + v_grad_y)

        return u_next, v_next