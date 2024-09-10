__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class RegularisationTermLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @tf.function
    def call(self, Inputs):
        flow = Inputs

        u = flow[:, :, :, 0]
        v = flow[:, :, :, 1]

        u_grad_x, u_grad_y = tf.image.image_gradients(tf.expand_dims(u, axis = -1))
        v_grad_x, v_grad_y = tf.image.image_gradients(tf.expand_dims(v, axis = -1))

        u_next = u - self.alpha * (u_grad_x[:, :, :, 0] + u_grad_y[:, :, :, 0])
        v_next = v - self.alpha * (v_grad_x[:, :, :, 0] + v_grad_y[:, :, :, 0])

        flow_next = tf.stack([u_next, v_next], axis = -1)

        return flow_next