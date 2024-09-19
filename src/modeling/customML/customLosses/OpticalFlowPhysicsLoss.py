__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class OFP_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @tf.function
    def call(self, y_true, y_pred):
        u_true = y_true[:, :, :, 0]
        u_pred = y_pred[:, :, :, 0]

        v_true = y_true[:, :, :, 1]
        v_pred = y_pred[:, :, :, 1]

        # Compute the loss according to the optical flow constraint equation.
        physicsLoss = tf.reduce_mean(tf.square(tf.tensordot(I_x, u_pred) + tf.tensordot(I_y, v_pred) + I_t))

        return physicsLoss