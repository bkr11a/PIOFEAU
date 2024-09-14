__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

class AEPE_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @tf.function
    def call(self, y_true, y_pred):
        u_true = y_true[:, :, :, 0]
        u_pred = y_pred[:, :, :, 0]

        v_true = y_true[:, :, :, 1]
        v_pred = y_pred[:, :, :, 1]

        epe = tf.reduce_mean(tf.sqrt(tf.square(u_true - u_pred) + tf.square(v_true - v_pred)))

        return epe