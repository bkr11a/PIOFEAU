__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

class AE_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @tf.function
    def __call__(self, y_true, y_pred):
        angularError = self.call(y_true=y_true, y_pred=y_pred)

        return angularError

    @tf.function
    def call(self, y_true, y_pred):
        u_true = y_true[:, :, :, 0]
        u_pred = y_pred[:, :, :, 0]

        v_true = y_true[:, :, :, 1]
        v_pred = y_pred[:, :, :, 1]
        
        angularError = tf.math.acos((u_pred * u_true + v_pred * v_true + 1.0)) / tf.sqrt( (tf.square(u_pred) + tf.square(v_pred) + 1.0)*(tf.square(u_true) + tf.square(v_true) + 1.0) )
        angularError = tf.reduce_mean(angularError)

        return angularError
