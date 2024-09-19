__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class ACV_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def costVolume(self, flow):
        raise NotImplementedError("Currently have to await for an implementation of this cost volume error.")
        pass
    
    @tf.function
    def call(self, y_true, y_pred):
        u_true = y_true[:, :, :, 0]
        u_pred = y_pred[:, :, :, 0]

        v_true = y_true[:, :, :, 1]
        v_pred = y_pred[:, :, :, 1]

        # epe = tf.reduce_mean(tf.sqrt(tf.square(u_true - u_pred) + tf.square(v_true - v_pred)))
        # Need to determine how to calculate costVolumeError from optical flow vectors
        predictedCostVolume = self.costVolume(flow=y_pred)
        trueCostVolume = self.costVolume(flow=y_true)
        costVolumeError = tf.reduce_mean(tf.abs(predictedCostVolume - trueCostVolume))

        return costVolumeError