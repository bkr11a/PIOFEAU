__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class IWE_Loss(tf.keras.losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def warpImage(self, flow):
        raise NotImplementedError("Currently have to await for an implementation of this image warping error.")
        pass
    
    @tf.function
    def call(self, y_true, y_pred):
        u_true = y_true[:, :, :, 0]
        u_pred = y_pred[:, :, :, 0]

        v_true = y_true[:, :, :, 1]
        v_pred = y_pred[:, :, :, 1]

        # epe = tf.reduce_mean(tf.sqrt(tf.square(u_true - u_pred) + tf.square(v_true - v_pred)))
        # Need to determine how to calculate costVolumeError from optical flow vectors
        predictedImageWarp = self.warpImage(flow=y_pred)
        trueImageWarp = self.warpImage(flow=y_true)
        imageWarpError = tf.reduce_mean(tf.abs(predictedImageWarp - trueImageWarp))

        return imageWarpError