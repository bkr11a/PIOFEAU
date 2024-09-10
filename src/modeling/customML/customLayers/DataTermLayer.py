__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf
from src.modeling.customML.customLayers.ImageWarpingLayer import ImageWarpingLayer as warpImage

class DataTermLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = tf.Variable(1e-5, trainable = False)
        self.warpImage = warpImage()

    @tf.function
    def call(self, Inputs): 
        I1, I2, flow = Inputs

        # Approximate the image gradients
        # I(x, y) - I(x+1, y)
        # I(x, y) - I(x, y+1)
        I1_grad_x, I1_grad_y = tf.image.image_gradients(I1)
        I2_grad_x, I2_grad_y = tf.image.image_gradients(I2)

        # Compute Image Warping here!
        # I2_warped = tfa.image.dense_image_warp(I2, -flow)
        I2_warped = self.warpImage([I2, flow])
        u = flow[:, :, :, 0]
        v = flow[:, :, :, 1]

        # Warping is normally done from a backwards manner.
        # Forward warping can generate 'holes' since we are going out of the domain.
        dataTerm = I2_warped - I1

        u_next = u - self.alpha * dataTerm[:, :, :, 0] * I1_grad_x[:, :, :, 0]
        v_next = v - self.alpha * dataTerm[:, :, :, 0] * I1_grad_y[:, :, :, 0]

        flow_next = tf.stack([u_next, v_next], axis = -1)

        return flow_next