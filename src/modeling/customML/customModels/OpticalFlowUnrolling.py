__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

from src.modeling.customML.customLayers.DataTermLayer import DataTermLayer as dtl
from src.modeling.customML.customLayers.ImageWarpingLayer import ImageWarpingLayer as warpImage
from src.modeling.customML.customLayers.ImagePyramidLayer import ImagePyramidLayer as pyramidLayer
from src.modeling.customML.customLayers.RegularisationTermLayer import RegularisationTermLayer as rtl
from src.modeling.customML.customLayers.CostCorrelationLayer import CostVolumeCorrelationLayer as correlationLayer

class UnrolledOFModel(tf.keras.Model):
    def __init__(self, num_iterations, numPyramidLevels, correlationMaxDisplacement, **kwargs):
        super().__init__(**kwargs)
        self.num_iterations = num_iterations
        self.correlationMaxDisplacement = correlationMaxDisplacement
        self.numPyramidLevels = numPyramidLevels
        # self.alphas = [tf.Variable(tf.random.normal(shape=[1], mean=0.5, stddev=1), trainable=True, constraint=self.constraint_fn, dtype=tf.float32) for i in range(self.num_iterations)]
        self.alphas = [tf.Variable(1e-5, trainable=False, constraint=self.constraint_fn, dtype=tf.float32) for i in range(self.num_iterations)]
        # self.data_term_layers = [dtl(alpha=self.alphas[i], name = f"DataTermLayer_{i+1}") for i in range(self.num_iterations)]
        # self.regularisation_term_layers = [rtl(alpha=self.alphas[i], name = f"RegularisationTermLayer_{i+1}") for i in range(self.num_iterations)]
        self.featureExtractionLayers = [
            # tf.keras.layers.Conv2D(2, kernel_size = 3, padding='same', activation = tf.keras.layers.LeakyReLU(negative_slope = 0.1)),
            tf.keras.layers.Conv2D(32, kernel_size = 3, padding='same', activation = tf.keras.layers.LeakyReLU(negative_slope = 0.1)),
            # tf.keras.layers.Conv2D(64, kernel_size = 3, padding='same', activation = tf.keras.layers.LeakyReLU(negative_slope = 0.1)),
        ]
        self.costVolumeLayer = correlationLayer(maxDisplacement = self.correlationMaxDisplacement)
        self.pyramid = pyramidLayer(numberOfPyramids=numPyramidLevels, scale=0.5, name = "ImagePyramidLayer")
        self.flowRefinementLayers = [
            tf.keras.layers.Conv2D(32, kernel_size = 3, padding='same', activation = tf.keras.layers.LeakyReLU(negative_slope = 0.1)),
            # tf.keras.layers.Conv2D(2, kernel_size = 3, padding='same', activation = tf.keras.layers.LeakyReLU(negative_slope = 0.1))
        ]
        self.upsample = tf.keras.layers.UpSampling2D(size = (2, 2), interpolation = 'bilinear')
        self.flowPrediction = tf.keras.layers.Conv2D(2, kernel_size = 3, padding = 'same', activation = tf.keras.layers.LeakyReLU(negative_slope = 0.1))


    def constraint_fn(self, x):
        return tf.minimum(tf.maximum(x, 0) + 1e-6, tf.minimum(x, 5e-2))

    @tf.function
    def call(self, X):
        I1 = X[:, 0, :, :, :]
        I2 = X[:, 1, :, :, :]
        batch, height, width, channels = I1.shape
        flowshape = (batch, height, width, 2)
        flow = tf.zeros(shape=flowshape, dtype=tf.float32)

        Z1 = I1
        Z2 = I2
        
        pyramid_1 = self.pyramid(Z1)
        pyramid_2 = self.pyramid(Z2)
        flow_pyramids = self.pyramid(flow)

        flows = []
        for i in range(self.numPyramidLevels):
            Z1 = pyramid_1[i]
            Z2 = pyramid_2[i]
            flow_pyramid = flow_pyramids[i]
            
            # for i in range(self.num_iterations):
                # flow_pyramid = self.data_term_layers[i]([Z1, Z2, flow_pyramid])
                # flow_pyramid = self.regularisation_term_layers[i](flow_pyramid)

            for layer in self.featureExtractionLayers:
                Z1 = layer(Z1)
                Z2 = layer(Z2)

            corr = self.costVolumeLayer([Z1, Z2])

            flow_pyramid = tf.concat([flow_pyramid, corr], axis=3)
            
            for layer in self.flowRefinementLayers:
                flow_pyramid = layer(flow_pyramid)

            
            for j in range(0, i, 1):
                flow_pyramid = self.upsample(flow_pyramid)

            # flow_pyramid = tf.image.resize(flow_pyramid, size=(height, width), method='bilinear')
            flows.append(flow_pyramid)

        flow = flows[0]
        for i in range(1, len(flows)):
            flow = tf.concat([flow, flows[i]], axis = 3)

        flow = self.flowPrediction(flow)

        return flow