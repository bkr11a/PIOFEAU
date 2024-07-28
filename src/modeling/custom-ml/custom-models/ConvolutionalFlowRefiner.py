__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

from assets.ml.src.FlowConv import FlowConvNet

class FlowRefinerConvNet(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.flowNET = FlowConvNet()
        self.inputDim = (436, 1024, 1)
        self.flowInputDim = (436, 1024, 2)

        self.hidden = [tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same'),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.UpSampling2D(size = (2, 2)),
                       tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01))]

        self.out = tf.keras.layers.Conv2D(filters = 2, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01))                    
        # Whatever we need here!

    @tf.function
    def call(self, X):
        estFlow = self.flowNET(X)
        
        Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 0])
        Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 1])
    
        Z = tf.concat([Z1, Z2, estFlow], axis = -1)

        for layer in self.hidden:
            Z = layer(Z)
        
        # Output
        out = self.out(Z)
        
        return out