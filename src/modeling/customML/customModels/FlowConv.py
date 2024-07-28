__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf
# CNN Custom Class
# TODO - FIX THIS MESS!

# Should I pass the architecture through as a dictionary instead?
class FlowConvNet(tf.keras.Model):
    def __init__(self, twoFrameInput = True, flowInput = False, **kwargs):
        super().__init__(**kwargs)

        self.inputDim = (436, 1024, 1)
        self.flowInputDim = (436, 1024, 2)
        self.twoFrameInput = twoFrameInput
        self.flowInput = flowInput

        self.hidden = [tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same'),
                       tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01)),
                       tf.keras.layers.UpSampling2D(size = (2, 2)),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01))]

        self.out = tf.keras.layers.Conv2D(filters = 2, kernel_size = (3, 3), padding='same', activation = tf.keras.layers.LeakyReLU(alpha=0.01))                    

    @tf.function
    def call(self, X):
        # Should just hand craft it here rather than have the customisability for experimentation?
        
        # Have some form of aggregation network for the combination of two images here?
        if self.twoFrameInput:
            Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 0])
            Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 1])
        
            Z = tf.concat([Z1, Z2], axis = -1)
             
        if self.flowInput:
            I = X[0]
            warped = X[2]
            errors = X[3]
            
            # Flow field to refine
            Z0 = tf.keras.layers.InputLayer(input_shape = self.flowInputDim)(X[1])
            
            # Image pair to calculate optical flow
            Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(I[:, 0])
            Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(I[:, 1])
            
            # Warped image from previous optical flow estimation
            Z3 = tf.keras.layers.InputLayer(input_shape=self.inputDim)(warped)
            
            # # Calculated Error
            # Z4 = tf.keras.layers.subtract([Z2, Z3])
            Z4 = tf.keras.layers.InputLayer(input_shape=self.inputDim)(errors)
        
            # Z = tf.concat([Z0, Z1, Z2, Z3], axis = -1)
            Z = tf.concat([Z0, Z1, Z2, Z3, Z4], axis = -1)
        
        for layer in self.hidden:
            Z = layer(Z)
        
        # Output
        out = self.out(Z)
        
        return out