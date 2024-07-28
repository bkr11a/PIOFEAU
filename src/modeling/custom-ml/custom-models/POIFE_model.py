__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

from assets.ml.src.CustomLayers import BiasLayer
from assets.ml.src.ResidualFlowNET import ResidualFlowEncoder

class POIFE_BranchNET(tf.keras.Model):
    def __init__(self, numBlocks = 1, twoFrameInput = True, flowInput = False, **kwargs):
        super().__init__(**kwargs)

        self.outputDim = (436, 1024, 2)
        self.inputDim = (436, 1024, 1)
        self.flowInputDim = (436, 1024, 2)
        
        self.twoFrameInput = twoFrameInput
        self.flowInput = flowInput
        
        self.numBlocks = numBlocks
        
        self.preprocessConv = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding='same')
        self.preprocessMaxPool = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same')
                
        self.ResBlockA = [tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1),
                         tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1)]
        
        self.ResBlockB = [tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1),
                         tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1)]
        
        self.ResBlockC = [tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1),
                         tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1)]
        
        self.ResBlockD = [tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1),
                         tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1)]
        
        self.ResBlockE = [tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1),
                         tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.1)]

        self.maxPoolA = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolB = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolC = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolD = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolE = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')

        self.convs = [tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.LeakyReLU(alpha=0.1),
                      tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.LeakyReLU(alpha=0.1),
                      tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.LeakyReLU(alpha=0.1),
                      tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'),
                      tf.keras.layers.BatchNormalization(),
                      tf.keras.layers.LeakyReLU(alpha=0.1)]

        self.dense = [tf.keras.layers.Flatten(),
                      tf.keras.layers.Dense(1536, activation = 'tanh'),
                      tf.keras.layers.Dense(768, activation = 'tanh'),
                      tf.keras.layers.Dense(384, activation = 'tanh')]
        
        self.out = tf.keras.layers.Dense(250, activation = 'linear')

    @tf.function
    def call(self, X):
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
        
        # Actually run through the network here!
        
        Z = Z                                                               # Input Shape (n, 436, 1024, 1) -> 2 678 784
        
        # Preprocess layer here!
        Z = self.preprocessConv(Z)                                          # Input Shape (n, 436, 1024, 16) -> 7 143 424 
        Z = self.preprocessMaxPool(Z)                                       # Input Shape (n, 218, 512, 16) -> 1 785 856 
        
        # Encoder - ResNET - Convolution -> Batch Normalisation -> Activation -> Convolution -> Batch Normalisation -> Addition w/identity -> Activation
        
        # ResBlock A
        Y = Z 
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockA):
                if j - 1 == len(self.ResBlockA):
                    Z += self.conv1A(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 218, 512, 32) -> 3 571 712
                    
        Z = self.maxPoolA(Z)                                                # Input Shape (n, 109, 256, 32) -> 892 928
        
        # ResBlock B
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockB):
                if j - 1 == len(self.ResBlockB):
                    Z += self.conv1B(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 109, 256, 64) -> 1 785 856
                    
        Z = self.maxPoolB(Z)                                                # Input Shape (n, 54, 128, 64) -> 442 368
        
        # ResBlock C
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockC):
                if j - 1 == len(self.ResBlockC):
                    Z += self.conv1C(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 54, 128, 128) -> 884 736
                    
        Z = self.maxPoolC(Z)                                                # Input Shape (n, 27, 64, 128) -> 221 184
        
        # ResBlock D
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockD):
                if j - 1 == len(self.ResBlockD):
                    Z += self.conv1D(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 27, 64, 256) -> 442 368
                    
        Z = self.maxPoolD(Z)                                                # Input Shape (n, 13, 32, 256) -> 106 496
        
        # ResBlock E
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockE):
                if j - 1 == len(self.ResBlockE):
                    Z += self.conv1E(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 13, 32, 512) -> 212 992
                    
        Z = self.maxPoolE(Z)                                                # Input Shape (n, 6, 16, 512) -> 49 152

        # Will need to get the two networks into the same shape
        for layer in self.convs:
            Z = layer(Z)

        for layer in self.dense:
            Z = layer(Z)

        out = self.out(Z)

        return out

class POIFE_TrunkNET(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inp = tf.keras.layers.InputLayer(input_shape = (2, ))
        self.hidden = [tf.keras.layers.Dense(400, activation = 'tanh', name = f"POIFE_TrunkLayer_{i}") for i in range(10)]
        self.out = tf.keras.layers.Dense(250, activation = 'linear', name = "POIFE_TrunkOuput")

    @tf.function
    def call(self, X):
        Z = self.inp(X)
        for layer in self.hidden:
            Z = layer(Z)

        out = self.out(Z)

        return out

class OpticalFlowOperatorNetwork(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.branchNN = POIFE_BranchNET() 
        self.trunkNN = POIFE_TrunkNET() 
        self.biasLayer_u = BiasLayer()
        self.biasLayer_v = BiasLayer()

    @tf.function
    def call(self, X):
        branch = self.branchNN(X[0])
        trunk = self.trunkNN(X[1])
        m = tf.multiply(branch, trunk)
        u, v = tf.split(m, [125, 125], num = 2, axis = 1)
        u_dot = tf.reduce_sum(u, axis = 1, keepdims = True)
        v_dot = tf.reduce_sum(v, axis = 1, keepdims = True)
        u_dot_b = self.biasLayer_u(u_dot)
        v_dot_b = self.biasLayer_v(v_dot)

        out = tf.concat([u_dot_b, v_dot_b], axis = 1)

        return out

class POIFE(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.inputDim = (436, 1024, 1)
        self.flowInputDim = (436, 1024, 2)

        self.encoder = ResidualFlowEncoder(numBlocks = 1)
        
        self.hidden = [tf.keras.layers.Conv2D(filters = 16, kernel_size = (7, 7), padding='same', strides = (2, 2)),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(alpha=0.1),
                       tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same'),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), padding='same', strides = (1, 1)),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(alpha=0.1),
                       tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'same'),
                       tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', strides = (1, 1)),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(alpha=0.1),
                       tf.keras.layers.UpSampling2D(size = (2, 2)),
                       tf.keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), padding='same', strides = (1, 1)),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(alpha=0.1),
                       tf.keras.layers.UpSampling2D(size = (2, 2)),
                       tf.keras.layers.Conv2D(filters = 16, kernel_size = (5, 5), padding='same', strides = (1, 1)),
                       tf.keras.layers.BatchNormalization(),
                       tf.keras.layers.LeakyReLU(alpha=0.1),
                       tf.keras.layers.UpSampling2D(size = (2, 2)),
                       tf.keras.layers.Conv2D(filters = 1, kernel_size = (5, 5), padding='same', strides = (1, 1)),
                       ]

        self.out = tf.keras.layers.Conv2D(filters = 2, kernel_size = (3, 3), padding='same', activation = 'linear')                    
        # Whatever we need here!

    @tf.function
    def call(self, X):
        encodedFlow = self.encoder(X)
        print(f"Encoded Flow Shape: {encodedFlow.shape}")
        # Output shape (n, 6, 16, 512)
        # How do I combine this to create a set of outputs where we need (n, 436, 1024, 2)
        # Pull inspiration from my DeepONET
        # Make feature extractors as two channels and standardise into the same output shape.
        # Then curate those outputs to become combined and used appropriately
        # Then update the training of this network to make it 'physics informed'
        
        Z1 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 0])
        Z2 = tf.keras.layers.InputLayer(input_shape = self.inputDim)(X[:, 1])

        print(f"Z1 Shape: {Z1.shape}")
        print(f"Z2 Shape: {Z2.shape}")

        Z1_T = tf.transpose(Z1, perm=[0, 2, 1, 3])
        Z2_T = tf.transpose(Z2, perm=[0, 2, 1, 3])

        print(f"Z1_T Shape: {Z1_T.shape}")
        print(f"Z2_T Shape: {Z2_T.shape}")

        Z1 = tf.reshape(Z1, shape=(tf.shape(Z1)[0], tf.shape(Z1)[1], tf.shape(Z1)[2]))
        Z2 = tf.reshape(Z2, shape=(tf.shape(Z2)[0], tf.shape(Z2)[1], tf.shape(Z2)[2]))

        Z1_T = tf.reshape(Z1_T, shape=(tf.shape(Z1_T)[0], tf.shape(Z1_T)[1], tf.shape(Z1_T)[2]))
        Z2_T = tf.reshape(Z2_T, shape=(tf.shape(Z2_T)[0], tf.shape(Z2_T)[1], tf.shape(Z2_T)[2]))

        print(f"Z1 Shape: {Z1.shape}")
        print(f"Z2 Shape: {Z2.shape}")

        print(f"Z1_T Shape: {Z1_T.shape}")
        print(f"Z2_T Shape: {Z2_T.shape}")

        Z1_Z1_T = tf.matmul(Z1_T, Z1)
        Z2_Z2_T = tf.matmul(Z2_T, Z2)

        Z = tf.matmul(Z1_Z1_T, Z2_Z2_T)

        print(f"Z1_Z1_T Z2_Z2_T Shape: {Z.shape}")

        Z = tf.reshape(Z, shape = (tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[2], 1))

        print(f"Z1_Z1_T Z2_Z2_T Shape: {Z.shape}")
    
        # Z = tf.concat([Z1, Z2], axis = -1)

        # print(Z.shape)

        for layer in self.hidden:
            Z = layer(Z)
            print(f"Hidden Z shape {Z.shape}")
        
        # Why not matrix multiply (None, 436, 1024) \cdot (None, 1024, 1024) -> (None, 436, 1024, 1)
        encodedFlow = tf.reshape(encodedFlow, shape=(tf.shape(encodedFlow)[0], tf.shape(encodedFlow)[1], tf.shape(encodedFlow)[2]))
        Z = tf.reshape(Z, shape=(tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[2]))
        
        Z = tf.matmul(encodedFlow, Z)
        Z = tf.reshape(Z, shape = (tf.shape(Z)[0], tf.shape(Z)[1], tf.shape(Z)[2], 1))

        print(f"Fused shape: {Z.shape}")

        # Z = tf.concat([Z, encodedFlow], axis = -1)
        # print(f"Concat Z shape {Z.shape}")

        
        # Output
        out = self.out(Z)
        print(f"Output shape: {out.shape}")
        
        return out

# Underneath let's introduce the physics here too.