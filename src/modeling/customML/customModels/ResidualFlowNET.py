__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

class ResidualFlowNET(tf.keras.Model):
    def __init__(self, numBlocks = 2, twoFrameInput = True, flowInput = False, **kwargs):
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
                         tf.keras.layers.LeakyReLU(alpha=0.01),
                         tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01)]
        
        self.ResBlockB = [tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01),
                         tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01)]
        
        self.ResBlockC = [tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01),
                         tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01)]
        
        self.ResBlockD = [tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01),
                         tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01)]
        
        self.ResBlockE = [tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01),
                         tf.keras.layers.Conv2D(filters = 512, kernel_size = (3, 3), padding='same'),
                         tf.keras.layers.BatchNormalization(),
                         tf.keras.layers.LeakyReLU(alpha=0.01)]
        
        self.conv1A = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same')
        self.conv1B = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same')
        self.conv1C = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same')
        self.conv1D = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same')
        self.conv1E = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same')
        
        self.maxPoolA = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolB = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolC = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolD = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolE = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        
        self.upSampleA = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleB = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleC = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleD = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleE = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleF = tf.keras.layers.UpSampling2D(size = (2, 2))
        
        self.conv256 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding='same')
        self.conv128 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same')
        self.conv64 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same')
        self.conv32 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same')
        self.conv16 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding='same')
        self.conv8 = tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), padding='same')
        
        self.bnA = tf.keras.layers.BatchNormalization()
        self.bnB = tf.keras.layers.BatchNormalization()
        self.bnC = tf.keras.layers.BatchNormalization()
        self.bnD = tf.keras.layers.BatchNormalization()
        self.bnE = tf.keras.layers.BatchNormalization()
        self.bnF = tf.keras.layers.BatchNormalization()
        
        self.out = tf.keras.layers.Conv2D(filters = 2, kernel_size = (3, 3), padding='same', activation = 'linear')
    
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
        
        # Actually run through the network here!
        
        Z = Z                                                               # Input Shape (n, 436, 1024, 1) -> 2 678 784
        print(f"Input Shape: {Z.shape}")
        
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
                    
        print(f"ResBlockA Shape: {Z.shape}")
                    
        Z = self.maxPoolA(Z)                                                # Input Shape (n, 109, 256, 32) -> 892 928
        print(f"MaxPoolA Shape: {Z.shape}")
        
        # ResBlock B
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockB):
                if j - 1 == len(self.ResBlockB):
                    Z += self.conv1B(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 109, 256, 64) -> 1 785 856
                    
        print(f"ResBlockB Shape: {Z.shape}")
        Z = self.maxPoolB(Z)                                                # Input Shape (n, 54, 128, 64) -> 442 368
        print(f"MaxPoolB Shape: {Z.shape}")
        
        # ResBlock C
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockC):
                if j - 1 == len(self.ResBlockC):
                    Z += self.conv1C(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 54, 128, 128) -> 884 736
                    
        print(f"ResBlockC Shape: {Z.shape}")
        Z = self.maxPoolC(Z)                                                # Input Shape (n, 27, 64, 128) -> 221 184
        print(f"MaxPoolC Shape: {Z.shape}")
        
        # ResBlock D
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockD):
                if j - 1 == len(self.ResBlockD):
                    Z += self.conv1D(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 27, 64, 256) -> 442 368
                    
        print(f"ResBlockD Shape: {Z.shape}")
                    
        Z = self.maxPoolD(Z)                                                # Input Shape (n, 13, 32, 256) -> 106 496
        print(f"MaxPoolD Shape: {Z.shape}")
        
        # ResBlock E
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockE):
                if j - 1 == len(self.ResBlockE):
                    Z += self.conv1E(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 13, 32, 512) -> 212 992
                    
        print(f"ResBlockE Shape: {Z.shape}")
                    
        Z = self.maxPoolE(Z)                                                # Input Shape (n, 6, 16, 512) -> 49 152
        print(f"MaxPoolE Shape: {Z.shape}")
        
        # Decoder!
        
        # Should this be done as a standard upscaled convolutional network or analoguous to the resnet architecture?
        # First pass is a standard convolutional decoder mapped to the optical flow outputs
        
        Z = self.conv256(Z)  # (n, 6, 16, 256)
        Z = self.bnA(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderA Shape: {Z.shape}")
        Z = self.upSampleA(Z)                                  # (n, 12, 32, 256)
        print(f"UpSampleA Shape: {Z.shape}")

        # Pad to correct the size - (need (n, 13, 32, 256))
        paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        Z = tf.pad(Z, paddings)                                                             # (n, 13, 32, 256)
        
        print(f"After Padding Shape: {Z.shape}")
        
        Z = self.conv128(Z)  # (n, 13, 32, 128)
        Z = self.bnB(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderB Shape: {Z.shape}")
        Z = self.upSampleB(Z)                                  # (n, 26, 64, 128)
        print(f"UpSampleB Shape: {Z.shape}")

        # Pad to correct the size - (need (n, 27, 64, 128))
        paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        Z = tf.pad(Z, paddings)                                                             # (n, 27, 64, 128)

        print(f"After Padding Shape: {Z.shape}")
        
        Z = self.conv64(Z)   # (n, 27, 64, 128)
        Z = self.bnC(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderC Shape: {Z.shape}")
        Z = self.upSampleC(Z)                                  # (n, 54, 128, 64)
        print(f"UpSampleC Shape: {Z.shape}")
        
        Z = self.conv32(Z)   # (n, 54, 128, 32)
        Z = self.bnD(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderD Shape: {Z.shape}")
        Z = self.upSampleD(Z)                                  # (n, 108, 256, 32)
        print(f"UpSampleD Shape: {Z.shape}")
        
        # Pad to correct the size - (need (n, 109, 256, 32))
        paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        Z = tf.pad(Z, paddings)                                                             # (n, 109, 256, 16)

        print(f"After Padding Shape: {Z.shape}")
        
        Z = self.conv16(Z)   # (n, 109, 256, 16)
        Z = self.bnE(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderE Shape: {Z.shape}")
        Z = self.upSampleE(Z)                                  # (n, 218, 512, 16)
        print(f"UpSampleE Shape: {Z.shape}")
        
        Z = self.conv8(Z)   # (n, 218, 512, 8)
        Z = self.bnF(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderF Shape: {Z.shape}")
        Z = self.upSampleF(Z)                                  # (n, 436, 1024, 8)
        print(f"UpSampleF Shape: {Z.shape}")
        
        # Output
        out = self.out(Z)                                                   # Output Shape (n, 436, 1024, 2)
        print(f"Output Shape Shape: {out.shape}")
        
        return out

class ResidualFlowEncoder(tf.keras.Model):
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
        
        self.conv1A = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same', activation='relu')
        self.conv1B = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same', activation='relu')
        self.conv1C = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same', activation='relu')
        self.conv1D = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same', activation='relu')
        self.conv1E = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding = 'same', activation='relu')
        
        self.maxPoolA = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolB = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolC = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolD = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        self.maxPoolE = tf.keras.layers.MaxPool2D(pool_size = (2, 2), strides = None, padding = 'valid')
        
        self.upSampleA = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleB = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleC = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleD = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleE = tf.keras.layers.UpSampling2D(size = (2, 2))
        self.upSampleF = tf.keras.layers.UpSampling2D(size = (2, 2))
        
        self.conv256 = tf.keras.layers.Conv2D(filters = 256, kernel_size = (3, 3), padding='same', activation='relu')
        self.conv128 = tf.keras.layers.Conv2D(filters = 128, kernel_size = (3, 3), padding='same', activation='relu')
        self.conv64 = tf.keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), padding='same', activation='relu')
        self.conv32 = tf.keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), padding='same', activation='relu')
        self.conv16 = tf.keras.layers.Conv2D(filters = 16, kernel_size = (3, 3), padding='same', activation='relu')
        self.conv8 = tf.keras.layers.Conv2D(filters = 8, kernel_size = (3, 3), padding='same', activation='relu')
        
        self.bnA = tf.keras.layers.BatchNormalization()
        self.bnB = tf.keras.layers.BatchNormalization()
        self.bnC = tf.keras.layers.BatchNormalization()
        self.bnD = tf.keras.layers.BatchNormalization()
        self.bnE = tf.keras.layers.BatchNormalization()
        self.bnF = tf.keras.layers.BatchNormalization()
        
        self.out = tf.keras.layers.Conv2D(filters = 1, kernel_size = (3, 3), padding='same', activation = 'linear')
    
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
        
        # Actually run through the network here!
        
        Z = Z                                                               # Input Shape (n, 436, 1024, 1) -> 2 678 784
        print(f"Input Shape: {Z.shape}")
        
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
                    
        print(f"ResBlockA Shape: {Z.shape}")
                    
        Z = self.maxPoolA(Z)                                                # Input Shape (n, 109, 256, 32) -> 892 928
        print(f"MaxPoolA Shape: {Z.shape}")
        
        # ResBlock B
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockB):
                if j - 1 == len(self.ResBlockB):
                    Z += self.conv1B(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 109, 256, 64) -> 1 785 856
                    
        print(f"ResBlockB Shape: {Z.shape}")
        Z = self.maxPoolB(Z)                                                # Input Shape (n, 54, 128, 64) -> 442 368
        print(f"MaxPoolB Shape: {Z.shape}")
        
        # ResBlock C
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockC):
                if j - 1 == len(self.ResBlockC):
                    Z += self.conv1C(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 54, 128, 128) -> 884 736
                    
        print(f"ResBlockC Shape: {Z.shape}")
        Z = self.maxPoolC(Z)                                                # Input Shape (n, 27, 64, 128) -> 221 184
        print(f"MaxPoolC Shape: {Z.shape}")
        
        # ResBlock D
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockD):
                if j - 1 == len(self.ResBlockD):
                    Z += self.conv1D(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 27, 64, 256) -> 442 368
                    
        print(f"ResBlockD Shape: {Z.shape}")
                    
        Z = self.maxPoolD(Z)                                                # Input Shape (n, 13, 32, 256) -> 106 496
        print(f"MaxPoolD Shape: {Z.shape}")
        
        # ResBlock E
        Y = Z
        for i in range(self.numBlocks):
            for j, layer in enumerate(self.ResBlockE):
                if j - 1 == len(self.ResBlockE):
                    Z += self.conv1E(Y)
                    Z = layer(Z)
                else:
                    Z = layer(Z)                                            # Input Shape (n, 13, 32, 512) -> 212 992
                    
        print(f"ResBlockE Shape: {Z.shape}")
                    
        Z = self.maxPoolE(Z)                                                # Input Shape (n, 6, 16, 512) -> 49 152
        print(f"MaxPoolE Shape: {Z.shape}")
        
        # Decoder!
        
        # Should this be done as a standard upscaled convolutional network or analoguous to the resnet architecture?
        # First pass is a standard convolutional decoder mapped to the optical flow outputs
        
        Z = self.conv256(Z)  # (n, 6, 16, 256)
        Z = self.bnA(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderA Shape: {Z.shape}")
        Z = self.upSampleA(Z)                                  # (n, 12, 32, 256)
        print(f"UpSampleA Shape: {Z.shape}")

        # Pad to correct the size - (need (n, 13, 32, 256))
        paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        Z = tf.pad(Z, paddings)                                                             # (n, 13, 32, 256)
        
        print(f"After Padding Shape: {Z.shape}")
        
        Z = self.conv128(Z)  # (n, 13, 32, 128)
        Z = self.bnB(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderB Shape: {Z.shape}")
        Z = self.upSampleB(Z)                                  # (n, 26, 64, 128)
        print(f"UpSampleB Shape: {Z.shape}")

        # Pad to correct the size - (need (n, 27, 64, 128))
        paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        Z = tf.pad(Z, paddings)                                                             # (n, 27, 64, 128)

        print(f"After Padding Shape: {Z.shape}")
        
        Z = self.conv64(Z)   # (n, 27, 64, 128)
        Z = self.bnC(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderC Shape: {Z.shape}")
        Z = self.upSampleC(Z)                                  # (n, 54, 128, 64)
        print(f"UpSampleC Shape: {Z.shape}")
        
        Z = self.conv32(Z)   # (n, 54, 128, 32)
        Z = self.bnD(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderD Shape: {Z.shape}")
        Z = self.upSampleD(Z)                                  # (n, 108, 256, 32)
        print(f"UpSampleD Shape: {Z.shape}")
        
        # Pad to correct the size - (need (n, 109, 256, 32))
        paddings = ((0, 0), (0, 1), (0, 0), (0, 0))
        Z = tf.pad(Z, paddings)                                                             # (n, 109, 256, 16)

        print(f"After Padding Shape: {Z.shape}")
        
        Z = self.conv16(Z)   # (n, 109, 256, 16)
        Z = self.bnE(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderE Shape: {Z.shape}")
        Z = self.upSampleE(Z)                                  # (n, 218, 512, 16)
        print(f"UpSampleE Shape: {Z.shape}")
        
        Z = self.conv8(Z)   # (n, 218, 512, 8)
        Z = self.bnF(Z)
        Z = tf.keras.layers.LeakyReLU(alpha=0.01)(Z)
        print(f"DecoderF Shape: {Z.shape}")
        Z = self.upSampleF(Z)                                  # (n, 436, 1024, 8)
        print(f"UpSampleF Shape: {Z.shape}")

        Z = self.out(Z)
        
        return Z 