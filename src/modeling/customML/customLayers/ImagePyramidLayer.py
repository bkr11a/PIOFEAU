__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class ImagePyramidLayer(tf.keras.layers.Layer):
    def __init__(self, numberOfPyramids, scale = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.numPyramids = numberOfPyramids
        self.scale = scale

    def get_dim(self, tensor, index):
        if tensor.shape.ndims is None:
            return tf.shape(tensor)[index]
        return tensor.shape[index] or tf.shape(tensor)[index]

    @tf.function
    def call(self, image):
        pyramids = [image]
        batch, height, width, channels = (self.get_dim(image, 0),
                                        self.get_dim(image, 1),
                                        self.get_dim(image, 2),
                                        self.get_dim(image, 3))
        img = image
        for level in range(1, self.numPyramids):
            batch, height, width, channels = (self.get_dim(img, 0),
                                            self.get_dim(img, 1),
                                            self.get_dim(img, 2),
                                            self.get_dim(img, 3))
            img = tf.image.resize(img, size=(int(self.scale*height), int(self.scale*width)), method='bilinear')
            pyramids.append(img)
        
        return pyramids