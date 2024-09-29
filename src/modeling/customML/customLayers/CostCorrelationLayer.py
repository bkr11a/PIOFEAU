__author__ = "Brad Rice"
__version__ = "0.1.0"

from typing import List

import tensorflow as tf

class CostVolumeCorrelationLayer(tf.keras.layers.Layer):
    """
    CostVolumeCorrelationLayer _summary_

    _extended_summary_

    Args:
        tf (_type_): _description_

    Examples:
        ``` py
        # Import the custom module
        from modeling.customML.customLayers import CostVolumeCorrelationLayer
        
        # Instantiate an instance of a CostVolumeCorrelationLayer
        costVolumeLayer = CostCorrelationLayer(maxDisplacement = 3)
        ```
    """
    def __init__(self, maxDisplacement = 5, **kwargs):
        """
        __init__ _summary_

        _extended_summary_

        Args:
            maxDisplacement (int, optional): _description_. Defaults to 5.
        """
        super().__init__(**kwargs)
        self.maxDisplacement = maxDisplacement

    @tf.function
    def call(self, Inputs : List[tf.Tensor]) -> tf.Tensor:
        """
        call _summary_

        _extended_summary_

        Args:
            Inputs (List[tf.Tensor]): _description_

        Returns:
            tf.Tensor: _description_
        """
        F1, F2 = Inputs
        batch_size, height, width, channels = F1.shape
        padded_level = tf.pad(F2, [[0, 0], [self.maxDisplacement, self.maxDisplacement], [self.maxDisplacement, self.maxDisplacement], [0,0]])
        b, h, w, c = tf.unstack(tf.shape(F1))
        maxOffset = self.maxDisplacement * 2 + 1

        costVolume = []
        for y in range(0, maxOffset):
            for x in range(0, maxOffset):
                slc = tf.slice(padded_level, [0, y, x, 0], [-1, h, w, -1])
                cost = tf.reduce_mean(F1 * slc, axis = 3, keepdims = True)
                costVolume.append(cost)
        
        costVolume = tf.concat(costVolume, axis = 3)
        costVolume = tf.nn.leaky_relu(costVolume, alpha = 0.1)

        return costVolume