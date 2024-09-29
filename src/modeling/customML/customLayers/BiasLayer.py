__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

class BiasLayer(tf.keras.layers.Layer):
    """
    BiasLayer _summary_

    _extended_summary_

    Warning: Deprecated - Not Relevant to the Project
        This class serves no purpose to the repository. Use with caution.

    Args:
        tf (_type_): _description_
    """
    def __init__(self, **kwargs):
        """
        __init__ _summary_

        _extended_summary_
        """
        super().__init__(**kwargs)

    def build(self, input_shape: tuple[int]):
        """
        build _summary_

        _extended_summary_

        Args:
            input_shape (tuple[int]): _description_
        """
        self.bias = self.add_weight(initializer = tf.keras.initializers.Zeros, trainable = True)

    @tf.function
    def call(self, X: tf.Tensor) -> tf.Tensor:
        """
        call _summary_

        _extended_summary_

        Args:
            X (tf.Tensor): _description_

        Returns:
            tf.Tensor: _description_
        """
        return X + self.bias