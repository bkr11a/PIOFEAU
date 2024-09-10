__author__ = "Brad Rice"
__version__ = 0.1

import tensorflow as tf

from assets.ml.src.POIFE_model import POIFE
from assets.ml.src.CustomLosses import EPE_Loss

poife = POIFE()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999)
poife.compile(optimizer=optimizer, loss=EPE_Loss(), metrics = ['mse'])

poife.build(input_shape=(None, 2, 436, 1024, 1))
poife.summary()