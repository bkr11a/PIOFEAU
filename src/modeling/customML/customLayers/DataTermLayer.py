__author__ = "Brad Rice"
__version__ = "0.1.0"

import tensorflow as tf

class DataTermLayer(tf.keras.layers.Layer):
    def __init__(self, alpha, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha

    @tf.function
    def call(self, Inputs): 
        def warpImage(image, flow):
            """
            Warps an image according to the given optical flow.

            Args:
            - img: A 4D tensor of shape (batch_size, height, width, channels).
            - flow: A 4D tensor of shape (batch_size, height, width, 2), containing the optical flow vectors.

            Returns:
            - A 4D tensor of the warped image.
            """
            displacement = flow * 0.5

            batch_size, height, width, channels = image.shape

            # Generate a fresh grid of points for remapping
            grid_x, grid_y = tf.meshgrid(tf.range(width, dtype=tf.float32), tf.range(height, dtype=tf.float32))
            grid = tf.stack([grid_x, grid_y], axis=-1)  # shape: (height, width, 2)
            grid = tf.expand_dims(grid, 0)  # shape: (1, height, width, 2)
            grid = tf.tile(grid, [batch_size, 1, 1, 1])  # shape: (batch_size, height, width, 2)

            # Add some extra spice by mixing in the displacement
            new_coords = grid + displacement

            # Normalize the new coordinates to the range [-1, 1]
            new_coords_x = 2.0 * new_coords[..., 0] / tf.cast(width - 1, tf.float32) - 1.0
            new_coords_y = 2.0 * new_coords[..., 1] / tf.cast(height - 1, tf.float32) - 1.0
            new_coords = tf.stack([new_coords_x, new_coords_y], axis=-1)

            # Use grid_sample to perform the warping
            warped_image = tf.nn.grid_sample(image, new_coords, method='bilinear', padding_mode='zeros')
            
            return warped_image
        
        # Grab the inputs from the layer
        I1, I2, u, v = Inputs

        # Approximate the image gradients
        I1_grad_x, I1_grad_y = tf.image.image_gradients(I1)
        I2_grad_x, I2_grad_y = tf.image.image_gradients(I2)

        # Compute Image Warping here!
        # warp image using appropriate method.
        I1_warped = warpImage(image=I1, flow=tf.stack([u, v]))

        dataTerm = I1_warped - I2

        u_next = u - self.alpha * dataTerm * I1_grad_x
        v_next = v - self.alpha * dataTerm * I1_grad_y

        return u_next, v_next