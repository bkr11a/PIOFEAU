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
            warped_image = bilinear_sampler(image, new_coords)
            
            return warped_image

        def bilinear_sampler(img, coords):
            """
            Bilinear sampling of the image at coordinates.
            
            Args:
            - img: A 4D tensor of shape (batch_size, height, width, channels).
            - coords: A 4D tensor of shape (batch_size, height, width, 2), containing normalized coordinates.

            Returns:
            - A 4D tensor of the sampled image.
            """
            batch_size, height, width, channels = img.shape

            x = coords[..., 0]
            y = coords[..., 1]

            x0 = tf.floor(x)
            x1 = x0 + 1
            y0 = tf.floor(y)
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, 0, tf.cast(width - 1, tf.float32))
            x1 = tf.clip_by_value(x1, 0, tf.cast(width - 1, tf.float32))
            y0 = tf.clip_by_value(y0, 0, tf.cast(height - 1, tf.float32))
            y1 = tf.clip_by_value(y1, 0, tf.cast(height - 1, tf.float32))

            Ia = get_pixel_value(img, x0, y0)
            Ib = get_pixel_value(img, x0, y1)
            Ic = get_pixel_value(img, x1, y0)
            Id = get_pixel_value(img, x1, y1)

            wa = (x1 - x) * (y1 - y)
            wb = (x1 - x) * (y - y0)
            wc = (x - x0) * (y1 - y)
            wd = (x - x0) * (y - y0)

            wa = tf.expand_dims(wa, axis=-1)
            wb = tf.expand_dims(wb, axis=-1)
            wc = tf.expand_dims(wc, axis=-1)
            wd = tf.expand_dims(wd, axis=-1)

            return wa * Ia + wb * Ib + wc * Ic + wd * Id

        def get_pixel_value(img, x, y):
            """
            Get pixel value for coordinate (x, y) from a 4D tensor image.

            Args:
            - img: A 4D tensor of shape (batch_size, height, width, channels).
            - x: A 2D tensor of x coordinates.
            - y: A 2D tensor of y coordinates.

            Returns:
            - A 4D tensor of the pixel values.
            """
            x = tf.cast(x, tf.int32)
            y = tf.cast(y, tf.int32)
            
            batch_size, height, width, channels = img.shape
            batch_indices = tf.range(batch_size, dtype=tf.int32)
            batch_indices = tf.reshape(batch_indices, (batch_size, 1, 1))
            b = tf.tile(batch_indices, (1, height, width))

            indices = tf.stack([b, y, x], axis=-1)
            return tf.gather_nd(img, indices)
        
        # Grab the inputs from the layer
        I1, I2, flow = Inputs

        # Approximate the image gradients
        I1_grad_x, I1_grad_y = tf.image.image_gradients(I1)
        I2_grad_x, I2_grad_y = tf.image.image_gradients(I2)

        # Compute Image Warping here!
        # warp image using appropriate method.
        I1_warped = warpImage(image=I1, flow=flow)
        u = flow[:, :, :, 0]
        v = flow[:, :, :, 1]

        dataTerm = I1_warped - I2

        u_next = u - self.alpha * dataTerm[:, :, :, 0] * I1_grad_x[:, :, :, 0]
        v_next = v - self.alpha * dataTerm[:, :, :, 0] * I1_grad_y[:, :, :, 0]

        flow_next = tf.stack([u_next, v_next], axis = -1)

        return flow_next