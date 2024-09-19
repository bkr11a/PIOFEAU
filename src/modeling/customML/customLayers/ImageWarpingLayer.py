__author__ = "Brad Rice"
__version__ = "0.1.0"

import cv2
import tensorflow as tf

class ImageWarpingLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_dim(self, tensor, index):
        if tensor.shape.ndims is None:
            return tf.shape(tensor)[index]
        return tensor.shape[index] or tf.shape(tensor)[index]

    def _interpolate_bilinear(self, grid, query_points, indexing):
        if indexing != "ij" and indexing != "xy":
            raise ValueError("Indexing mode must be 'ij' or 'xy'")

        grid = tf.convert_to_tensor(grid)
        query_points = tf.convert_to_tensor(query_points)
        grid_shape = tf.shape(grid)
        query_shape = tf.shape(query_points)

        with tf.control_dependencies(
            [
                tf.debugging.assert_equal(tf.rank(grid), 4, "Grid must be 4D Tensor"),
                tf.debugging.assert_greater_equal(
                    grid_shape[1], 2, "Grid height must be at least 2."
                ),
                tf.debugging.assert_greater_equal(
                    grid_shape[2], 2, "Grid width must be at least 2."
                ),
                tf.debugging.assert_equal(
                    tf.rank(query_points), 3, "Query points must be 3 dimensional."
                ),
                tf.debugging.assert_equal(
                    query_shape[2], 2, "Query points last dimension must be 2."
                ),
            ]
        ):

            return self._interpolate_bilinear_impl(grid, query_points, indexing)

    def _interpolate_bilinear_impl(self, grid, query_points, indexing):
        with tf.name_scope("interpolate_bilinear"):
            grid_shape = tf.shape(grid)
            query_shape = tf.shape(query_points)

            batch_size, height, width, channels = (
                grid_shape[0],
                grid_shape[1],
                grid_shape[2],
                grid_shape[3],
            )

            num_queries = query_shape[1]

            query_type = query_points.dtype
            grid_type = grid.dtype

            alphas = []
            floors = []
            ceils = []
            index_order = [0, 1] if indexing == "ij" else [1, 0]
            unstacked_query_points = tf.unstack(query_points, axis=2, num=2)

            for i, dim in enumerate(index_order):
                with tf.name_scope("dim-" + str(dim)):
                    queries = unstacked_query_points[dim]

                    size_in_indexing_dimension = grid_shape[i + 1]

                    # max_floor is size_in_indexing_dimension - 2 so that max_floor + 1
                    # is still a valid index into the grid.
                    max_floor = tf.cast(size_in_indexing_dimension - 2, query_type)
                    min_floor = tf.constant(0.0, dtype=query_type)
                    floor = tf.math.minimum(
                        tf.math.maximum(min_floor, tf.math.floor(queries)), max_floor
                    )
                    int_floor = tf.cast(floor, tf.dtypes.int32)
                    floors.append(int_floor)
                    ceil = int_floor + 1
                    ceils.append(ceil)

                    # alpha has the same type as the grid, as we will directly use alpha
                    # when taking linear combinations of pixel values from the image.
                    alpha = tf.cast(queries - floor, grid_type)
                    min_alpha = tf.constant(0.0, dtype=grid_type)
                    max_alpha = tf.constant(1.0, dtype=grid_type)
                    alpha = tf.math.minimum(tf.math.maximum(min_alpha, alpha), max_alpha)

                    # Expand alpha to [b, n, 1] so we can use broadcasting
                    # (since the alpha values don't depend on the channel).
                    alpha = tf.expand_dims(alpha, 2)
                    alphas.append(alpha)

                flattened_grid = tf.reshape(grid, [batch_size * height * width, channels])
                batch_offsets = tf.reshape(
                    tf.range(batch_size) * height * width, [batch_size, 1]
                )

            # This wraps tf.gather. We reshape the image data such that the
            # batch, y, and x coordinates are pulled into the first dimension.
            # Then we gather. Finally, we reshape the output back. It's possible this
            # code would be made simpler by using tf.gather_nd.
            def gather(y_coords, x_coords, name):
                with tf.name_scope("gather-" + name):
                    linear_coordinates = batch_offsets + y_coords * width + x_coords
                    gathered_values = tf.gather(flattened_grid, linear_coordinates)
                    return tf.reshape(gathered_values, [batch_size, num_queries, channels])

            # grab the pixel values in the 4 corners around each query point
            top_left = gather(floors[0], floors[1], "top_left")
            top_right = gather(floors[0], ceils[1], "top_right")
            bottom_left = gather(ceils[0], floors[1], "bottom_left")
            bottom_right = gather(ceils[0], ceils[1], "bottom_right")

            # now, do the actual interpolation
            with tf.name_scope("interpolate"):
                interp_top = alphas[1] * (top_right - top_left) + top_left
                interp_bottom = alphas[1] * (bottom_right - bottom_left) + bottom_left
                interp = alphas[0] * (interp_bottom - interp_top) + interp_top

            return interp

    def bilinear_interpolation(self, grid, query_points, indexing = "ij"):
        return self._interpolate_bilinear(grid, query_points, indexing)

    def dense_image_warp(self, image, flow):
        with tf.name_scope("dense_image_warp"):
            image = tf.convert_to_tensor(image)
            flow = tf.convert_to_tensor(flow)
            batch, height, width, channels = (self.get_dim(image, 0),
                                            self.get_dim(image, 1),
                                            self.get_dim(image, 2),
                                            self.get_dim(image, 3))
            grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
            stacked_grid = tf.cast(tf.stack([grid_y, grid_x], axis = 2), flow.dtype)
            batched_grid = tf.expand_dims(stacked_grid, axis = 0)
            query_points_on_grid = batched_grid - flow
            query_points_flattened = tf.reshape(query_points_on_grid, [batch, height * width, 2])
            interpolated = self.bilinear_interpolation(image, query_points_flattened)
            interpolated = tf.reshape(interpolated, [batch, height, width, channels])
            return interpolated

    def warp_flow(self, image, flow):
        raise NotImplementedError("Currently this function will throw an OperatorNotAllowedInGraphError, still to explore reasons why")
        batch, height, width, channels = (self.get_dim(flow, 0),
                                          self.get_dim(flow, 1),
                                          self.get_dim(flow, 2),
                                          self.get_dim(flow, 3))
    
        remap_flow = tf.transpose(flow, perm=[0, 3, 1, 2])
        x = tf.range(0, width, dtype=tf.float32)
        y = tf.range(0, height, dtype=tf.float32)
        remap_xy = tf.meshgrid(x, y)
        remap_x, remap_y = (remap_xy + remap_flow)
        warpedImage = cv2.remap(image, remap_x, remap_y, cv2.INTER_LINEAR)
        return warpedImage

    @tf.function
    def call(self, Inputs):
        x = Inputs[0]
        flow = Inputs[1]
        warped = self.dense_image_warp(image=x, flow=-flow)
        # TODO: Fix the underneat since it raises an exception OperatorNotAllowedInGraphError. Investigate this!
        # warped = self.warp_flow(image = x, flow = flow)
        return warped


