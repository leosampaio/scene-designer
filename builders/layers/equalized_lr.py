'''
    From https://stackoverflow.com/questions/60012406/how-may-i-do-equalized-learning-rate-with-tensorflow-2
'''

import tensorflow as tf
import numpy as np
# from tensorflow.keras.utils import conv_utils


class DenseEQ(tf.keras.layers.Dense):
    """

    Standard dense layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Dense layer and overides the call method.
    """

    def __init__(self, *args, learning_rate=1., **kwargs):
        self.learning_rate = learning_rate
        std_dev = 1.0 / learning_rate
        if 'kernel_initializer' in kwargs:
            print("Trying to override kernel_initializer on eq. learning rate layer")
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=std_dev), **kwargs)
        

    def build(self, input_shape):
        super().build(input_shape)
        self.c = 1 / np.sqrt(input_shape[-1])

    def call(self, inputs):
        output = tf.keras.backend.dot(inputs, self.kernel * self.c * self.learning_rate)  # scale kernel
        if self.use_bias:
            output = tf.keras.backend.bias_add(output, self.bias * self.learning_rate, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Conv2DEQ(tf.keras.layers.Conv2D):
    """
    Standard Conv2D layer but includes learning rate equilization
    at runtime as per Karras et al. 2017.

    Inherits Conv2D layer and overrides the call method, following
    https://github.com/keras-team/keras/blob/master/keras/layers/convolutional.py

    """

    def __init__(self, *args, learning_rate=1., **kwargs):
        self.learning_rate = learning_rate
        std_dev = 1.0 / learning_rate
        if 'kernel_initializer' in kwargs:
            print("Trying to override kernel_initializer on eq. learning rate layer")
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=std_dev), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)
        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        fan_in = np.prod(kernel_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        self.c = 1 / np.sqrt(fan_in)  # He init

    def call(self, inputs):
        if self.rank == 2:
            outputs = tf.keras.backend.conv2d(
                inputs,
                self.kernel * self.c * self.learning_rate,  # scale kernel
                strides=self.strides,
                padding=self.padding,
                data_format=self.data_format,
                dilation_rate=self.dilation_rate)

        if self.use_bias:
            outputs = tf.keras.backend.bias_add(
                outputs,
                self.bias * self.learning_rate,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)
        return outputs

class Conv2DTransposeEQ(tf.keras.layers.Conv2DTranspose):
    def __init__(self, *args, learning_rate=1., **kwargs):
        self.learning_rate = learning_rate
        std_dev = 1.0 / learning_rate
        if 'kernel_initializer' in kwargs:
            print("Trying to override kernel_initializer on eq. learning rate layer")
            super().__init__(*args, **kwargs)
        else:
            super().__init__(*args, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=std_dev), **kwargs)

    def build(self, input_shape):
        super().build(input_shape)

        kernel_shape = self.kernel_size + (input_dim, self.filters)
        fan_in = np.prod(kernel_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
        self.c = 1 / np.sqrt(fan_in)  # He init

    def call(self, inputs):
        inputs_shape = tf.compat.v1.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
          h_axis, w_axis = 2, 3
        else:
          h_axis, w_axis = 1, 2

        # Use the constant height and weight when possible.
        # TODO(scottzhu): Extract this into a utility function that can be applied
        # to all convolutional layers, which currently lost the static shape
        # information due to tf.shape().
        height, width = None, None
        if inputs.shape.rank is not None:
          dims = inputs.shape.as_list()
          height = dims[h_axis]
          width = dims[w_axis]
        height = height if height is not None else inputs_shape[h_axis]
        width = width if width is not None else inputs_shape[w_axis]

        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
          out_pad_h = out_pad_w = None
        else:
          out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = inputs_shape[1] * stride_h
        out_width = inputs_shape[2] * stride_w
        if self.data_format == 'channels_first':
          output_shape = (batch_size, self.filters, out_height, out_width)
        else:
          output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = tf.stack(output_shape)
        outputs = tf.keras.backend.conv2d_transpose(
            inputs,
            self.kernel * self.c * self.learning_rate,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not tf.executing_eagerly():
          # Infer the static output shape:
          out_shape = self.compute_output_shape(inputs.shape)
          outputs.set_shape(out_shape)

        # if self.use_bias:
        #   outputs = tf.nn.bias_add(
        #       outputs,
        #       self.bias,
        #       data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
          return self.activation(outputs)
        return outputs