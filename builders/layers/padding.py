import tensorflow as tf


class ReflectionPad2D(tf.keras.layers.Layer):

    def __init__(self, paddings=(1, 1, 1, 1)):
        super(ReflectionPad2D, self).__init__()
        self.paddings = paddings

    def call(self, inputs):
        l, r, t, b = self.paddings

        return tf.pad(inputs, paddings=[[0, 0], [t, b], [l, r], [0, 0]], mode='REFLECT')
