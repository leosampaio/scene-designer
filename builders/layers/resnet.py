import tensorflow as tf

import builders


class ResidualBlock(tf.keras.layers.Layer):

    def __init__(self, channels, normalization='batch', activation='relu',
                 padding='same', kernel_size=3):
        super(ResidualBlock, self).__init__()

        K = kernel_size
        C = channels
        layers = [
            builders.layers.helpers.get_normalization_2d(normalization),
            builders.layers.helpers.get_activation(activation),
            tf.keras.layers.Conv2D(C, kernel_size=K, padding=padding),
            builders.layers.helpers.get_normalization_2d(normalization),
            builders.layers.helpers.get_activation(activation),
            tf.keras.layers.Conv2D(C, kernel_size=K, padding=padding),
        ]
        layers = [layer for layer in layers if layer is not None]
        model = tf.keras.models.Sequential()
        for layer in layers:
            model.add(layer)
        self.net = model

    def call(self, x):
        shortcut = x
        return shortcut + self.net(x)
