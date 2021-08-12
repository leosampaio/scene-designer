import tensorflow as tf

from builders.layers.spectral import SNConv2D


class AddNoiseToEachChannel(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.scales = tf.Variable(tf.zeros(input_shape[-1],), trainable=True, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
        self.dense = tf.keras.layers.Dense(input_shape[-1], kernel_initializer='he_uniform')

    def call(self, x):
        input_shape = tf.shape(x)
        noise = tf.random.normal((input_shape[0], input_shape[1], input_shape[2], 1), 0., 1.)
        return x + noise * tf.reshape(self.scales, [1, 1, 1, -1])


class StyleGanBNoise(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.scale = tf.Variable(tf.zeros([]), trainable=True, aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

    def call(self, x):
        input_shape = tf.shape(x)
        noise = tf.random.normal((input_shape[0], input_shape[1], input_shape[2], 1))
        return x + noise * self.scale

class LayoutBasedModulation(tf.keras.layers.Layer):

    def __init__(self, use_sn=False):
        self.Conv2D = tf.keras.layers.Conv2D if not use_sn else SNConv2D
        super().__init__()

    def upsample(self, x, scale_factor=2):
        h, w = x.shape[1], x.shape[2]
        new_size = [h * scale_factor, w * scale_factor]
        return tf.image.resize(x, size=new_size, method='bilinear')

    def downsample(self, x, scale_factor=2):
        h, w = x.shape[1], x.shape[2]
        new_size = [h // scale_factor, w // scale_factor]
        return tf.image.resize(x, size=new_size, method='nearest')

    def build(self, input_shape):
        img_shape, layout_shape = input_shape
        self.w, self.h, self.channels = img_shape[1], img_shape[2], img_shape[3]
        layout_w = layout_shape[1]
        if layout_w > self.w:
            factor = layout_w // self.w
            self.resample = lambda x: self.downsample(x, scale_factor=factor)
        elif layout_w < self.w:
            factor = self.w // layout_w
            self.resample = lambda x: self.upsample(x, scale_factor=factor)
        else:
            self.resample = lambda x: x

        self.conv01 = self.Conv2D(128, kernel_size=5, padding='same')
        self.conv_gamma = self.Conv2D(self.channels, kernel_size=5, padding='same')
        self.conv_beta = self.Conv2D(self.channels, kernel_size=5, padding='same')

    def call(self, inputs):
        img, layout = inputs
        style = self.resample(layout)
        style = self.conv01(style)
        style_gamma = self.conv_gamma(style)
        style_beta = self.conv_beta(style)

        return img + img * style_gamma + style_beta
