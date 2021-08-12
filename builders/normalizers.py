import tensorflow as tf
import tensorflow_addons as tfa

import builders


class ConditionalInstanceNorm(tf.keras.layers.Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)
        self.beta_embedding = tf.keras.layers.Embedding(input_shape[1][-1], input_shape[0][-1], embeddings_initializer=tf.keras.initializers.Constant(0.))
        self.gamma_embedding = tf.keras.layers.Embedding(input_shape[1][-1], input_shape[0][-1], embeddings_initializer=tf.keras.initializers.Constant(1.))

    def call(self, inputs):
        x, y = inputs
        labels = tf.argmax(y, -1)
        mean, var = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        beta = self.beta_embedding(labels)[:, None, None, :]
        gamma = self.gamma_embedding(labels)[:, None, None, :]
        return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-10)


class AdaptiveInstanceNorm(tf.keras.layers.Layer):

    def __init__(self, use_sn=False, custom_dense=None):
        if custom_dense is None:
            self.dense_fun = tf.keras.layers.Dense
        else:
            self.dense_fun = custom_dense
        super().__init__()

    def build(self, input_shape):
        self.affine = self.dense_fun(input_shape[0][-1] * 2)
        self.size = input_shape[0][-1]

    def call(self, inputs):
        img, z = inputs
        style = self.affine(z)
        style_gamma, style_beta = style[:, :self.size], style[:, self.size:]

        return img + img * style_gamma[:, None, None, :] + style_beta[:, None, None, :]


def pixel_norm(x, epsilon=1e-8):
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    return x * tf.math.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def multi_stage_norm_denorm(x, y, w, custom_dense=None, custom_conv=None):
    # first does batch norm followed by discrete conditional denorm
    condnorm = ConditionalInstanceNorm()
    # then we do continuous denorm with w
    adainorm = AdaptiveInstanceNorm(custom_dense=custom_dense)
    return adainorm([condnorm([x, y]), w])

def adain(x, y, w, custom_dense=None, custom_conv=None):
    instance_normalized = tfa.layers.InstanceNormalization()(x)
    return AdaptiveInstanceNorm(custom_dense=custom_dense)([instance_normalized, w])

class YWWDenormCombined(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.y_conv = tf.keras.layers.Conv2D(input_shape[0][-1], 3, padding='same')
        self.w_conv = tf.keras.layers.Conv2D(input_shape[0][-1], 3, padding='same')
        self.condnorm = ConditionalInstanceNorm()
        self.condnorm.build([input_shape[0], input_shape[1]])
        self.adainorm_1 = AdaptiveInstanceNorm()
        self.adainorm_1.build([input_shape[0], input_shape[2]])
        self.adainorm_2 = AdaptiveInstanceNorm()
        self.adainorm_2.build([input_shape[0], input_shape[3]])
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.instance_norm.build(input_shape[0])

    def call(self, inputs):
        x, y, w_1, w_2 = inputs

        y_conditioned = self.condnorm([x, y])
        w_1_conditoned = self.adainorm_1([self.instance_norm(x), w_1])
        w_2_conditoned = self.adainorm_2([self.instance_norm(x), w_2])

        mask_inputs = tf.concat([y_conditioned, w_1_conditoned, w_2_conditoned], axis=-1)
        y_mask = self.y_conv(mask_inputs)
        y_mask = builders.layers.helpers.get_activation('sigmoid')(y_mask)

        w_mask = self.w_conv(mask_inputs)
        w_mask = builders.layers.helpers.get_activation('sigmoid')(w_mask)

        w_conditioned_masked = w_mask * w_1_conditoned + (1 - w_mask) * w_2_conditoned
        return y_mask * y_conditioned + (1 - y_mask) * w_conditioned_masked


def yww_denorm_combined(x, y, w_1, w_2):
    return YWWDenormCombined()([x, y, w_1, w_2])


class YWDenormCombined(tf.keras.layers.Layer):

    def __init__(self, use_sn=False, custom_dense=None, custom_conv=None):
        if custom_dense is None:
            self.dense_fun, self.conv_fun = tf.keras.layers.Dense, tf.keras.layers.Conv2D
        else:
            self.dense_fun, self.conv_fun = custom_dense, custom_conv
        super().__init__()

    def build(self, input_shape):
        self.conv = self.conv_fun(input_shape[0][-1], 3, padding='same')
        self.condnorm = ConditionalInstanceNorm()
        self.condnorm.build([input_shape[0], input_shape[1]])
        self.adainorm = AdaptiveInstanceNorm(custom_dense=self.dense_fun)
        self.adainorm.build([input_shape[0], input_shape[2]])
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.instance_norm.build(input_shape[0])

    def call(self, inputs):
        x, y, w = inputs

        y_conditioned = self.condnorm([x, y])
        w_conditoned = self.adainorm([self.instance_norm(x), w])

        mask_inputs = tf.concat([y_conditioned, w_conditoned], axis=-1)
        y_mask = self.conv(mask_inputs)
        y_mask = builders.layers.helpers.get_activation('sigmoid')(y_mask)

        return y_mask * y_conditioned + (1 - y_mask) * w_conditoned


def yw_denorm_combined(x, y, w, custom_dense=None, custom_conv=None):
    return YWDenormCombined(custom_dense=custom_dense,
                            custom_conv=custom_conv)([x, y, w])


class YWSpatialWDenormCombined(tf.keras.layers.Layer):

    def build(self, input_shape):
        self.y_conv = tf.keras.layers.Conv2D(input_shape[0][-1], 3, padding='same')
        self.w_conv = tf.keras.layers.Conv2D(input_shape[0][-1], 3, padding='same')
        self.condnorm = ConditionalInstanceNorm()
        self.condnorm.build([input_shape[0], input_shape[1]])
        self.adainorm = AdaptiveInstanceNorm()
        self.adainorm.build([input_shape[0], input_shape[2]])
        self.spade = builders.layers.stylegan.LayoutBasedModulation()
        self.spade.build([input_shape[0], input_shape[3]])
        self.instance_norm = tfa.layers.InstanceNormalization()
        self.instance_norm.build(input_shape[0])
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.batch_norm.build(input_shape[0])

    def call(self, inputs):
        x, y, w_1, w_2 = inputs

        y_conditioned = self.condnorm([x, y])
        w_1_conditoned = self.adainorm([self.instance_norm(x), w_1])
        w_2_conditoned = self.spade([self.batch_norm(x), w_2])

        mask_inputs = tf.concat([y_conditioned, w_1_conditoned, w_2_conditoned], axis=-1)
        y_mask = self.y_conv(mask_inputs)
        y_mask = builders.layers.helpers.get_activation('sigmoid')(y_mask)

        w_mask = self.w_conv(mask_inputs)
        w_mask = builders.layers.helpers.get_activation('sigmoid')(w_mask)

        w_conditioned_masked = w_mask * w_1_conditoned + (1 - w_mask) * w_2_conditoned
        return y_mask * y_conditioned + (1 - y_mask) * w_conditioned_masked


def yw_spatialw_denorm_combined(x, y, w_1, w_2):
    return YWSpatialWDenormCombined()([x, y, w_1, w_2])
