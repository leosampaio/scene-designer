import tensorflow as tf
import tensorflow_addons as tfa

import builders
from builders.layers.helpers import build_cnn, get_normalization_2d
from builders.layers import stylegan
from builders.layers.spectral import SNConv2D
from builders.layers.syncbn import SyncBatchNormalization


def build_mask_net(hidden_channel_dim, mask_size, norm='batch'):
    output_dim = 1
    cur_size = 1
    model = tf.keras.models.Sequential()
    while cur_size < mask_size:
        model.add(tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='nearest'))
        model.add(tf.keras.layers.Conv2D(hidden_channel_dim, kernel_size=3, padding='same'))
        model.add(get_normalization_2d(norm))
        model.add(tf.keras.layers.Activation('relu'))
        cur_size *= 2
    if cur_size != mask_size:
        raise ValueError('Mask size must be a power of 2')
    model.add(tf.keras.layers.Conv2D(output_dim, kernel_size=1, padding='same'))
    return model


class AppearanceEncoder(tf.keras.layers.Layer):

    def __init__(self, arch, normalization='none', activation='relu',
                 padding='same', vecs_size=1024, pooling='avg'):
        super().__init__()
        cnn, channels = build_cnn(arch=arch,
                                  normalization=normalization,
                                  activation=activation,
                                  pooling=pooling,
                                  padding=padding)
        self.cnn = tf.keras.models.Sequential()
        self.cnn.add(cnn)
        self.cnn.add(tf.keras.layers.GlobalMaxPooling2D())
        self.cnn.add(tf.keras.layers.Dense(vecs_size))

    def call(self, crops):
        return self.cnn(crops)


class LayoutToImageGenerator(tf.keras.layers.Layer):

    def __init__(self, input_shape, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer='batch',
                 padding_type='same', final_activation='tanh', extra_mult=2):
        super().__init__()
        assert (n_blocks >= 0)

        if norm_layer == 'batch':
            norm_layer_builder = tf.keras.layers.BatchNormalization
        elif norm_layer == 'instance':
            norm_layer_builder = tfa.layers.InstanceNormalization

        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]]),
            input_shape=input_shape))
        model.add(tf.keras.layers.Conv2D(ngf, kernel_size=7, padding='valid'))

        model.add(norm_layer_builder())
        model.add(tf.keras.layers.Activation('relu'))

        # downsample
        for i in range(n_downsampling):
            n_filters = int(ngf * (2 ** i) * extra_mult)
            model.add(tf.keras.layers.Conv2D(n_filters, kernel_size=3, strides=2, padding='same'))
            model.add(norm_layer_builder())
            model.add(tf.keras.layers.Activation('relu'))

        # resnet blocks
        n_filters = int(ngf * (2 ** (n_downsampling - 1)) * extra_mult)
        for i in range(n_blocks):
            model.add(builders.layers.resnet.ResidualBlock(n_filters, padding=padding_type, activation='relu', normalization=norm_layer))

        # upsample
        for i in range(n_downsampling):
            n_filters = int(ngf * (2 ** (n_downsampling - i)) * extra_mult / 2)
            model.add(tf.keras.layers.Conv2DTranspose(n_filters, kernel_size=3, strides=2, padding='same', output_padding=1))
            model.add(norm_layer_builder())
            model.add(tf.keras.layers.Activation('relu'))

        model.add(tf.keras.layers.Lambda(
            lambda x: tf.pad(x, [[0, 0], [3, 3], [3, 3], [0, 0]])))
        model.add(tf.keras.layers.Conv2D(output_nc, kernel_size=7, padding='valid'))
        model.add(tf.keras.layers.Activation(final_activation))
        self.model = model

    def call(self, input):
        return self.model(input)


class LayoutToImageGeneratorSPADESArch(tf.keras.layers.Layer):

    def __init__(self, nf=512, image_size=128, input_c=205,
                 random_input=False, use_sn=False, add_noise=True,
                 norm='instance'):
        super().__init__()
        self.nf = nf
        self.image_size = image_size
        self.Conv2D = tf.keras.layers.Conv2D if not use_sn else SNConv2D
        if norm == 'instance':
            self.Norm = tfa.layers.InstanceNormalization
        elif norm == 'batch':
            self.Norm = SyncBatchNormalization
        self.use_sn = use_sn
        self.add_noise = add_noise
        self.random_input = random_input

        layout_in = tf.keras.Input((image_size, image_size, input_c))
        if not self.random_input:
            self.initial_image = tf.Variable(tf.ones((1, 4, 4, self.nf)), trainable=True, name='initial_image', aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)
            x = tf.tile(self.initial_image, [tf.shape(layout_in)[0], 1, 1, 1])
        else:
            self.initial_image = tf.random.normal((tf.shape(layout_in)[0], 256), 0., 1.)
            x = tf.keras.layers.Dense(self.nf * 4 * 4)(self.initial_image)
            x = tf.reshape(x, [tf.shape(layout_in)[0], 4, 4, self.nf])

        x = self.mescheder_resblock(x, layout_in, c=self.nf)
        x = self.upsample(x, 2)  # 8

        x = self.adaptive_spacial_instance_norm(x, layout_in)
        x = self.mescheder_resblock(x, layout_in, c=self.nf)
        x = self.upsample(x, 2)  # 16

        x = self.adaptive_spacial_instance_norm(x, layout_in)
        x = self.mescheder_resblock(x, layout_in, c=self.nf)
        x = self.upsample(x, 2)  # 32

        cur_nf = self.nf // 2
        x = self.adaptive_spacial_instance_norm(x, layout_in)
        x = self.mescheder_resblock(x, layout_in, c=cur_nf, learn_skip=True)
        x = self.upsample(x, 2)  # 64

        cur_nf = cur_nf // 2
        x = self.adaptive_spacial_instance_norm(x, layout_in)
        x = self.mescheder_resblock(x, layout_in, c=cur_nf, learn_skip=True)
        x = self.upsample(x, 2)  # 128

        if self.image_size == 256:
            cur_nf = cur_nf // 2
            x = self.adaptive_spacial_instance_norm(x, layout_in)
            x = self.mescheder_resblock(x, layout_in, c=cur_nf, learn_skip=True)
            x = self.upsample(x, 2)  # 256

        cur_nf = cur_nf // 2
        x = self.adaptive_spacial_instance_norm(x, layout_in)
        x = self.mescheder_resblock(x, layout_in, c=cur_nf, learn_skip=True)

        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.Conv2D(3, kernel_size=3, padding='same')(x)
        x = tf.keras.layers.Activation('tanh')(x)

        self.model = tf.keras.Model(
            name='stylegan_spades_mix_generator',
            inputs=[layout_in],
            outputs=x)

    def adaptive_spacial_instance_norm(self, x, layout):
        if self.add_noise:
            x = stylegan.AddNoiseToEachChannel()(x)
        x = tf.keras.layers.LeakyReLU(0.2)(x)
        x = self.Norm()(x)
        x = stylegan.LayoutBasedModulation(self.use_sn)([x, layout])

        return x

    def upsample(self, x, scale_factor=2):
        h, w = x.shape[1], x.shape[2]
        new_size = [h * scale_factor, w * scale_factor]
        return tf.image.resize(x, size=new_size, method='bilinear')

    def mescheder_resblock(self, x, layout, c=1024, learn_skip=False):
        out = self.Conv2D(c, kernel_size=3, padding='same')(x)
        out = self.adaptive_spacial_instance_norm(out, layout)
        out = self.Conv2D(c, kernel_size=3, padding='same')(out)
        out = self.adaptive_spacial_instance_norm(out, layout)

        if learn_skip:
            x = self.Conv2D(c, kernel_size=3, padding='same')(x)
            x = self.adaptive_spacial_instance_norm(x, layout)

        return x + out

    def call(self, input):
        return self.model(input)
