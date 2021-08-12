import tensorflow as tf

from builders.layers.scenegenerators import (LayoutToImageGenerator,
                                             LayoutToImageGeneratorSPADESArch)
from builders.layers.scenediscriminators import (MultiscaleDiscriminator,
                                                 MultiscaleMaskDiscriminator)
from builders.layers.helpers import build_mlp


class CommonEmbeddingLayersMixin(object):

    def build_pretrained_image_encoder(self, name="0"):
        image_shape = (self.hps['image_size'], self.hps['image_size'], 3)
        x = tf.keras.Input(shape=image_shape, dtype=tf.float32)
        if self.hps['pretrained_e_model'] == 'mobilenet2':
            base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                           weights='imagenet')
            cut_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('block_16_project_BN').output)
            feats = cut_model(x)
            feats = tf.keras.layers.GlobalAveragePooling2D()(feats)
        elif self.hps['pretrained_e_model'] == 'inception':
            base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
            feats = base_model(x)
            feats = tf.keras.layers.GlobalAveragePooling2D()(feats)
        elif self.hps['pretrained_e_model'] == 'resnet':
            base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet')
            feats = base_model(x)
            feats = tf.keras.layers.GlobalAveragePooling2D()(feats)

        feats = tf.keras.layers.Dense(self.hps['encoding_size'] * 2)(feats)
        feats = tf.keras.layers.Dense(self.hps['encoding_size'])(feats)
        return tf.keras.Model(
            name='image_encoder_{}'.format(name),
            inputs=[x],
            outputs=[feats])

    def build_pretrained_sketch_encoder(self, name="0"):
        skt_shape = (self.hps['image_size'], self.hps['image_size'], 1)
        x = tf.keras.Input(shape=skt_shape, dtype=tf.float32)

        feats = tf.tile(x, [1, 1, 1, 3])
        if self.hps['pretrained_e_model'] == 'mobilenet2':
            base_model = tf.keras.applications.MobileNetV2(include_top=False,
                                                           weights='imagenet')
            self.sketch_cut_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('block_16_project_BN').output)
            self.sketch_small_model = tf.keras.models.Model(inputs=base_model.input, outputs=base_model.get_layer('block_8_project_BN').output)
            feats = self.sketch_cut_model(feats)
            feats = tf.keras.layers.GlobalAveragePooling2D()(feats)
        elif self.hps['pretrained_e_model'] == 'inception':
            base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights='imagenet')
            feats = base_model(feats)
            feats = tf.keras.layers.GlobalAveragePooling2D()(feats)
        elif self.hps['pretrained_e_model'] == 'resnet':
            base_model = tf.keras.applications.resnet_v2.ResNet50V2(include_top=False, weights='imagenet')
            feats = base_model(feats)
            feats = tf.keras.layers.GlobalAveragePooling2D()(feats)

        feats = tf.keras.layers.Dense(self.hps['encoding_size'])(feats)
        return tf.keras.Model(
            name='sketch_encoder_{}'.format(name),
            inputs=[x],
            outputs=[feats])

    def build_common_representation_net(self):
        repr_layers = [self.hps['encoding_size'], self.hps['edge_repr_net_hidden_size'], self.hps['common_repr_size']]
        return build_mlp(repr_layers, batch_norm='none')

    def build_full_image_discriminator(self):
        if self.hps['img_d_arch'] == 'big':
            dimage_ndf = 64
            dimage_n_layers = 3
            dimage_num_D = 2
        elif self.hps['img_d_arch'] == 'small':
            dimage_ndf = 32
            dimage_n_layers = 3
            dimage_num_D = 2
        elif self.hps['img_d_arch'] == 'verysmall':
            dimage_ndf = 32
            dimage_n_layers = 2
            dimage_num_D = 2
        if self.hps['use_appearence']:
            netD_input_nc = self.num_objs + self.hps['rep_size'] + 3
        else:
            netD_input_nc = self.num_objs + 3
        return MultiscaleDiscriminator(
            input_nc=netD_input_nc,
            ndf=dimage_ndf,
            n_layers=dimage_n_layers,
            norm='instance',
            use_sigmoid=False,
            num_D=dimage_num_D,
            use_sn=self.hps['use_sn_img_dis'])

    def build_layout_discriminator(self, name="0"):
        if self.hps['layout_d_arch'] == 'big':
            dimage_ndf = 32
            dimage_n_layers = 3
            dimage_num_D = 2
        elif self.hps['layout_d_arch'] == 'small':
            dimage_ndf = 32
            dimage_n_layers = 3
            dimage_num_D = 1
        elif self.hps['layout_d_arch'] == 'verysmall':
            dimage_ndf = 16
            dimage_n_layers = 2
            dimage_num_D = 1
        netD_input_nc = self.num_objs if not self.hps['do_channel_dis'] else self.num_objs + 1

        x = tf.keras.Input(shape=(self.image_size[0], self.image_size[1], netD_input_nc), dtype=tf.float32)
        discriminator = MultiscaleDiscriminator(
            input_nc=netD_input_nc,
            ndf=dimage_ndf,
            n_layers=dimage_n_layers,
            norm=self.hps['d_layout_norm'],
            use_sigmoid=False,
            num_D=dimage_num_D)
        scores = discriminator(x)
        return tf.keras.Model(
            name='layout_discriminator_{}'.format(name),
            inputs=[x],
            outputs=scores)

    def build_mask_discriminator(self, name="0"):
        if self.hps['layout_d_arch'] == 'big':
            ndf, n_layers = 64, 3
        elif self.hps['layout_d_arch'] == 'small':
            ndf, n_layers = 32, 3
        elif self.hps['layout_d_arch'] == 'verysmall':
            ndf, n_layers = 16, 2
        nc = 1 if not self.hps['do_channel_dis'] else 2
        x = tf.keras.Input(shape=(self.dataset.mask_size, self.dataset.mask_size, nc), dtype=tf.float32)
        y = tf.keras.Input(shape=(self.num_objs,), dtype=tf.float32)
        mask_dis = MultiscaleMaskDiscriminator(
            input_nc=nc, ndf=ndf, n_layers=n_layers,
            norm=self.hps['d_layout_norm'], use_sigmoid=False, num_D=1,
            use_sn=self.hps['use_sn_mask_dis'])
        scores = mask_dis(x, y)
        return tf.keras.Model(
            name='mask_discriminator_{}'.format(name),
            inputs=[x, y],
            outputs=scores)

    def build_layout_conditioned_image_generator(self):
        if self.hps['use_appearence']:
            netG_input_nc = self.num_objs + self.hps['rep_size']
        else:
            netG_input_nc = self.num_objs
        if self.hps['img_g_arch'] == 'big':
            gen_ngf = 64
            gen_n_blocks_global = 9
            n_downsample_global = 4
            extra_mult = 2
        elif self.hps['img_g_arch'] == 'small':
            gen_ngf = 32
            gen_n_blocks_global = 6
            n_downsample_global = 4
            extra_mult = 1
        elif self.hps['img_g_arch'] == 'verysmall':
            gen_ngf = 32
            gen_n_blocks_global = 4
            n_downsample_global = 3
            extra_mult = 0.5
        elif self.hps['img_g_arch'].startswith('spades'):
            if self.hps['img_g_arch'] == 'spades_orig':
                nf = 1024
                random_input = True
                use_sn = True
                add_noise = False
                norm = 'batch'
            else:
                random_input = False
                use_sn = False
                add_noise = True
                norm = 'instance'
            if self.hps['img_g_arch'] == 'spades':
                nf = 512
            elif self.hps['img_g_arch'] == 'spades_small':
                nf = 256
            elif self.hps['img_g_arch'] == 'spades_verysmall':
                nf = 128
            return LayoutToImageGeneratorSPADESArch(nf=nf, image_size=self.image_size[0], input_c=netG_input_nc,
                                                    random_input=random_input, use_sn=use_sn, add_noise=add_noise, norm=norm)
        return LayoutToImageGenerator(input_shape=(self.image_size[0], self.image_size[1], netG_input_nc),
                                      output_nc=3,
                                      ngf=gen_ngf,
                                      n_downsampling=n_downsample_global,
                                      n_blocks=gen_n_blocks_global,
                                      norm_layer='instance',
                                      padding_type='same',
                                      final_activation='tanh',
                                      extra_mult=extra_mult)
