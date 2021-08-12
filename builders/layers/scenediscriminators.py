import tensorflow as tf

from builders.layers.helpers import build_cnn, get_normalization_2d
from builders.layers.spectral import SNConv2D
from utils.bbox import crop_bbox_batch


class AuxClasDiscriminator(tf.keras.layers.Layer):

    def __init__(self, arch, num_classes, normalization='none', activation='relu', padding='same', pooling='avg'):
        super().__init__()
        model = tf.keras.models.Sequential()
        cnn, _ = build_cnn(arch=arch,
                           normalization=normalization,
                           activation=activation,
                           pooling=pooling,
                           padding=padding,)
        model.add(cnn)
        model.add(tf.keras.layers.GlobalMaxPooling2D())
        model.add(tf.keras.layers.Dense(1024))
        self.cnn = model

        self.real_classifier = tf.keras.layers.Dense(1)
        self.obj_classifier = tf.keras.layers.Dense(num_classes)

    def build(self, input_shapes):
        self.cnn.build(input_shapes)

    def call(self, x, y):
        if len(x.shape) == 3:
            x = x[:, None]
        vecs = self.cnn(x)
        real_scores = self.real_classifier(vecs)
        obj_scores = self.obj_classifier(vecs)
        class_loss = tf.nn.softmax_cross_entropy_with_logits(logits=obj_scores,
                                                             labels=y)
        return real_scores, class_loss, obj_scores


class AuxClasCropDiscriminator(tf.keras.layers.Layer):

    def __init__(self, arch, num_classes, normalization='none', activation='relu',
                 object_size=64, padding='same', pooling='avg'):
        super().__init__()
        self.discriminator = AuxClasDiscriminator(
            arch, num_classes, normalization,
            activation, padding, pooling)
        self.object_size = object_size

    def call(self, imgs, objs, boxes, obj_to_img):
        crops = crop_bbox_batch(imgs, boxes, obj_to_img, self.object_size)
        real_scores, aux_clas_loss, obj_scores = self.discriminator(crops, objs)
        return real_scores, aux_clas_loss, obj_scores


class MultiscaleMaskDiscriminator(tf.keras.layers.Layer):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm='instance',
                 use_sigmoid=False, num_D=3, use_sn=False):
        super().__init__()
        self.num_D = num_D
        self.submodels = []

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm,
                                       use_sigmoid, kernel_size=3, use_sn=use_sn)
            self.submodels.append(netD)

        self.downsample = tf.keras.layers.AveragePooling2D((3, 3), strides=2, padding='same')

    def singleD_forward(self, submodel, input, cond):
        result = [input]
        for subsubmodel in submodel.submodels[:-2]:
            result.append(subsubmodel(result[-1]))

        b, c = tf.shape(result[-1])[1], tf.shape(result[-1])[2]
        cond = tf.expand_dims(tf.expand_dims(cond, 1), 1)
        cond = tf.tile(cond, (1, b, c, 1))
        concat = tf.concat([result[-1], cond], axis=-1)
        result.append(submodel.submodels[-2](concat))
        result.append(submodel.submodels[-1](result[-1]))
        return result[1:]

    def call(self, input, cond):
        result = []
        input_downsampled = input
        for i, submodel in enumerate(self.submodels):
            result.append(self.singleD_forward(submodel, input_downsampled, cond))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result


class NLayerDiscriminator(tf.keras.layers.Layer):
    """PatchGAN discriminator
    Based on the GauGAN/SPADE discriminator
    https://github.com/NVlabs/SPADE/blob/1a687baeada266a3c92be41295f1ea3d5efd4f93/models/networks/discriminator.py
    """

    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm='instance', use_sigmoid=False, kernel_size=3,
                 use_sn=False):
        super().__init__()

        Conv = tf.keras.layers.Conv2D if not use_sn else SNConv2D
        layers_by_submodel = [[
            Conv(ndf, kernel_size=kernel_size, strides=2, padding='same'),
            tf.keras.layers.LeakyReLU(0.2)
        ]]
        shapes_by_submodel = [(2**0, input_nc), (2, ndf)]

        nf = ndf
        for n in range(1, n_layers):
            nf = min(nf * 2, 512)
            layers_by_submodel.append([
                Conv(nf, kernel_size=kernel_size, strides=2, padding='same'),
                get_normalization_2d(norm), tf.keras.layers.LeakyReLU(0.2)
            ])
            shapes_by_submodel.append((2**(n*2), nf))

        nf = min(nf * 2, 512)
        layers_by_submodel.append([
            Conv(nf, kernel_size=kernel_size, strides=1, padding='same'),
            get_normalization_2d(norm),
            tf.keras.layers.LeakyReLU(0.2)
        ])
        shapes_by_submodel.append((2**(n*2), nf))

        layers_by_submodel.append([Conv(1, kernel_size=kernel_size, strides=1, padding='same')])
        shapes_by_submodel.append((2**(n*2), nf))

        if use_sigmoid:
            layers_by_submodel.append([tf.keras.layers.Activation('sigmoid')])
            shapes_by_submodel.append((2**(n*2), nf))

        self.submodels = []
        for model_layers in layers_by_submodel:
            submodel = tf.keras.models.Sequential()
            for l in model_layers:
                submodel.add(l)
            self.submodels.append(submodel)
        self.shapes_by_submodel = shapes_by_submodel

    def build(self, input_shape):
        n, w, h, c = input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        for shape, model in zip(self.shapes_by_submodel, self.submodels):
            ratio, nf = shape
            model.build((n, w//ratio, h//ratio, nf))

    def call(self, input):
        res = [input]
        for submodel in self.submodels:
            res.append(submodel(res[-1]))
        return res[1:]


class MultiscaleDiscriminator(tf.keras.layers.Layer):
    """Based on the GauGAN/SPADE discriminator
    https://github.com/NVlabs/SPADE/blob/1a687baeada266a3c92be41295f1ea3d5efd4f93/models/networks/discriminator.py
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm='instance',
                 use_sigmoid=False, num_D=3, use_sn=False):
        super().__init__()
        self.num_D = num_D
        self.submodels = []

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm,
                                       use_sigmoid, kernel_size=4, use_sn=use_sn)
            self.submodels.append(netD)

        self.downsample = tf.keras.layers.AveragePooling2D(2, padding='same')

    def build(self, input_shape):
        for i, submodel in enumerate(self.submodels):
            cur_shape = [input_shape[0], input_shape[1]//2**i, input_shape[2]//2**i, input_shape[3]]
            submodel.build(cur_shape)

    def call(self, inputs):
        result = []
        input_downsampled = inputs
        for i, submodel in enumerate(self.submodels):
            result.append(submodel(input_downsampled))
            if i != (self.num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result
