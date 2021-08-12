import tensorflow as tf
import tensorflow_addons as tfa

import builders


def build_mlp(dim_list, activation='relu', batch_norm='none',
              dropout=0, final_nonlinearity=True):
    model = tf.keras.models.Sequential()
    for i in range(len(dim_list) - 1):
        dim_out = dim_list[i + 1]
        model.add(tf.keras.layers.Dense(dim_out))
        is_final_layer = (i == len(dim_list) - 2)
        if not is_final_layer or final_nonlinearity:
            if batch_norm == 'batch':
                model.add(tf.keras.layers.BatchNormalization())
            if activation == 'relu':
                model.add(tf.keras.layers.Activation('relu'))
            elif activation == 'leakyrelu':
                model.add(tf.keras.layers.LeakyReLU())
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(rate=dropout))
    return model


def build_cnn(arch, normalization='batch', activation='relu', padding='same',
              pooling='max', init='default'):
    """
    Build a CNN from an architecture string, which is a list of layer
    specification strings. The overall architecture can be given as a list or as
    a comma-separated string.

    All convolutions *except for the first* are preceeded by normalization and
    nonlinearity.

    All other layers support the following:
    - IX: Indicates that the number of input channels to the network is X.
          Can only be used at the first layer; if not present then we assume
          3 input channels.
    - CK-X: KxK convolution with X output channels
    - CK-X-S: KxK convolution with X output channels and stride S
    - R: Residual block keeping the same number of channels
    - UX: Nearest-neighbor upsampling with factor X
    - PX: Spatial pooling with factor X
    - FC-X-Y: Flatten followed by fully-connected layer

    Returns a tuple of:
    - cnn: An nn.Sequential
    - channels: Number of output channels
    """
    if isinstance(arch, str):
        arch = arch.split(',')
    cur_C = 3
    if len(arch) > 0 and arch[0][0] == 'I':
        cur_C = int(arch[0][1:])
        arch = arch[1:]

    first_conv = True
    flat = False
    layers = []
    for i, s in enumerate(arch):
        if s[0] == 'C':
            if not first_conv:
                layers.append(get_normalization_2d(normalization))
                layers.append(get_activation(activation))
            first_conv = False
            vals = [int(i) for i in s[1:].split('-')]
            if len(vals) == 2:
                K, next_C = vals
                stride = 1
            elif len(vals) == 3:
                K, next_C, stride = vals
            conv = tf.keras.layers.Conv2D(next_C, kernel_size=K, padding=padding, strides=stride)
            layers.append(conv)
            cur_C = next_C
        elif s[0] == 'R':
            norm = 'none' if first_conv else normalization
            res = builders.layers.resnet.ResidualBlock(cur_C, normalization=norm,
                                                       activation=activation,
                                                       padding=padding, init=init)
            layers.append(res)
            first_conv = False
        elif s[0] == 'U':
            factor = int(s[1:])
            layers.append(tf.keras.layers.UpSampling2D(size=(factor, factor),
                                                       interpolation='nearest'))
        elif s[0] == 'P':
            factor = int(s[1:])
            if pooling == 'max':
                pool = tf.keras.layers.MaxPool2d(kernel_size=factor)
            elif pooling == 'avg':
                pool = tf.keras.layers.AvgPool2d(kernel_size=factor)
            layers.append(pool)
        elif s[:2] == 'FC':
            _, Din, Dout = s.split('-')
            Din, Dout = int(Din), int(Dout)
            if not flat:
                layers.append(tf.keras.layers.Flatten())
            flat = True
            layers.append(tf.keras.layers.Dense(Dout))
            if i + 1 < len(arch):
                layers.append(get_activation(activation))
            cur_C = Dout
        else:
            raise ValueError('Invalid layer "%s"' % s)
    layers = [layer for layer in layers if layer is not None]
    model = tf.keras.models.Sequential()
    for layer in [l for l in layers if l is not None]:
        model.add(layer)
    return model, cur_C


def get_normalization_2d(normalization, reusable=False):
    if normalization == 'batch':
        layer = tf.keras.layers.BatchNormalization
        norm_layer = (lambda x: layer()(x)) if reusable else layer()
    elif normalization == 'instance':
        layer = tfa.layers.InstanceNormalization
        norm_layer = (lambda x: layer()(x)) if reusable else layer()
    elif normalization == 'group':
        norm_layer = lambda x: tfa.layers.GroupNormalization(groups=16)(x)
    elif normalization == 'adain':
        norm_layer = lambda x, z: builders.normalizers.AdaptiveInstanceNorm()([tfa.layers.InstanceNormalization()(x), z])
    elif normalization == 'multistage-denorm':
        norm_layer = builders.normalizers.multi_stage_norm_denorm
    elif normalization == 'pixelnorm':
        norm_layer = builders.normalizers.pixel_norm
    elif normalization == 'yww-denorm':
        norm_layer = builders.normalizers.yww_denorm_combined
    elif normalization == 'yw-denorm':
        norm_layer = builders.normalizers.yw_denorm_combined
    elif normalization == 'ywsw-denorm':
        norm_layer = builders.normalizers.yw_spatialw_denorm_combined
    elif normalization == 'none':
        return None
    elif normalization.lower().startswith('conditionalin'):
        norm_layer = lambda x: builders.normalizers.ConditionalInstanceNorm()(x)
    else:
        raise ValueError("Unrecognized normalization type '{}'".format(normalization))
    return norm_layer


def get_activation(activation):
    if activation.lower().startswith('leakyrelu'):
        alpha = float(activation.split('-')[1])
        layer = tf.keras.layers.LeakyReLU(alpha)
    elif activation.lower().startswith('miurelu'):
        layer = builders.activations.miu_relu
    elif activation.lower().startswith('prelu'):
        layer = builders.activations.prelu
    else:
        layer = tf.keras.layers.Activation(activation)
    return layer
