import tensorflow as tf


def build_VGG19_feature_extractor(input_shape):
    x = tf.keras.Input(shape=input_shape)
    base_model = VGG19(weights='imagenet',
                       include_top=False)
    outputs = []
    out = x
    for layer in base_model.layers:
        out = layer(out)
        if layer.name in ['block1_conv1', 'block2_conv1', 'block3_conv1',
                          'block4_conv1', 'block5_conv1']:
            outputs.append(out)
        if layer.name == 'block5_conv1':
            break
    return base_model, tf.keras.Model(inputs=x,
                                      outputs=outputs)

from tensorflow.python.keras.applications import imagenet_utils

WEIGHTS_PATH = ('https://storage.googleapis.com/tensorflow/keras-applications/'
                'vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5')
WEIGHTS_PATH_NO_TOP = ('https://storage.googleapis.com/tensorflow/'
                       'keras-applications/vgg19/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')


def preprocess_for_vgg(x):
    return imagenet_utils.preprocess_input(x * 255., data_format=tf.keras.backend.image_data_format(), mode='caffe')


def VGG19(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax'):
    """Instantiates the VGG19 architecture.
    Reference:
    - [Very Deep Convolutional Networks for Large-Scale Image Recognition](
        https://arxiv.org/abs/1409.1556) (ICLR 2015)
    By default, it loads weights pre-trained on ImageNet. Check 'weights' for
    other options.
    This model can be built both with 'channels_first' data format
    (channels, height, width) or 'channels_last' data format
    (height, width, channels).
    The default input size for this model is 224x224.
    Note: each Keras Application expects a specific kind of input preprocessing.
    For VGG19, call `tf.keras.applications.vgg19.preprocess_input` on your
    inputs before passing them to the model.
    Arguments:
      include_top: whether to include the 3 fully-connected
        layers at the top of the network.
      weights: one of `None` (random initialization),
          'imagenet' (pre-training on ImageNet),
          or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)`
        (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels,
        and width and height should be no smaller than 32.
        E.g. `(200, 200, 3)` would be one valid value.
      pooling: Optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional block.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional block, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
    Returns:
      A `keras.Model` instance.
    Raises:
      ValueError: in case of invalid argument for `weights`,
        or invalid input shape.
      ValueError: if `classifier_activation` is not `softmax` or `None` when
        using a pretrained top layer.
    """
    def relu(x):
        return tf.where(x <= 0., 0., x)
    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation=relu, padding='same', name='block1_conv1')(
            img_input)
    x = tf.keras.layers.Conv2D(
        64, (3, 3), activation=relu, padding='same', name='block1_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation=relu, padding='same', name='block2_conv1')(x)
    x = tf.keras.layers.Conv2D(
        128, (3, 3), activation=relu, padding='same', name='block2_conv2')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation=relu, padding='same', name='block3_conv1')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation=relu, padding='same', name='block3_conv2')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation=relu, padding='same', name='block3_conv3')(x)
    x = tf.keras.layers.Conv2D(
        256, (3, 3), activation=relu, padding='same', name='block3_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block4_conv1')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block4_conv2')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block4_conv3')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block4_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block5_conv1')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block5_conv2')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block5_conv3')(x)
    x = tf.keras.layers.Conv2D(
        512, (3, 3), activation=relu, padding='same', name='block5_conv4')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(4096, activation=relu, name='fc1')(x)
        x = tf.keras.layers.Dense(4096, activation=relu, name='fc2')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation,
                                  name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = tf.keras.utils.layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = tf.keras.Model(inputs, x, name='vgg19')

    # Load weights.
    if weights == 'imagenet':
        if include_top:
            weights_path = tf.keras.utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels.h5',
                WEIGHTS_PATH,
                cache_subdir='models',
                file_hash='cbe5617147190e668d6c5d5026f83318')
        else:
            weights_path = tf.keras.utils.get_file(
                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
                WEIGHTS_PATH_NO_TOP,
                cache_subdir='models',
                file_hash='253f8cb515780f3b799900260a226db6')
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


BASE_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/resnet/')
WEIGHTS_HASHES = {
    'resnet50': ('2cb95161c43110f7111970584f804107',
                 '4d473c1dd8becc155b73f8504c6f6626'),
    'resnet101': ('f1aeb4b969a6efcfb50fad2f0c20cfc5',
                  '88cf7a10940856eca736dc7b7e228a21'),
    'resnet152': ('100835be76be38e30d865e96f2aaae62',
                  'ee4c566cf9a93f14d82f913c2dc6dd0c'),
    'resnet50v2': ('3ef43a0b657b3be2300d5770ece849e0',
                   'fac2f116257151a9d068a22e544a4917'),
    'resnet101v2': ('6343647c601c52e1368623803854d971',
                    'c0ed64b8031c3730f411d2eb4eea35b5'),
    'resnet152v2': ('a49b44d1979771252814e80f8ec446f9',
                    'ed17cf2e0169df9d443503ef94b23b33'),
    'resnext50': ('67a5b30d522ed92f75a1f16eef299d1a',
                  '62527c363bdd9ec598bed41947b379fc'),
    'resnext101':
        ('34fb605428fcc7aa4d62f44404c11509', '0f678c91647380debd923963594981b3')
}

layers = None


def ResNet(stack_fn,
           preact,
           use_bias,
           model_name='resnet',
           include_top=True,
           weights='imagenet',
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classes=1000,
           classifier_activation='softmax',
           conv_fun=tf.keras.layers.Conv2D,
           **kwargs):
    if kwargs:
        raise ValueError('Unknown argument(s): %s' % (kwargs,))
    if not (weights in {'imagenet', None} or file_io.file_exists_v2(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                         ' as true, `classes` should be 1000')

    # Determine proper input shape
    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=tf.keras.backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = tf.keras.layers.Input(shape=input_shape)
    else:
        if not tf.keras.backend.is_keras_tensor(input_tensor):
            img_input = tf.keras.layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = tf.keras.layers.ZeroPadding2D(
        padding=((3, 3), (3, 3)), name='conv1_pad')(img_input)
    x = conv_fun(64, 7, strides=2, use_bias=use_bias, name='conv1_conv')(x)

    if not preact:
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name='conv1_relu')(x)

    x = tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)

    if preact:
        x = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = tf.keras.layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        imagenet_utils.validate_activation(classifier_activation, weights)
        x = tf.keras.layers.Dense(classes, activation=classifier_activation,
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = tf.keras.layers.GlobalMaxPooling2D(name='max_pool')(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = tf.keras.Model(inputs, x, name=model_name)

    # Load weights.
    if (weights == 'imagenet') and (model_name in WEIGHTS_HASHES):
        if include_top:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels.h5'
            file_hash = WEIGHTS_HASHES[model_name][0]
        else:
            file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_notop.h5'
            file_hash = WEIGHTS_HASHES[model_name][1]
        weights_path = tf.keras.utils.get_file(
            file_name,
            BASE_WEIGHTS_PATH + file_name,
            cache_subdir='models',
            file_hash=file_hash)
        model.load_weights(weights_path)
    elif weights is not None:
        model.load_weights(weights)

    return model


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None, conv_fun=tf.keras.layers.Conv2D):
    """A residual block.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.
    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = conv_fun(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = tf.keras.layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = conv_fun(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_1_relu')(x)

    x = conv_fun(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = tf.keras.layers.Activation('relu', name=name + '_2_relu')(x)

    x = conv_fun(4 * filters, 1, name=name + '_3_conv')(x)
    x = tf.keras.layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = tf.keras.layers.Add(name=name + '_add')([shortcut, x])
    x = tf.keras.layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, conv_fun=tf.keras.layers.Conv2D, name=None):
    """A set of stacked residual blocks.
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.
    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1', conv_fun=conv_fun)
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i), conv_fun=conv_fun)
    return x


def ResNet50(include_top=True,
             weights='imagenet',
             input_tensor=None,
             input_shape=None,
             pooling='avg',
             classes=1000,
             conv_fun=tf.keras.layers.Conv2D,
             **kwargs):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 128, 4, name='conv3')
        x = stack1(x, 256, 6, name='conv4')
        return stack1(x, 512, 3, name='conv5')

    return ResNet(stack_fn, False, True, 'resnet50', include_top, weights,
                  input_tensor, input_shape, pooling, classes, conv_fun=conv_fun, **kwargs)


def build_ResNet_feature_extractor(input_shape, conv_fun=tf.keras.layers.Conv2D, teacher=False):
    if teacher:
        base_model = ResNet50(weights='imagenet',
                              include_top=True,
                              conv_fun=conv_fun)
        return base_model
    else:
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              conv_fun=conv_fun,
                              input_shape=input_shape[:-1] + (3,))
        return tf.keras.Model(inputs=base_model.input, outputs=[base_model.get_layer('avg_pool').output,
                                                                base_model.get_layer('conv4_block6_out').output,
                                                                base_model.get_layer('conv3_block4_out').output,
                                                                base_model.get_layer('conv2_block3_out').output])
