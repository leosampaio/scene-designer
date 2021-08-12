import tensorflow as tf

import builders


def mru_conv(inp, ht, filter_depth, sn,
             stride, dilate=1,
             activation='relu',
             normalizer=None,
             normalizer_params=None,
             weights_initializer=tf.keras.initializers.GlorotUniform(),
             biases_initializer_mask=tf.keras.initializers.Constant(0.5),
             weight_decay_rate=1e-8,
             norm_mask=False,
             norm_input=True,
             labels=None,
             paper_version=False,
             w_rep=None):
    Conv2D = tf.keras.layers.Conv2D if not sn else builders.layers.spectral.SNConv2D
    activation_fn = builders.layers.helpers.get_activation(activation) if activation else tf.identity
    normalizer_fn_ = builders.layers.helpers.get_normalization_2d(normalizer, reusable=True) if normalizer else tf.identity
    if normalizer == 'conditionalin':
        normalizer_fn = lambda x: normalizer_fn_([x, labels])
    elif normalizer == 'adain':
        normalizer_fn = lambda x: normalizer_fn_(x, w_rep)
    else:
        normalizer_fn = normalizer_fn_
    mask_normalizer_fn = normalizer_fn if norm_mask else tf.identity

    regularizer = tf.keras.regularizers.L2(weight_decay_rate) if weight_decay_rate > 0 else None
    weights_initializer_mask = weights_initializer
    biases_initializer = tf.zeros_initializer()

    ht_orig = tf.identity(ht)
    hidden_depth = ht.shape[-1]

    # normalize hidden state
    if norm_input:
        normed_ht = activation_fn(normalizer_fn(ht))
        full_inp = tf.concat([normed_ht, inp], axis=-1)
    else:
        full_inp = tf.concat([ht, inp], axis=-1)

    # update gate
    rg = Conv2D(hidden_depth, 3, dilation_rate=dilate, padding='same',
                kernel_regularizer=regularizer,
                kernel_initializer=weights_initializer_mask,
                bias_initializer=biases_initializer_mask)(full_inp)
    rg = mask_normalizer_fn(rg)
    rg = builders.layers.helpers.get_activation('leakyrelu-0.2')(rg)
    rg = normalize_to_01_range(rg)

    # Input Image conv
    img_new = Conv2D(hidden_depth, 3, dilation_rate=dilate, padding='same',
                     bias_initializer=biases_initializer,
                     kernel_regularizer=regularizer,
                     kernel_initializer=weights_initializer)(inp)

    ht_plus = ht + rg * img_new
    ht_new_in = activation_fn(normalizer_fn(ht_plus))

    # new hidden state
    h_new = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                   bias_initializer=biases_initializer,
                   kernel_regularizer=regularizer,
                   kernel_initializer=weights_initializer)(ht_new_in)
    h_new = activation_fn(normalizer_fn(h_new))
    h_new = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                   bias_initializer=biases_initializer,
                   kernel_regularizer=regularizer,
                   kernel_initializer=weights_initializer)(h_new)

    # new hidden state out
    # linear project for filter depth
    if ht.shape[-1] != filter_depth:
        ht_orig = Conv2D(filter_depth, 1, padding='same',
                         bias_initializer=biases_initializer,
                         kernel_regularizer=regularizer,
                         kernel_initializer=weights_initializer)(ht_orig)
    ht_new = ht_orig + h_new
    ht_new = tf.keras.layers.AveragePooling2D((stride, stride))(ht_new)

    return ht_new


def mru_conv_paper_version(inp, ht, filter_depth, sn,
                           stride, dilate=1,
                           activation='relu',
                           normalizer=None,
                           normalizer_params=None,
                           weights_initializer=tf.keras.initializers.GlorotUniform(),
                           biases_initializer_mask=tf.keras.initializers.Constant(0.5),
                           weight_decay_rate=1e-8,
                           norm_mask=False,
                           norm_input=True,
                           labels=None,
                           paper_version=True,
                           w_rep=None):
    Conv2D = tf.keras.layers.Conv2D if not sn else builders.layers.spectral.SNConv2D
    activation_fn = builders.layers.helpers.get_activation(activation) if activation else tf.identity
    normalizer_fn_ = builders.layers.helpers.get_normalization_2d(normalizer, reusable=True) if normalizer else tf.identity
    if normalizer == 'conditionalin':
        normalizer_fn = lambda x: normalizer_fn_([x, labels])
    elif normalizer == 'adain':
        normalizer_fn = lambda x: normalizer_fn_(x, w_rep)
    else:
        normalizer_fn = normalizer_fn_
    mask_normalizer_fn = normalizer_fn if norm_mask else tf.identity

    regularizer = tf.keras.regularizers.L2(weight_decay_rate) if weight_decay_rate > 0 else None
    weights_initializer_mask = weights_initializer
    biases_initializer = tf.zeros_initializer()

    ht_orig = tf.identity(ht)
    hidden_depth = ht.shape[-1]

    # normalize hidden state
    if norm_input:
        normed_ht = activation_fn(normalizer_fn(ht))
        full_inp = tf.concat([normed_ht, inp], axis=-1)
    else:
        full_inp = tf.concat([ht, inp], axis=-1)

    # update gate
    rg = Conv2D(hidden_depth, 3, dilation_rate=dilate, padding='same',
                kernel_regularizer=regularizer,
                kernel_initializer=weights_initializer_mask,
                bias_initializer=biases_initializer_mask)(full_inp)
    rg = mask_normalizer_fn(rg)
    rg = builders.layers.helpers.get_activation('sigmoid')(rg)

    # output gate
    zg = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                kernel_regularizer=regularizer,
                kernel_initializer=weights_initializer_mask,
                bias_initializer=biases_initializer_mask)(full_inp)
    zg = mask_normalizer_fn(zg)
    zg = builders.layers.helpers.get_activation('sigmoid')(zg)

    # new hidden state
    ht_new_in = tf.concat([rg * ht, inp], axis=-1)  # m_i mask in paper
    h_new = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                   bias_initializer=biases_initializer,
                   kernel_regularizer=regularizer,
                   kernel_initializer=weights_initializer)(ht_new_in)
    h_new = activation_fn(normalizer_fn(h_new))
    h_new = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                   bias_initializer=biases_initializer,
                   kernel_regularizer=regularizer,
                   kernel_initializer=weights_initializer)(h_new)
    h_new = activation_fn(normalizer_fn(h_new))

    # new hidden state out
    # linear project for filter depth
    if ht.shape[-1] != filter_depth:
        ht = Conv2D(filter_depth, 1, padding='same',
                    bias_initializer=biases_initializer,
                    kernel_regularizer=regularizer,
                    kernel_initializer=weights_initializer)(ht_orig)
    ht_new = ht * (1 - zg) + h_new * zg
    ht_new = tf.keras.layers.AveragePooling2D((stride, stride))(ht_new)

    return ht_new


def mru_block(x, ht, filter_depth, sn, stride=2, dilate_rate=1,
              num_cells=5, last_unit=False,
              activation='relu',
              normalizer=None,
              weights_initializer=tf.keras.initializers.GlorotUniform(),
              weight_decay_rate=1e-5,
              deconv=False,
              labels=None,
              paper_version=False,
              w_rep=None):
    assert len(ht) == num_cells
    activation_fn = builders.layers.helpers.get_activation(activation) if activation else tf.identity
    normalizer_fn_ = builders.layers.helpers.get_normalization_2d(normalizer, reusable=True) if normalizer else tf.identity
    if normalizer == 'conditionalin':
        normalizer_fn = lambda x: normalizer_fn_([x, labels])
    elif normalizer == 'adain':
        normalizer_fn = lambda x: normalizer_fn_(x, w_rep)
    else:
        normalizer_fn = normalizer_fn_
    if paper_version:
        cell_fun = mru_deconv if deconv else mru_conv_paper_version
    else:
        cell_fun = mru_deconv if deconv else mru_conv

    if dilate_rate != 1:
        stride = 1

    hts_new = []
    inp = x
    ht_new = cell_fun(inp, ht[0], filter_depth, sn=sn, stride=stride,
                      dilate=dilate_rate,
                      activation=activation,
                      normalizer=normalizer,
                      weights_initializer=weights_initializer,
                      weight_decay_rate=weight_decay_rate,
                      labels=labels,
                      paper_version=paper_version,
                      w_rep=w_rep)
    hts_new.append(ht_new)
    inp = ht_new

    for i in range(1, num_cells):
        if deconv:
            ht[i] = tf.keras.layers.UpSampling2D((stride, stride))(ht[i])
        else:
            ht[i] = tf.keras.layers.AveragePooling2D((stride, stride))(ht[i])
        ht_new = cell_fun(inp, ht[i], filter_depth, sn=sn, stride=1,
                          dilate=dilate_rate,
                          activation=activation,
                          normalizer=normalizer,
                          weights_initializer=weights_initializer,
                          weight_decay_rate=weight_decay_rate,
                          labels=labels,
                          paper_version=paper_version)
        hts_new.append(ht_new)
        inp = ht_new

    if last_unit:
        hts_new[-1] = activation_fn(normalizer_fn(hts_new[-1]))

    return hts_new


def normalize_to_01_range(x):
    return (x - tf.reduce_min(x, axis=[1, 2], keepdims=True)) / (
        tf.reduce_max(x, axis=[1, 2], keepdims=True) - tf.reduce_min(x, axis=[1, 2], keepdims=True))


def mru_deconv(inp, ht, filter_depth, sn,
               stride, dilate=1,
               activation='relu',
               normalizer=None,
               normalizer_params=None,
               weights_initializer=tf.keras.initializers.GlorotUniform(),
               biases_initializer_mask=tf.keras.initializers.Constant(0.5),
               weight_decay_rate=1e-8,
               norm_mask=False,
               norm_input=True,
               labels=None,
               paper_version=False,
               w_rep=None):
    Conv2D = tf.keras.layers.Conv2D if not sn else builders.layers.spectral.SNConv2D
    activation_fn = builders.layers.helpers.get_activation(activation) if activation else tf.identity
    normalizer_fn_ = builders.layers.helpers.get_normalization_2d(normalizer, reusable=True) if normalizer else tf.identity
    if normalizer == 'conditionalin':
        normalizer_fn = lambda x: normalizer_fn_([x, labels])
    elif normalizer == 'adain':
        normalizer_fn = lambda x: normalizer_fn_(x, w_rep)
    else:
        normalizer_fn = normalizer_fn_
    mask_normalizer_fn = normalizer_fn if norm_mask else tf.identity

    hidden_depth = ht.shape[-1]
    regularizer = tf.keras.regularizers.L2(weight_decay_rate) if weight_decay_rate > 0 else None
    weights_initializer_mask = weights_initializer
    biases_initializer = tf.zeros_initializer()

    ht = tf.keras.layers.UpSampling2D((stride, stride))(ht)

    # normalize hidden state
    if norm_input:
        normed_ht = activation_fn(normalizer_fn(ht))
        full_inp = tf.concat([normed_ht, inp], axis=-1)
    else:
        full_inp = tf.concat([ht, inp], axis=-1)

    # update gate
    rg = Conv2D(hidden_depth, 3, dilation_rate=dilate, padding='same',
                kernel_regularizer=regularizer,
                kernel_initializer=weights_initializer_mask,
                bias_initializer=biases_initializer_mask)(full_inp)
    rg = mask_normalizer_fn(rg)
    if not paper_version:
        rg = builders.layers.helpers.get_activation('leakyrelu-0.2')(rg)
        rg = normalize_to_01_range(rg)
    else:
        rg = builders.layers.helpers.get_activation('sigmoid')(rg)

    # output gate
    zg = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                kernel_regularizer=regularizer,
                kernel_initializer=weights_initializer_mask,
                bias_initializer=biases_initializer_mask)(full_inp)
    zg = mask_normalizer_fn(zg)
    if not paper_version:
        zg = builders.layers.helpers.get_activation('leakyrelu-0.2')(zg)
        zg = normalize_to_01_range(zg)
    else:
        zg = builders.layers.helpers.get_activation('sigmoid')(zg)

    # new hidden state
    ht_new_in = tf.concat([rg * ht, inp], axis=-1)  # m_i mask in paper
    h_new = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                   bias_initializer=biases_initializer,
                   kernel_regularizer=regularizer,
                   kernel_initializer=weights_initializer)(ht_new_in)
    h_new = activation_fn(normalizer_fn(h_new))
    h_new = Conv2D(filter_depth, 3, dilation_rate=dilate, padding='same',
                   bias_initializer=biases_initializer,
                   kernel_regularizer=regularizer,
                   kernel_initializer=weights_initializer)(h_new)
    h_new = activation_fn(normalizer_fn(h_new))

    # new hidden state out
    # linear project for filter depth
    if ht.shape[-1] != filter_depth:
        ht = Conv2D(filter_depth, 1, padding='same',
                    bias_initializer=biases_initializer,
                    kernel_regularizer=regularizer,
                    kernel_initializer=weights_initializer)(ht)
        ht = activation_fn(normalizer_fn(ht))
    ht_new = ht * (1 - zg) + h_new * zg  # n_i mask in paper

    return ht_new
