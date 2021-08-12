"""
https://github.com/thisisiron/spectral_normalization-tf2
"""


import tensorflow as tf


class SpectralNormalization(tf.keras.layers.Wrapper):

    def __init__(self, layer, iteration=1, eps=1e-12, training=True, **kwargs):
        self.iteration = iteration
        self.eps = eps
        self.do_power_iteration = training
        if not isinstance(layer, tf.keras.layers.Layer):
            raise ValueError(
                'Please initialize `TimeDistributed` layer with a '
                '`Layer` instance. You passed: {input}'.format(input=layer))
        super(SpectralNormalization, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        def variable_creator(next_creator, **kwargs):
            kwargs['aggregation'] = tf.VariableAggregation.ONLY_FIRST_REPLICA
            kwargs['synchronization'] = tf.VariableSynchronization.ON_READ
            return next_creator(**kwargs)
        with tf.variable_creator_scope(variable_creator):
            self.layer.build(input_shape)

        self.w = self.layer.kernel
        self.w_shape = self.w.shape.as_list()

        self.v = self.add_weight(shape=(1, self.w_shape[0] * self.w_shape[1] * self.w_shape[2]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_v',
                                 dtype=tf.float32,
                                 synchronization=tf.VariableSynchronization.ON_READ,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        self.u = self.add_weight(shape=(1, self.w_shape[-1]),
                                 initializer=tf.initializers.TruncatedNormal(stddev=0.02),
                                 trainable=False,
                                 name='sn_u',
                                 dtype=tf.float32,
                                 synchronization=tf.VariableSynchronization.ON_READ,
                                 aggregation=tf.VariableAggregation.ONLY_FIRST_REPLICA)

        super(SpectralNormalization, self).build()

    def call(self, inputs):
        self.update_weights()
        output = self.layer(inputs)
        self.restore_weights()  # Restore weights because of this formula "W = W - alpha * W_SN`"
        return output

    def update_weights(self):
        w_reshaped = tf.reshape(self.w, [-1, self.w_shape[-1]])

        u_hat = self.u
        v_hat = self.v  # init v vector

        if self.do_power_iteration:
            for _ in range(self.iteration):
                v_ = tf.matmul(u_hat, tf.transpose(w_reshaped))
                v_hat = v_ / (tf.reduce_sum(v_**2)**0.5 + self.eps)

                u_ = tf.matmul(v_hat, w_reshaped)
                u_hat = u_ / (tf.reduce_sum(u_**2)**0.5 + self.eps)

        sigma = tf.matmul(tf.matmul(v_hat, w_reshaped), tf.transpose(u_hat))
        self.u.assign(u_hat)
        self.v.assign(v_hat)

        self.layer.kernel.assign(self.w / sigma)

    def restore_weights(self):
        self.layer.kernel.assign(self.w)


class SNConv2D(tf.keras.layers.Layer):
    """
    Wrapper for Conv2D
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv2DSN = SpectralNormalization(tf.keras.layers.Conv2D(*args, **kwargs))

    def call(self, inputs):
        return self.conv2DSN(inputs)
