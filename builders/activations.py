import tensorflow as tf


def multiply_gradients(alpha):
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad
    return _lr_mult


def miu_relu(x, miu=0.7):
    return (x + tf.sqrt((1 - miu) ** 2 + x ** 2)) / 2.


def prelu(x):
    leak = tf.Variable(0.2, aggregation=tf.VariableAggregation.SUM)
    return tf.maximum(leak * x, x)