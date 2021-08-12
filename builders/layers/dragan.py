import tensorflow as tf


def perturb_image(images):
    alpha = tf.random.uniform(shape=(tf.shape(images)[0], 1, 1, 3), dtype=tf.float32)
    perturb_images = images + 0.5 * tf.sqrt(tf.nn.moments(images, axes=[0, 1, 2, 3])[1]) * tf.random.uniform(shape=tf.shape(images))
    diff = perturb_images - images
    interp = images + (alpha * diff)
    return interp


def compute_penalty(perturbed_score, perturbed_image):
    if isinstance(perturbed_score, list):  # if it's a multiscape discriminator
        penalties = []
        for score in perturbed_score:
            penalties.append(_compute_penalty(score[-1], perturbed_image))
        return tf.reduce_mean(penalties)
    else:  # normal GANs with only one score
        return _compute_penalty(perturbed_score, perturbed_image)


def _compute_penalty(perturbed_score, perturbed_image):
    gradients = tf.gradients(perturbed_score, [perturbed_image])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
    gradient_penalty = tf.reduce_mean(tf.maximum(0., slopes - 1.) ** 2)
    return gradient_penalty
