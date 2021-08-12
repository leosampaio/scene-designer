import tensorflow as tf
import numpy as np


def make_gaussian_kernel(kernel_size, sigma=1):
    size = int(kernel_size) // 2
    x, y = tf.range(-size, size + 1), tf.range(-size, size + 1)
    x, y = tf.meshgrid(x, y)
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = tf.exp(-(tf.cast(x**2 + y**2, tf.float32) / (2.0 * sigma**2))) * normal
    return tf.reshape(g, (kernel_size, kernel_size, 1, 1))


def apply_sobel(img):

    img_sobel_edges = tf.image.sobel_edges(img)
    G = tf.sqrt(img_sobel_edges[..., 0]**2 + img_sobel_edges[..., 1]**2)
    G = G / tf.reduce_max(G)
    phi = tf.atan2(img_sobel_edges[..., 0], img_sobel_edges[..., 1])

    return G, phi


def non_max_suppresion_at_angle(G, filter, mask):
    targetPixels = tf.nn.convolution(G, filter, padding='SAME')
    isGreater = tf.cast(tf.greater(G * mask, targetPixels), tf.float32)
    isMax = isGreater[:, :, :, 0:1] * isGreater[:, :, :, 1:2]
    return isMax


class CannyEdgeDetector(tf.keras.layers.Layer):

    def __init__(self, max_rate=0.4, min_rate=0.1, gauss_size=3, gauss_sigma=1.2):
        super().__init__()

        # define filters for each angle
        np_filter_0 = np.zeros((3, 3, 1, 2))
        np_filter_0[1, 0, 0, 0], np_filter_0[1, 2, 0, 1] = 1, 1  # Left & Right
        filter_0 = tf.constant(np_filter_0, tf.float32)
        np_filter_90 = np.zeros((3, 3, 1, 2))
        np_filter_90[0, 1, 0, 0], np_filter_90[2, 1, 0, 1] = 1, 1  # Top & Bottom
        filter_90 = tf.constant(np_filter_90, tf.float32)
        np_filter_45 = np.zeros((3, 3, 1, 2))
        np_filter_45[0, 2, 0, 0], np_filter_45[2, 0, 0, 1] = 1, 1  # Top-Right & Bottom-Left
        filter_45 = tf.constant(np_filter_45, tf.float32)
        np_filter_135 = np.zeros((3, 3, 1, 2))
        np_filter_135[0, 0, 0, 0], np_filter_135[2, 2, 0, 1] = 1, 1  # Top-Left & Bottom-Right
        filter_135 = tf.constant(np_filter_135, tf.float32)
        self.filters = [filter_0, filter_45, filter_90, filter_135]

        np_filter_sure = np.ones([3, 3, 1, 1])
        np_filter_sure[1, 1, 0, 0] = 0
        self.filter_sure = tf.constant(np_filter_sure, tf.float32)

        self.max_rate = max_rate
        self.min_rate = min_rate
        self.gaussian_kernel = make_gaussian_kernel(gauss_size, gauss_sigma)

    def non_maximum_suppresion(self, G, phi):

        # define the masks for each angle
        phi_degrees = (phi * 180 / np.pi) % 180
        d0 = tf.cast(tf.greater_equal(phi_degrees, 157.5), tf.float32) + tf.cast(tf.less(phi_degrees, 22.5), tf.float32)
        d45 = tf.cast(tf.greater_equal(phi_degrees, 22.5), tf.float32) * tf.cast(tf.less(phi_degrees, 67.5), tf.float32)
        d90 = tf.cast(tf.greater_equal(phi_degrees, 67.5), tf.float32) * tf.cast(tf.less(phi_degrees, 112.5), tf.float32)
        d135 = tf.cast(tf.greater_equal(phi_degrees, 112.5), tf.float32) * tf.cast(tf.less(phi_degrees, 157.5), tf.float32)
        masks = [d0, d45, d90, d135]

        isMax_0 = non_max_suppresion_at_angle(G, self.filters[0], masks[0])
        isMax_45 = non_max_suppresion_at_angle(G, self.filters[1], masks[1])
        isMax_90 = non_max_suppresion_at_angle(G, self.filters[2], masks[2])
        isMax_135 = non_max_suppresion_at_angle(G, self.filters[3], masks[3])

        # merge edges on all directions
        edges_raw = G * (isMax_0 + isMax_90 + isMax_45 + isMax_135)
        edges_raw = tf.clip_by_value(edges_raw, 0., 1.)
        return edges_raw

    def hysteresis_thresholding(self, edges):
        edges_sure = tf.cast(tf.greater_equal(edges, self.max_rate), tf.float32)
        edges_weak = tf.cast(tf.less(edges, self.max_rate), tf.float32) * tf.cast(tf.greater_equal(edges, self.min_rate), tf.float32)

        edges_connected = tf.nn.convolution(edges_sure, self.filter_sure, padding='SAME') * edges_weak
        for _ in range(10):
            edges_connected = tf.nn.convolution(edges_connected, self.filter_sure, padding='SAME') * edges_weak

        edges_final = edges_sure + tf.clip_by_value(edges_connected, 0., 1.)
        return edges_final

    def call(self, imgs, masks):

        masks = tf.expand_dims(masks, axis=-1)
        if tf.shape(masks)[1] != tf.shape(imgs)[1]:
            masks = tf.image.resize(masks, (tf.shape(imgs)[1], tf.shape(imgs)[2]),
                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        gray_imgs = tf.image.rgb_to_grayscale(imgs)
        gray_imgs = tf.multiply(gray_imgs, masks)
        x_blurred = tf.nn.convolution(gray_imgs, self.gaussian_kernel, padding='SAME')

        G, phi = apply_sobel(x_blurred)
        edges_raw = self.non_maximum_suppresion(G, phi)
        edges = self.hysteresis_thresholding(edges_raw)
        return 1. - edges
