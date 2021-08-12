#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
builders/losses.py
Created on Oct 16 2019 17:34

@author: Tu Bui tb0035@surrey.ac.uk
"""

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa


class LossManager(object):

    def __init__(self):
        self.loss_names = []
        self.loss_fns = {}
        self.loss_weights = {}

    def _add_loss(self, name, weight, func):
        self.loss_names.append(name)
        self.loss_weights[name] = weight
        self.loss_fns[name] = func

    def add_categorical_crossentropy(self, name='class', weight=1.0, from_logits=False):

        def categorical_crossentropy(real, pred, focus=1.):
            return tf.reduce_mean(focus * tf.losses.categorical_crossentropy(real, pred, from_logits=from_logits))

        self._add_loss(
            name, weight,
            categorical_crossentropy)

    def add_focal_class_loss(self, name='focal', weight=1.0, lamb=2):

        def focal_loss(real, pred):
            focus = (1 - tf.reduce_sum(tf.nn.softmax(pred) * tf.squeeze(real), axis=1)) ** lamb
            cross_entropy = tf.losses.categorical_crossentropy(real, pred, from_logits=True)
            return tf.reduce_mean(focus * cross_entropy)
        self._add_loss(name, weight, focal_loss)

    def add_binary_crossentropy(self, name='class', weight=1.0):

        def binary_crossentropy(real, pred):
            return tf.reduce_mean(tf.losses.binary_crossentropy(real, pred))

        self._add_loss(
            name, weight,
            binary_crossentropy)

    def add_reconstruction_loss(self, name='recon', weight=1.0):
        loss_recon_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction='none', name=name + '_loss')

        def loss_function(real, pred):
            real = tf.squeeze(real)
            pred = tf.squeeze(pred)
            mask = tf.math.logical_not(tf.math.equal(real, 0))
            loss_ = loss_recon_object(real, pred)

            mask = tf.cast(mask, dtype=loss_.dtype)
            loss_ *= mask

            return tf.reduce_mean(loss_)

        self._add_loss(name, weight, loss_function)

    def add_hinge_loss(self, target_emb, margin=0.1, name='hinge', weight=1.0):
        """ L = max{0, |pred-real| - margin}"""
        margin = tf.constant(margin, dtype=tf.float32)
        target_emb = tf.constant(target_emb, dtype=tf.float32)

        def hinge_loss_function(real, pred):
            """
            :param real: label tensor (B,1)
            :param pred: predicted target tensor (B, D)
            :return: hinge loss
            """
            real_target = tf.gather_nd(target_emb, real)
            dist = tf.sqrt(tf.reduce_sum(tf.square(real_target - pred), axis=1))
            loss = tf.maximum(0.0, dist - margin)
            return tf.reduce_mean(loss)

        self._add_loss(name, weight, hinge_loss_function)

    def add_mae_loss(self, name, weight=1.):
        self._add_loss(name, weight, lambda a, b: tf.reduce_mean(tf.abs(a - b)))

    def add_mean_loss(self, name, weight=1.):
        self._add_loss(name, weight, tf.reduce_mean)

    def add_mse_loss(self, name, weight=1.):
        self._add_loss(name, weight, lambda a, b: tf.keras.losses.MSE(a, b))

    def add_giou_loss(self, name, weight=1.):
        giou_loss = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self._add_loss(name, weight, lambda a, b: giou_loss(a, b) * 0.1 + tf.keras.losses.MSE(a, b))

    def add_lsgan_fake_loss(self, name, weight=1.,
                            target_value=0., apply_sigmoid=True):

        def lsgan_g(scores):
            scores = tf.reshape(scores, (-1,))
            target = tf.ones_like(scores) * target_value
            scores = tf.sigmoid(scores) if apply_sigmoid else scores
            return tf.keras.losses.MSE(scores, target)

        self._add_loss(name, weight, lsgan_g)

    def add_lsgan_real_loss(self, name, weight=1.,
                            target_value=1., apply_sigmoid=True):
        self.add_lsgan_fake_loss(name, weight=weight,
                                 target_value=target_value,
                                 apply_sigmoid=apply_sigmoid)

    def gan(self, scores, target_value, focus=1.):
        scores = tf.reshape(scores, (-1,))
        target = tf.ones_like(scores) * target_value
        scores = tf.sigmoid(scores)
        return tf.reduce_mean(focus * tf.keras.losses.binary_crossentropy(target, scores, from_logits=False))

    def add_gan_d_loss(self, name, weight=1.):
        def d_loss(real_score, fake_score, focus=1.):
            return 0.5 * (self.gan(real_score, 1.) + self.gan(fake_score, 0., focus))
        self._add_loss(name, weight, d_loss)

    def add_gan_g_loss(self, name, weight=1.):
        def g_loss(fake_score, focus=1.):
            return self.gan(fake_score, 1., focus)
        self._add_loss(name, weight, g_loss)

    def multiscale_lsgan(self, scores, target_value):
        loss = 0.
        for sub_scores in scores:
            if isinstance(sub_scores, list):  # might include other activations, we want the last one
                sub_scores = tf.reshape(sub_scores[-1], (-1,))
            else:  # if it's only the output activations, we use it directly
                sub_scores = tf.reshape(sub_scores, (-1,))
            target = tf.ones_like(sub_scores) * target_value
            loss += tf.keras.losses.MSE(tf.sigmoid(sub_scores), target)
        return loss

    def add_multiscale_lsgan_g_loss(self, name, weight=1.):
        def lsgan_g(scores_fake, focus=1.):
            return self.multiscale_lsgan(scores_fake, 1.)
        self._add_loss(name, weight, lsgan_g)

    def add_multiscale_lsgan_d_loss(self, name, weight=1.):
        def lsgan_g(scores_real, scores_fake, focus=1.):
            return 0.5 * (self.multiscale_lsgan(scores_real, 1.) + self.multiscale_lsgan(scores_fake, 0.))
        self._add_loss(name, weight, lsgan_g)

    # legacy version
    def add_multiscale_lsgan_fake_loss(self, name, weight=1.,
                                       target_value=0., smooth=False):

        def lsgan_g(scores):
            loss = 0.
            for sub_scores in scores:
                if isinstance(sub_scores, list):
                    sub_scores = tf.reshape(sub_scores[-1], (-1,))
                else:
                    sub_scores = tf.reshape(sub_scores, (-1,))
                if smooth:
                    target = tf.random.normal(tf.shape(sub_scores), target_value - 1e-1, 1e-1)
                else:
                    target = tf.ones_like(sub_scores) * target_value
                loss += tf.keras.losses.MSE(tf.sigmoid(sub_scores), target)
            return loss

        self._add_loss(name, weight, lsgan_g)

    # legacy version
    def add_multiscale_lsgan_real_loss(self, name, weight=1.,
                                       target_value=1., smooth=False):
        self.add_multiscale_lsgan_fake_loss(name, weight=weight,
                                            target_value=target_value,
                                            smooth=smooth)

    def add_multiscale_hinge_gan_fake_loss(self, name, weight=1.):

        def hinge_d(scores):
            loss = 0.
            for sub_scores in scores:
                if isinstance(sub_scores, list):
                    nD = len(scores)
                    sub_scores = tf.reshape(sub_scores[-1], (-1,))
                else:
                    nD = 1
                    sub_scores = tf.reshape(sub_scores, (-1,))
                loss += (-tf.reduce_mean(tf.minimum(-sub_scores - 1, 0.0))) / nD
            return loss

        self._add_loss(name, weight, hinge_d)

    def add_multiscale_hinge_gan_real_loss(self, name, weight=1.,
                                           target_value=1.):
        def hinge_d(scores):
            loss = 0.
            for sub_scores in scores:
                if isinstance(sub_scores, list):
                    nD = len(scores)
                    sub_scores = tf.reshape(sub_scores[-1], (-1,))
                else:
                    nD = 1
                    sub_scores = tf.reshape(sub_scores, (-1,))
                loss += (-tf.reduce_mean(tf.minimum(sub_scores - 1, 0.0))) / nD
            return loss

        self._add_loss(name, weight, hinge_d)

    def add_multiscale_hinge_gan_g_loss(self, name, weight=1.,
                                        target_value=1.):
        def hinge_g(scores):
            loss = 0.
            for sub_scores in scores:
                if isinstance(sub_scores, list):
                    nD = len(scores)
                    sub_scores = tf.reshape(sub_scores[-1], (-1,))
                else:
                    nD = 1
                    sub_scores = tf.reshape(sub_scores, (-1,))
                loss += (-tf.reduce_mean(sub_scores)) / nD
            return loss

        self._add_loss(name, weight, hinge_g)

    def add_feature_matching_loss(self, name, weight=1.):

        def feature_matching_loss(feats_real, feats_fake):
            loss = 0.
            per_layer_weight = 4.0 / len(feats_fake[0])
            per_D_weight = 1.0 / len(feats_fake)
            for dis_feats_fake, dis_feats_real in zip(feats_fake, feats_real):
                for feat_vec_fake, feat_vec_real in zip(dis_feats_fake[:-1], dis_feats_real[:-1]):
                    no_grad_real_vec = tf.stop_gradient(feat_vec_real)
                    mae = tf.reduce_mean(tf.abs(feat_vec_fake - no_grad_real_vec))
                    loss += per_layer_weight * per_D_weight * mae
            return loss

        self._add_loss(name, weight, feature_matching_loss)

    def add_vgg19_feature_matching_loss(self, name, weight=1.):

        def vgg_feature_matching_loss(feats_fake, feats_real, focus=1.):
            loss = 0.
            per_layer_weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
            for feat_fake, feat_real, w in zip(feats_fake, feats_real, per_layer_weights):
                mae = tf.reduce_mean(focus * tf.reduce_mean(tf.abs(feat_fake - feat_real), axis=[1, 2, 3]))
                loss += w * mae
            return loss

        self._add_loss(name, weight, vgg_feature_matching_loss)

    def add_style_loss(self, name, weight=1.):

        def gram_matrix(feat):
            channels = feat.shape[-1]
            a = tf.reshape(feat, [-1, feat.shape[1]**2, channels])
            n = feat.shape[1]
            gram = tf.matmul(a, a, transpose_a=True)
            return gram / tf.cast(n, tf.float32)

        def style_loss(feats_fake, feats_real):
            loss = 0.
            per_layer_weights = 1 / len(feats_fake)
            for feat_fake, feat_real in zip(feats_fake, feats_real):
                gram_fake, gram_real = gram_matrix(feat_fake), gram_matrix(feat_real)
                mse = tf.reduce_mean(tf.square(gram_fake - gram_real))
                N, M = feat_fake.shape[1], feat_fake.shape[2]
                loss += per_layer_weights * mse * 1 / (4 * (N**2) * (M**2))
            return loss

        self._add_loss(name, weight, style_loss)

    def add_triplet_loss(self, name, weight=1., margin=.5):
        def t_loss(a, p, n, mask=1):
            d_pos = tf.reduce_sum(tf.square(a - p), 1)
            d_neg = tf.reduce_sum(tf.square(a - n), 1)

            loss = tf.maximum(0., margin + d_pos - d_neg) * mask
            return tf.reduce_mean(loss)

        self._add_loss(name, weight, t_loss)

    def add_contrastive_plus_similarity_matrix_loss(self, name, weight=1., margin=.5, mul=1., abbl_no_shuffled_negative=False):
        def t_loss(a, p, n, sim_mat):
            d_pos = tf.reduce_sum(tf.square(a - p), 1)

            A, B = a, p
            M1, M2, D = tf.shape(A)[0], tf.shape(B)[0], B.shape[1]
            p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), tf.ones(shape=(1, M2)))
            p2 = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]), tf.ones(shape=(M1, 1)), transpose_b=True))
            res = tf.add(p1, p2) - 2     * tf.matmul(A, B, transpose_b=True)

            margin_mat = 1 - sim_mat
            d_neg = tf.maximum(0., (margin_mat * mul) - res)

            if not abbl_no_shuffled_negative:
                A, B = a, n
                M1, M2, D = tf.shape(A)[0], tf.shape(B)[0], B.shape[1]
                p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), tf.ones(shape=(1, M2)))
                p2 = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]), tf.ones(shape=(M1, 1)), transpose_b=True))
                res = tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)

                d_neg2 = tf.reduce_mean(tf.maximum(0., margin - res))
            else:
                d_neg2 = tf.constant(0.)

            loss = tf.reduce_mean(d_pos) + tf.reduce_mean(d_neg) + d_neg2
            return tf.reduce_mean(loss)

        self._add_loss(name, weight, t_loss)

    def add_knowledge_distilation_loss(self, name, weight=1., temperature=1):
        def kd_loss(student_logits, teacher_probs):
            kd_loss = tf.keras.losses.categorical_crossentropy(
                teacher_probs, student_logits / temperature, from_logits=True)
        
            return tf.reduce_mean(kd_loss)
        self._add_loss(name, weight, kd_loss)

    def add_self_contrastive_loss(self, name, weight=1., margin=.5, mul=1.):
        def t_loss(a, p):
            d_pos = tf.reduce_sum(tf.square(a - p), 1)

            A, B = a, p
            M1, M2, D = tf.shape(A)[0], tf.shape(B)[0], B.shape[1]
            p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), tf.ones(shape=(1, M2)))
            p2 = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]), tf.ones(shape=(M1, 1)), transpose_b=True))
            res = tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True)

            margin_mat = 1 - tf.eye(tf.shape(a)[0])
            d_neg = tf.maximum(0., (margin_mat * mul) - res)

            loss = tf.reduce_mean(d_pos) + tf.reduce_mean(d_neg)
            return tf.reduce_mean(loss)

        self._add_loss(name, weight, t_loss)

    def add_two_way_triplet_with_negative_queues_loss(self, name, weight=1., margin=.5):
        def t_loss(ap, bp, an, bn):
            d_pos = tf.reduce_sum(tf.square(ap - bp), 1)

            ap, bp = tf.expand_dims(ap, axis=1), tf.expand_dims(bp, axis=1)
            an, bn = tf.expand_dims(an, axis=0), tf.expand_dims(bn, axis=0)
            d_neg_ab = tf.reduce_sum(tf.reduce_mean(tf.square(ap - bn), axis=1), axis=1)
            d_neg_ba = tf.reduce_sum(tf.reduce_mean(tf.square(bp - an), axis=1), axis=1)

            loss_ab = tf.maximum(0., margin + d_pos - d_neg_ab)
            loss_ba = tf.maximum(0., margin + d_pos - d_neg_ba)

            return loss_ab + loss_ba
        self._add_loss(name, weight, t_loss)

    def add_cosine_distance_word_embedding_loss(self, name, weight=1.):
        def we_loss(x, y, word_embedding):
            embs = tf.gather(word_embedding, y[:, 0])
            embs_norm = tf.cast(tf.math.l2_normalize(embs), tf.float32)
            x_norm = tf.math.l2_normalize(x)

            N, C = tf.shape(x)[0], tf.shape(x)[1]
            cos_sim = tf.matmul(tf.reshape(x_norm, [N, 1, C]), tf.reshape(embs_norm, [N, C, 1]))
            cos_dist = 1. - cos_sim

            return cos_dist

        self._add_loss(name, weight, we_loss)

    def add_cos_triplet_loss(self, name, weight=1., margin=0.5):
        def t_loss(a, p, n, mask=1):
            N = tf.shape(a)[0]  # positive size
            C = tf.shape(n)[1]  # dims
            d_pos = tf.matmul(tf.reshape(a, [N, 1, C]), tf.reshape(p, [N, C, 1]))
            d_pos = 1. - d_pos
            d_neg = tf.matmul(tf.reshape(a, [N, 1, C]), tf.reshape(n, [N, C, 1]))
            d_neg = 1 - d_neg

            loss = tf.maximum(0., margin + d_pos - d_neg) * mask
            return tf.reduce_mean(loss)

        self._add_loss(name, weight, t_loss)

    def add_moco_loss(self, name, weight=1., temperature=0.07):
        def m_loss(ap, bp, bn):
            N = tf.shape(ap)[0]  # positive size
            K = tf.shape(bn)[0]  # queue size
            C = tf.shape(bn)[1]  # dims

            l_pos_ab = tf.matmul(tf.reshape(ap, [N, 1, C]), tf.reshape(bp, [N, C, 1]))
            l_pos_ab = tf.reshape(l_pos_ab, [N, 1])
            l_neg_ab = tf.matmul(tf.reshape(ap, [N, C]), tf.reshape(bn, [C, K]))
            logits_ab = tf.concat([l_pos_ab, l_neg_ab], axis=1)
            labels = tf.zeros([N], dtype="int64")
            loss_ab = tf.losses.sparse_categorical_crossentropy(labels, logits_ab / temperature)
            return tf.reduce_mean(loss_ab)
        self._add_loss(name, weight, m_loss)

    def add_wgan_sn_g_loss(self, name, weight=1.):
        def wgan_sn_g_loss(logits, focus=1.):
            return tf.reduce_mean(tf.nn.softplus(-logits))
        self._add_loss(name, weight, wgan_sn_g_loss)

    def add_wgan_sn_d_loss(self, name, weight=1.):
        def wgan_sn_d_loss(real_logits, fake_logits, focus=1.):
            return tf.reduce_mean(tf.nn.softplus(-real_logits)) + tf.reduce_mean(tf.nn.softplus(fake_logits))
        self._add_loss(name, weight, wgan_sn_d_loss)

    def add_travelgan_loss(self, name, weight=1.):
        def travel_loss(x_a, x_b):
            # compute distance from all to all
            d_a = x_a[:, None] - x_a[None, :]
            d_b = x_b[:, None] - x_b[None, :]
            return tf.reduce_mean(tf.keras.losses.cosine_similarity(d_a, d_b))
        self._add_loss(name, weight, travel_loss)

    def compute_all_loss(self, rp_dict):
        losses = {}
        for lid, name in enumerate(self.loss_names):
            losses[name] = self.loss_weights[name] * self.loss_fns[name](*rp_dict[name])
        return losses

    def compute_loss(self, name, *args):
        assert name in self.loss_names, "Error! Loss name {} not found".format(name)
        return self.loss_weights[name] * self.loss_fns[name](*args)
