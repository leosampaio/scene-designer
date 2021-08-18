#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 12:37:36 2018

@author: Tu Bui tb00083@surrey.ac.uk
"""
import numpy as np
# from scipy.spatial.distance import cdist
from concurrent import futures
import multiprocessing
import tensorflow as tf


def simple_sbir(queries, dataset, return_mat=False, return_recall=False, labels=None, return_distances=False):
    A, B = queries, dataset
    if labels is None:
        source_labels, query_labels = tf.range(tf.shape(B)[0]), tf.range(tf.shape(A)[0])
    else:
        query_labels, source_labels = labels
    M1, M2, D = A.shape[0], B.shape[0], B.shape[1]

    p1 = tf.matmul(tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1), tf.ones(shape=(1, M2)))
    p2 = tf.transpose(tf.matmul(tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]), tf.ones(shape=(M1, 1)), transpose_b=True))
    res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))
    if return_distances:
        return res

    ids = tf.argsort(res)

    rel = tf.equal(tf.gather(source_labels, ids), tf.expand_dims(query_labels, 1))
    rel = tf.cast(rel, tf.float32)

    precision = tf.cumsum(rel, axis=1) / tf.range(1, rel.shape[1] + 1, dtype=tf.float32)[None, ...]
    av_precision = tf.reduce_sum(precision * rel, axis=1) / (tf.reduce_sum(rel, axis=1) + 1e-10)
    mAP = tf.reduce_mean(av_precision)
    ave_precision = tf.reduce_mean(precision, axis=0)
    recall = tf.cumsum(rel, axis=1) / tf.reduce_sum(rel, axis=1, keepdims=True)
    ave_recall = tf.reduce_mean(recall, axis=0)
    if return_mat:
        return mAP, ave_recall[0], ave_recall[4], ave_recall[9], ids
    if return_recall:
        return mAP, ave_recall[0], ave_recall[4], ave_recall[9], ave_recall
    else:
        return mAP, ave_recall[0], ave_recall[4], ave_recall[9]

class sbir(object):
    """class for image retrieval
    """

    def __init__(self, query_feats, query_labels, src_feats, src_labels):
        self.query_feats = query_feats
        self.query_labels = query_labels
        self.src_feats = src_feats
        self.src_labels = src_labels

    def reg(self, query_feats, query_labels, src_feats, src_labels):
        self.query_feats = query_feats + np.zeros((1, 1), dtype=np.float32) if query_feats is not None else None
        self.query_labels = np.array(query_labels)[:, None] if query_labels is not None else None
        self.src_feats = src_feats + np.zeros((1, 1), dtype=np.float32) if src_feats is not None else None
        self.src_labels = np.array(src_labels).squeeze()

    def reg_labels(self, query_labels, src_labels):
        """useful when changing labels e.g. class -> instance"""
        self.query_labels = np.array(query_labels)[:, None] if query_labels is not None else None
        self.src_labels = np.array(src_labels).squeeze()

    def dist_L2(self, a_query):
        """
        Eucludean distance between a (single) query & all features in database
        used in pdist
        """
        return np.sum((a_query - self.src_feats)**2, axis=1)

    def retrieve(self, batch_size=345, debug=False, return_matrix=False):
        """
        Compute distance (L2) between queries and data features
        query_feat: NxD numpy array with N features of D dimensions
        """
        if debug:
            tf.config.experimental_run_functions_eagerly(True)
        train_step_signature = [
            tf.TensorSpec(shape=(None, self.src_feats.shape[-1]), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        ]
        source_feats = tf.constant(self.src_feats)
        source_labels = tf.constant(self.src_labels, dtype=tf.int32)

        @tf.function(input_signature=train_step_signature)
        def gpu_pdist(queries, labels):

            A, B = queries, source_feats
            M1, M2 = tf.shape(queries)[0], tf.shape(self.src_feats)[0]

            p1 = tf.matmul(
                tf.expand_dims(tf.reduce_sum(tf.square(A), 1), 1),
                tf.ones(shape=(1, M2))
            )
            p2 = tf.transpose(tf.matmul(
                tf.reshape(tf.reduce_sum(tf.square(B), 1), shape=[-1, 1]),
                tf.ones(shape=(M1, 1)),
                transpose_b=True
            ))
            res = tf.sqrt(tf.add(p1, p2) - 2 * tf.matmul(A, B, transpose_b=True))

            ids = tf.argsort(res)
            rel = tf.equal(tf.gather(source_labels, ids), labels)
            rel = tf.cast(rel, tf.float32)

            precision = tf.cumsum(rel, axis=1) / tf.range(1.0, 1.0 + tf.cast(tf.shape(rel)[1], tf.float32), dtype=tf.float32)[None, ...]
            av_precision = tf.reduce_sum(precision * rel, axis=1) / (tf.reduce_sum(rel, axis=1) + 1e-10)
            mAP = tf.reduce_mean(av_precision)
            ave_precision = tf.reduce_mean(precision, axis=0)
            recall = tf.cumsum(rel, axis=1) / tf.reduce_sum(rel, axis=1, keepdims=True)
            ave_recall = tf.reduce_mean(recall, axis=0)

            if return_matrix:
                return mAP, ave_precision, ave_recall, ids
            else:
                return mAP, ave_precision, ave_recall

        relevance_matrix, mAPs, ave_precision, ave_recall = [], [], [], []
        print("[SBIR] Computing cdist matrix...")
        for counter, i in enumerate(range(0, len(self.query_feats), batch_size)):
            end_idx = i + batch_size if i + batch_size < len(self.query_feats) else len(self.query_feats)
            batch_q = self.query_feats[i:end_idx]
            batch_y = self.query_labels[i:end_idx]
            if return_matrix:
                mAP, ap, ar, ids = gpu_pdist(batch_q, batch_y)
            else:
                mAP, ap, ar = gpu_pdist(batch_q, batch_y)
            relevance_matrix.append(ids)
            ave_precision.append(ap)
            ave_recall.append(ar)
            mAPs.append(mAP)
            if counter % 20 == 0:
                print("[SBIR] Computing cdist matrix... {}/{}".format(i, len(self.query_feats)))
        relevance_matrix = np.concatenate(relevance_matrix, axis=0)
        ave_precision = tf.reduce_mean(ave_precision, axis=0)
        ave_recall = tf.reduce_mean(ave_recall, axis=0)
        mAP = tf.reduce_mean(mAPs)

        return relevance_matrix, mAP, ave_precision, ave_recall

    def retrieve2file(self, queries, out_file, num=0):
        """perform retrieve but write output to a file
        num specifies number of returned images, if num==0, all images are returned
        """
        res = self.retrieve(queries)
        if num > 0 and num < self.data['feats'].shape[0]:
            res = res[:, 0:num]
        if out_file.endswith('.npz'):
            np.savez(out_file, results=res)
        return 0

    def compute_mAP(self, batch_size=345, gpu_id=0, debug=False):
        relevance_matrix, mAP, P_ave, R_ave = self.retrieve(
            batch_size, gpu_id, debug)
        return mAP, P_ave, R_ave
