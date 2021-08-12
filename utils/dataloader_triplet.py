#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataloader_triplet.py
Created on Nov 03 2019 14:27

@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from random import shuffle, Random
import tensorflow as tf
# import keras
from .helpers import Timer

MODE = ['class', 'instance', 'super']


def get_data_info(data_path):
    """
    get database info incl. size and num classes
    :param data_path: path to npz or txt or csv
    :return: dict
    """
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        labels = data['labels'][...]
        example_feat = data['feats'][0]
        out = {'size': len(labels),
               'num_classes': len(np.unique(labels)),
               'nsegments': 1,
               'full_paths': [data_path]
               }
    elif data_path.endswith('.csv') or data_path.endswith('.txt'):  # list of datasets
        lst = pd.read_csv(data_path)
        base_dir = os.path.dirname(data_path)
        segment0 = os.path.join(base_dir, lst['path'].tolist()[0])
        data = np.load(segment0)
        example_feat = data['feats'][0]
        full_paths = [os.path.join(base_dir, p) for p in lst['path'].tolist()]
        out = {'size': sum(lst['N'].tolist()),
               'num_classes': int(lst['num_classes'][0]),
               'nsegments': len(lst['path'].tolist()),
               'full_paths': full_paths
               }
    else:
        raise TypeError("Error! dataset not supported.")
    out['dim'] = len(example_feat)
    return out


def get_data_list(data_path):
    if data_path.endswith('.npz'):
        nsamples = len(np.load(data_path)['labels'])
        return [data_path], [0], nsamples
    else:  # csv or txt
        data_dir = os.path.dirname(data_path)
        csv_lst = pd.read_csv(data_path)
        lst = csv_lst['path'].tolist()
        nsamples = sum(csv_lst['N'].tolist())
        return [os.path.join(data_dir, p) for p in lst], list(range(len(lst))), nsamples


def load_all_data(data_path):
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        return data['feats'], data['labels']


class DataLayerTriplet(tf.keras.utils.Sequence):
    """Generates data for Keras
    used for raster and vector features
    keras parallel friendly
    """

    def __init__(self, feats_a, feats_p, labels_a, labels_p, batch_size=100, mode='class',
                 branch='ap', verbose=True):
        """labels_a and labels_p should be the same"""
        self.timer = Timer()
        self.feats_a = feats_a
        self.feats_p = feats_p
        if len(np.shape(labels_a)) == 2:  # double labels
            print('Double label detected. Assume first label is class-level, 2nd is instance-level')
            self.double_labels = True
            self.labels_a = np.array(labels_a)[:, 0]
            self.labels_p = np.array(labels_p)[:, 0]
            self.labels_a_ins = np.array(labels_a)[:, 1]
            self.labels_p_ins = np.array(labels_p)[:, 1]
        else:
            self.double_labels = False
            self.labels_a = labels_a
            self.labels_p = labels_p
        self.shuffle = True  # shuffle data
        self.branch = branch  # whether return data for individual branch or whole triplet
        assert mode in MODE, 'Error - invalid mode {}. Accepted ones are: {}'.format(mode, MODE)
        if mode == 'super':
            assert self.double_labels, ('Error. Super mode only applicable ',
                                        'for dataset with class- and instance-level labels')
        print('Triplet data mode: {} level - {}'.format(mode, self.timer.time(True)))
        self.level = MODE.index(mode)

        self.ncats = len(set(self.labels_a))
        ncats = len(set(self.labels_p))
        assert self.ncats == ncats, \
            'Error - number of classes. Anchor ({}) and pn ({}) mismatch'.format(self.ncats, ncats)
        self.nimgs = len(self.labels_a)
        # =============================================================================
        #     assert self.nimgs == len(self.labels_p),\
        #           'Error - number of samples. Anchor ({}) and pn ({}) mismatch'.format(self.nimgs, len(self.labels_p))
        # =============================================================================
        self.batch_size = batch_size
        self.nbpe = self.nimgs // self.batch_size
        if self.branch == 'ap':
            if verbose:
                print('Start caching label dictionary - %s.' % self.timer.time(True))
            self.labels_dict, self.class_lst = vec2dic(self.labels_a)
            self.labels_dict_p, self.class_lst_p = vec2dic(self.labels_p)
            if self.double_labels:
                self.labels_dict_ins, self.class_lst_ins = vec2dic(self.labels_a_ins)
                self.labels_dict_ins_p, self.class_lst_ins_p = vec2dic(self.labels_p_ins)
        self.indexlist = np.arange(self.nimgs)
        self.on_epoch_end()

        if verbose:
            print('Triplet Data layer:\n Num classes: {}'.format(self.ncats) +
                            '\n Num samples: {}'.format(self.nimgs) + '\n Num batches p epoch: {}'.format(self.nbpe))
            print('P-branch: Num samples{}'.format(len(self.labels_p)))
            print('Dataloader ready - %s' % self.timer.time(True))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.nbpe

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexlist)

    def __getitem__(self, batch_index):
        'Generate one batch of data'
        indexes = self.indexlist[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]
        return self.__load_next_batch(indexes)

    def __load_next_batch(self, list_IDs):
        """load a batch of triplet according to list_IDs of anchor
        return (X_a, X_pn, Y_a, Y_pn) """

        if self.branch=='a':
            Y_a = np.empty(self.batch_size, dtype=int)
            for i, ID in enumerate(list_IDs):
                res = self.__load_next_sample_a(ID)
                if i == 0:
                    X_a = np.zeros((self.batch_size,) + res[0].shape, dtype=np.float32)
                X_a[i, ...] = res[0]
                Y_a[i] = res[1]
            return X_a, Y_a
        elif self.branch == 'p':
            Y_p = np.empty(self.batch_size, dtype=int)
            for i, ID in enumerate(list_IDs):
                res = self.__load_next_sample_p(ID)
                if i == 0:
                    X_p = np.zeros((self.batch_size,) + res[0].shape, dtype=np.float32)
                X_p[i, ...] = res[0]
                Y_p[i] = res[1]
            return X_p, Y_p
        else:
            Y_a = np.empty(self.batch_size, dtype=int)
            Y_n = np.empty_like(Y_a)

            for i, ID in enumerate(list_IDs):
                res = self.__load_next_triplet(ID)
                if i == 0:
                    X_a = np.zeros((self.batch_size,) + res[0].shape, dtype=np.float32)
                    X_p = np.zeros((self.batch_size,) + res[1].shape, dtype=np.float32)
                    X_n = np.zeros_like(X_p)

                X_a[i, ...] = res[0]
                X_p[i, ...] = res[1]
                X_n[i, ...] = res[2]
                Y_a[i] = res[3]
                Y_n[i] = res[4]
            return [X_a, X_p, X_n], [Y_a, Y_a, Y_a, Y_n]  # the first output is dump

    def __load_next_sample_a(self, index):
        return self.feats_a[index], self.labels_a[index]

    def __load_next_sample_p(self, index):
        return self.feats_p[index], self.labels_p[index]

    def __load_next_triplet(self, index):
        """
        Load the next triplet in a batch.
        return (feat_a, feat_p, feat_n, label_a, label_n)
        thread safe
        """
        local_rnd = Random()
        local_rnd.seed()  # for multiprocessing
        x_a = self.feats_a[index]
        y_a = self.labels_a[index]
        y_n = y_a  # initial neg label
        if self.level == 0:  # class level
            # get pos
            index_p = index
            while index_p == index:
                index_p = local_rnd.choice(self.labels_dict_p[y_a])
            # get neg
            while y_n == y_a:
                y_n = local_rnd.choice(self.class_lst)
            index_n = local_rnd.choice(self.labels_dict_p[y_n])
        elif self.level == 1:  # instance level
            if self.double_labels:
                y_a_ins = self.labels_a_ins[index]
                # pos in the same instance group
                index_p = local_rnd.choice(self.labels_dict_ins_p[y_a_ins])
                # neg in the same class group but not instance group
                index_n = index
                while index_n in self.labels_dict_ins_p[y_a_ins]:
                    index_n = local_rnd.choice(self.labels_dict_p[y_a])

            else:
                index_p = index
                index_n = index
                while index_n == index:
                    index_n = local_rnd.choice(self.labels_dict_p[y_n])
        else:  # super instance mode:
            # pos is the same sketch
            index_p = index
            # neg is selected among the instance group
            index_n = index
            y_a_ins = self.labels_a_ins[index]
            while index_n == index:
                index_n = local_rnd.choice(self.labels_dict_ins_p[y_a_ins])

        x_p = self.feats_p[index_p]
        x_n = self.feats_p[index_n]
        return x_a, x_p, x_n, y_a, y_n


def vec2dic(vec):
    """Convert numpy vector to dictionary where elements with same values
    are grouped together.
    e.g. vec = [1 2 1 4 4 4 3] -> output = {'1':[0,2],'2':1,'3':6,'4':[3,4,5]}
    """
    vals = np.unique(vec)
    dic = {}
    for v in vals:
        dic[v] = [i for i in range(len(vec)) if vec[i] == v]
    return dic, vals




# class TripletData(object):
#
#     def __init__(self, vector_path, raster_path, batch_size=100, mode='class', shuffle=True, verbose=True):
#         self.vpaths, vsegids, vN = get_data_list(vector_path)
#         self.rpaths, rsegids, rN = get_data_list(raster_path)
#         self.batch_size = batch_size
#         self.mode = mode
#         assert mode in ['class', 'instance'], 'Error! Mode {} not supported'.format(mode)
#         assert vN == rN, 'Error! raster and vector not have equal len: {} vs {}'.format(rN, vN)
#         assert len(vsegids) == len(rsegids), 'Error! raster and vector not equal num segments.'
#         self.shuffle = shuffle
#         self.nsamples = vN
#         self.batch_num = self.nsamples // self.batch_size
#         self.segids = vsegids
#         self.vprev_seg_path = ''  # cache vector seg_path
#
#     def load(self):
#         while True:
#             if self.shuffle:
#                 random.shuffle(self.segids)
#             for segid in self.segids:
#                 vseg_path = self.vpaths[segid]
#                 rseg_path = self.rpaths[segid]
#                 if vseg_path != self.vprev_seg_path:  # cache current segment path and load new one
#                     self.vprev_seg_path = vseg_path
#                     vdata = np.load(vseg_path)
#                     rdata = np.load(rseg_path)
#                     img_nums = len(vdata['labels'])
#                     id_lst = list(range(img_nums))
#
#                 if self.shuffle:
#                     random.shuffle(id_lst)
#
#                 for batch_id in range(img_nums // self.batch_size):
#                     ids = range(batch_id*self.batch_size, (batch_id+1)*self.batch_size)
#