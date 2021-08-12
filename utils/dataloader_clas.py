#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 10/04/19

@author: Tu Bui tb00083@surrey.ac.uk
"""

import sys, os
import numpy as np
import random
import pandas as pd
from .hdf5_utils import HDF5Read
from .skt_tools import draw_lines, strokes_to_lines, centralise_lines
from tensorflow.python import keras
from PIL import Image


class DataLoaderSeq(keras.utils.Sequence):
    def __init__(self, strokes=None, labels=None, max_stroke_len=None, max_skt_len=None,
                 batch_size=1, scale=1.0, recon=False, training=True):
        # declare all vars
        self.strokes = self.labels = self.ncats = self.index_list = None
        self.max_skt_len = self.max_stroke_len = self.multi_layer = None
        self.nskts = self.len = 0
        self.batch_size = 1
        self.training = training  # shuffle at epoch end, length take ceil instead of floor
        self.shuffle = training
        self.scale = scale
        self.recon = recon  # reconstruction: if true also return input as part of output
        self.m = np.zeros(2, dtype=np.float32)
        self.std = np.eye(2).astype(np.float32) * 2.0

        if strokes is not None:
            self.register_data(strokes, labels, max_stroke_len, max_skt_len, batch_size, scale)

    @staticmethod
    def analyze(strokes, labels, verbose=True):
        out = {'nskts': len(labels)}
        class_ids = list(set(labels))
        out['ncats'] = len(class_ids)
        if out['ncats'] != max(class_ids) + 1 and verbose:
            print('Warning! Provided labels not in range [0,%d]' % out['ncats'])

        sample_per_class = np.zeros(out['ncats'], dtype=np.int64)
        for id_ in range(out['ncats']):
            cid = class_ids[id_]
            sample_per_class[id_] = len(np.where(labels == cid)[0])
        out['sample_per_class'] = sample_per_class

        skt_len = np.zeros(out['nskts'], dtype=np.int64)
        str_len = []
        for id_, stroke in enumerate(strokes):
            str_ends = np.where(stroke[:, 2] == 1)[0]
            skt_len[id_] = len(str_ends)
            str_len.extend(str_ends - np.r_[0, str_ends][:-1])
        str_len = np.array(str_len)
        out['str_len'] = str_len
        out['skt_len'] = skt_len

        if verbose:
            print('Num samples: %d' % out['nskts'])
            print('Num classes: %d' % out['ncats'])
            print('Num samples per class: ave %d, std %f' % (sample_per_class.mean(), sample_per_class.std()))
            print('Stroke len: %d-%d, mean %f, std %f' % (str_len.min(), str_len.max(), str_len.mean(), str_len.std()))
            print('Sketch len: %d-%d, mean %f, std %f' % (skt_len.min(), skt_len.max(), skt_len.mean(), skt_len.std()))
        return out

    @staticmethod
    def pad(skt3, max_str_len, max_skt_len):
        str_ends = np.where(skt3[:, 2] == 1)[0][:-1] + 1
        stroke_lst = np.split(skt3.astype(np.float32), str_ends, axis=0)
        last_point = np.array([0, 0], dtype=np.float32)[None, :]
        leftover = 0  # if a stroke is chopped, the leftover is reserved for the next stroke
        out = []  # np.empty((max_skt_len, max_str_len, 2), dtype=np.float32)
        for id_ in range(min(max_skt_len, len(stroke_lst))):
            stroke_ = stroke_lst[id_]
            if stroke_.size > 0:
                stroke_ = stroke_[:, :2]
                stroke_[0] += leftover
                npoints = len(stroke_)
                if npoints >= max_str_len:
                    leftover = stroke_[max_str_len:].sum(axis=0)
                    stroke_ = stroke_[:max_str_len]
                else:
                    leftover = 0
                    padarr = np.repeat(last_point, max_str_len - npoints, axis=0)
                    stroke_ = np.r_[stroke_, padarr]
                out.append(stroke_)
        out = np.concatenate(out, axis=0)
        assert len(out) <= max_str_len*max_skt_len, 'Error! padded sketch longer than allowed.'
        padarr = np.repeat(last_point, max_str_len*max_skt_len - len(out), axis=0)
        out = np.r_[out, padarr]
        return out

    @staticmethod
    def str2_to_str3(str2, max_str_len, max_skt_len):
        """
        convert stroke2 (x,y) format back to stroke3
        :param str2: stroke2 array (max_str_len x max_skt_len, 2)
        :param max_str_len:
        :param max_skt_len:
        :return: stroke3
        """
        penstate = np.array(([0] * (max_str_len-1) + [1]) * max_skt_len, dtype=np.float32)
        out = np.c_[str2, penstate]
        return out

    @staticmethod
    def augment2(str2, m, std):
        out = np.zeros_like(str2)
        out = np.cumsum(str2, axis=0)
        for i in range(len(str2)):
            if np.abs(str2[i]).sum() > 0:
                # out[i,...] = str2[i] + np.random.multivariate_normal(m, std, 1)  # naive
                out[i, ...] = out[i] + np.random.multivariate_normal(m, std, 1)
        out[1:, :] -= out[:-1, :]
        return out

    @staticmethod
    def augment3(str3, m, std):
        out = DataLoaderSeq.augment2(str3[:, :2], m, std)
        out = np.c_[out, str3[:, 2]]
        return out

    def register_data(self, strokes, labels, max_stroke_len, max_skt_len, batch_size, scale, verbose=True):
        data_info = self.analyze(strokes, labels, verbose)
        self.nskts = data_info['nskts']
        self.ncats = data_info['ncats']
        self.labels = np.array(labels, dtype=np.int64).squeeze()
        self.max_skt_len = data_info['skt_len'].max() if max_skt_len is None else max_skt_len
        self.max_stroke_len = max_stroke_len
        if max_stroke_len is None:
            # keep stroke3 format, truncate or pad data to fit max_skt_len
            self.multi_layer = False
            last_pen = np.array([0, 0, 1])[None, :]
            all_strokes = np.empty((self.nskts, self.max_skt_len, 3), dtype=np.float32)
            for id_, stroke in enumerate(strokes):
                nstr = len(stroke)
                if nstr >= self.max_skt_len:
                    stroke_ = stroke[:self.max_skt_len]
                    stroke_[-1, 2] = 1
                    all_strokes[id_, ...] = np.float32(stroke_)
                else:
                    pad = np.repeat(last_pen, self.max_skt_len - nstr, axis=0)
                    all_strokes[id_, ...] = np.r_[stroke, pad].astype(np.float32)
        else:
            # convert to 2 layer (dx,dy) format
            self.multi_layer = True
            all_strokes = np.empty((self.nskts, self.max_skt_len*self.max_stroke_len, 2), dtype=np.float32)
            for id_, stroke in enumerate(strokes):
                stroke_paded = self.pad(stroke, self.max_stroke_len, self.max_skt_len)
                all_strokes[id_, ...] = stroke_paded
        self.strokes = all_strokes
        self.strokes[:, :, :2] /= scale
        self.index_list = list(range(self.nskts))
        self.set_batch_size(batch_size)
        return data_info

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size
        self.len = int(self.nskts / float(self.batch_size))
        # if self.training:
        #     self.len = np.floor(self.nskts / float(self.batch_size))
        # else:
        #     self.len = np.ceil(self.nskts / float(self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.index_list)

    def __len__(self):
        return self.len

    def get_len(self):
        return self.__len__()

    def __getitem__(self, batch_id):
        """
        gen 1 batch of data given batch_id
        :param batch_id: [0, self.len]
        :return: batch_size x time_steps x feat_len
        """
        ids = self.index_list[batch_id*self.batch_size:
                              min((batch_id+1)*self.batch_size, self.nskts)]
        return self.get_data(ids)

    def getitem(self, batch_id):
        return self.__getitem__(batch_id)

    def get_data(self, ids):
        labels = self.labels[ids].astype(np.float32)
        if self.training and self.multi_layer:  # str2 format
            data = np.array([self.augment2(self.strokes[x], self.m, self.std) for x in ids])
        elif self.training and not self.multi_layer:  # str3 format
            data = np.array([self.augment3(self.strokes[x], self.m, self.std) for x in ids])
        else:  # test
            data = self.strokes[ids]
        if self.recon:
            return data, {'class': labels, 'recon': data}
        else:
            return data, labels


class DataLoaderCNN(keras.utils.Sequence):
    def __init__(self, data_path, batch_size=32, shape=None, color=True, prefn=None, train=True):
        self.data_path = data_path
        data = HDF5Read(data_path)
        self.batch_size = batch_size
        if shape is not None:
            self.set_output_shape(shape)
        self.color = color
        self.N = data.get_size()
        self.prefn = prefn
        if train:
            self.shuffle = True
            self.augment = True
        else:
            self.shuffle = False
            self.augment = False
            # assert self.N % self.batch_size == 0, ("Error! In test mode batchsize (%d)"
            #                                        "must be dividable by dataset size (%d)") % (self.batch_size, self.N)
        self.len = int(self.N/self.batch_size)
        self.index_list = list(range(self.N))
        self.on_epoch_end()

    def set_output_shape(self, shape):
        self.shape = np.array(shape)  # output shape
        max_skt_dim = np.int64(self.shape * 0.9)  # max dimensions of a sketch along x and y
        self.max_offset = self.shape - max_skt_dim - 2  # offset linewidth
        self.half_offset = np.int64(self.max_offset / 2)  # just for convenience
        self.scale = max(max_skt_dim)
        # augmentation settings
        self.min_scale = int(0.9*self.scale)
        self.max_scale = int(1.1*self.scale)
        self.min_rot = -5
        self.max_rot = 5

    def set_prefn(self, prefn):
        self.prefn = prefn

    def get_num_classes(self):
        return len(np.unique(self.data.labels))

    def on_epoch_end(self):
        self.index_list = list(range(self.N))
        if self.shuffle:
            random.shuffle(self.index_list)

    def __len__(self):
        return self.len

    def get_len(self):
        return self.__len__()

    def __getitem__(self, batch_id):
        """
        gen 1 batch of data given batch_id
        :param batch_id: [0, self.len)
        :return: batch_size x shape x channel
        """
        ids = self.index_list[batch_id * self.batch_size:(batch_id + 1) * self.batch_size]
        ids.sort()
        return self.get_data(ids)

    def getitem(self, batch_id):
        """
        public method for __getitem__
        :param batch_id:
        :return:
        """
        return self.__getitem__(batch_id)

    def get_data(self, ids):
        data = HDF5Read(self.data_path)
        strokes3, labels = data.get_data(ids)
        if self.augment:  # augmentation
            lines = [strokes_to_lines(stroke, np.random.randint(self.min_scale, self.max_scale)) for stroke in strokes3]
            lines = [centralise_lines(line, self.shape, True) for line in lines]
            imgs = [draw_lines(line, self.shape, colour=self.color) for line in lines]
            rots = np.random.randint(self.min_rot, self.max_rot + 1, size=self.batch_size)
            imgs = [np.array(Image.fromarray(imgs[i]).rotate(rots[i], Image.BILINEAR,
                                                             expand=False, fillcolor=(255, 255, 255)))
                    for i in range(self.batch_size)]
            flips = np.random.choice(2, self.batch_size) * 2 - 1
            imgs = [imgs[i][:, ::flips[i]] for i in range(self.batch_size)]
        else:
            lines = [strokes_to_lines(stroke, self.scale) for stroke in strokes3]
            lines = [centralise_lines(line, self.shape, False) for line in lines]
            imgs = [draw_lines(line, self.shape, colour=self.color) for line in lines]
        imgs = np.array(imgs, dtype=np.float32)
        if self.prefn is not None:
            imgs = self.prefn(imgs)
        return imgs, labels


def data_loader_cnn(data_path, shape, batch_size=32, color=True, prefn=None, load_all=False, train=True):
    """Equivalent to DataLoaderCNN but threadsafe"""
    out_shape = np.array(shape)
    max_skt_dim = np.int64(out_shape * 0.9)  # max dimensions of a sketch along x and y
    max_offset = out_shape - max_skt_dim - 2  # offset linewidth
    half_offset = np.int64(max_offset / 2)  # just for convenience
    scale = max(max_skt_dim)  # every sketch will be scaled up by this value
    min_scale = int(0.9 * scale)
    max_scale = int(1.1 * scale)
    min_rot = -5
    max_rot = 5
    if train:
        shuffle = True
        augment = True
    else:
        shuffle = False
        augment = False

    data = HDF5Read(data_path, load_all)
    N = data.get_size()
    nsteps = int(N / batch_size)
    batch_list = list(range(nsteps))
    while True:
        if shuffle:
            random.shuffle(batch_list)
        for i in range(nsteps):
            bid = batch_list[i]
            strokes3, labels = data.get_seq_data(bid*batch_size, (bid+1)*batch_size)
            # copy from DataLoaderCNN
            if augment:  # augmentation
                lines = [strokes_to_lines(stroke, np.random.randint(min_scale, max_scale)) for stroke in strokes3]
                lines = [centralise_lines(line, shape, True) for line in lines]
                imgs = [draw_lines(line, shape, colour=color) for line in lines]
                rots = np.random.randint(min_rot, max_rot + 1, size=batch_size)
                imgs = [np.array(Image.fromarray(imgs[i]).rotate(rots[i], Image.BILINEAR,
                                                                 expand=False, fillcolor=(255, 255, 255)))
                        for i in range(batch_size)]
                flips = np.random.choice(2, batch_size) * 2 - 1
                imgs = [imgs[i][:, ::flips[i]] for i in range(batch_size)]
            else:
                lines = [strokes_to_lines(stroke, scale) for stroke in strokes3]
                lines = [centralise_lines(line, shape, False) for line in lines]
                imgs = [draw_lines(line, shape, colour=color) for line in lines]
            imgs = np.array(imgs, dtype=np.float32)
            if prefn is not None:
                imgs = prefn(imgs)
            yield imgs, labels


def data_loader_segments(data_path, shape, batch_size=32, color=True, prefn=None, load_all=True, train=True):
    """
    same as data_loader_cnn, work on dataset comprising multiple files (e.g. quickdraw)
    :param data_path: path to csv/txt storing list of data segments
    :param shape: -
    :param batch_size: -
    :param color: -
    :param prefn: -
    :param load_all: -
    :param train: -
    :return:
    """
    # copy from data_loader_cnn()
    out_shape = np.array(shape)
    max_skt_dim = np.int64(out_shape * 0.9)  # max dimensions of a sketch along x and y
    max_offset = out_shape - max_skt_dim - 2  # offset linewidth
    half_offset = np.int64(max_offset / 2)  # just for convenience
    scale = max(max_skt_dim)  # every sketch will be scaled up by this value
    min_scale = int(0.9 * scale)
    max_scale = int(1.1 * scale)
    min_rot = -5
    max_rot = 5
    if train:
        shuffle = True
        augment = True
    else:
        shuffle = False
        augment = False
    data_dir = os.path.dirname(data_path)
    lst = pd.read_csv(data_path)
    paths = [os.path.join(data_dir, path) for path in lst['path'].tolist()]
    Ns = lst['N'].tolist()
    segids = list(range(len(paths)))
    while True:
        if shuffle:
            random.shuffle(segids)
        for segid in segids:
            data = HDF5Read(paths[segid], load_all)
            N = Ns[segid]
            # copy from data_loader_cnn()
            nsteps = int(N / batch_size)
            batch_list = list(range(nsteps))
            if shuffle:
                random.shuffle(batch_list)
            for i in range(nsteps):
                bid = batch_list[i]
                strokes3, labels = data.get_seq_data(bid * batch_size, (bid + 1) * batch_size)
                # copy from DataLoaderCNN
                if augment:  # augmentation
                    lines = [strokes_to_lines(stroke, np.random.randint(min_scale, max_scale)) for stroke in strokes3]
                    lines = [centralise_lines(line, shape, True) for line in lines]
                    imgs = [draw_lines(line, shape, colour=color) for line in lines]
                    rots = np.random.randint(min_rot, max_rot + 1, size=batch_size)
                    imgs = [np.array(Image.fromarray(imgs[i]).rotate(rots[i], Image.BILINEAR,
                                                                     expand=False, fillcolor=(255, 255, 255)))
                            for i in range(batch_size)]

                    flips = np.random.choice(2, batch_size) * 2 - 1
                    imgs = [imgs[i][:, ::flips[i]] for i in range(batch_size)]
                else:
                    lines = [strokes_to_lines(stroke, scale) for stroke in strokes3]
                    lines = [centralise_lines(line, shape, False) for line in lines]
                    imgs = [draw_lines(line, shape, colour=color) for line in lines]
                imgs = np.array(imgs, dtype=np.float32)
                if prefn is not None:
                    imgs = prefn(imgs)
                yield imgs, labels


def get_data_info(data_path):
    """
    get database info incl. size and num classes
    :param data_path: path to hdf5 data
    :return: dict
    """
    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        data = HDF5Read(data_path)
        out = {'size': data.get_size(),
               'num_classes': len(np.unique(data.labels)),
               'is_segment': False
               }
    elif data_path.endswith('.csv') or data_path.endswith('.txt'):  # list of datasets
        lst = pd.read_csv(data_path)
        out = {'size': sum(lst['N'].tolist()),
               'num_classes': int(lst['num_classes'][0]),
               'is_segment': True
               }
    else:
        raise TypeError("Error! dataset not supported.")
    return out
