#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataloader_tokens.py
Created on Oct 14 2019 12:13
Load sketch tokens
@author: Tu Bui tb0035@surrey.ac.uk
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import pandas as pd
import random
from copy import deepcopy
from .skt_tools import strokes_to_lines, lines_to_strokes, normalise_strokes3


def get_data_info(data_path):
    """
    get database info incl. size and num classes
    :param data_path: path to hdf5 data
    :return: dict
    """
    if data_path.endswith('.npz'):
        data = np.load(data_path)
        labels = data['labels'][...]
        example_img = data['images'][:data['ids'][1]]
        out = {'size': len(labels),
               'num_classes': len(np.unique(labels)),
               'is_segment': False
               }
    elif data_path.endswith('.csv') or data_path.endswith('.txt'):  # list of datasets
        lst = pd.read_csv(data_path)
        base_dir = os.path.dirname(data_path)
        segment0 = os.path.join(base_dir, lst['path'].tolist()[0])
        data = np.load(segment0)
        example_img = data['images'][:data['ids'][1]]
        out = {'size': sum(lst['N'].tolist()),
               'num_classes': int(lst['num_classes'][0]),
               'is_segment': True
               }
    else:
        raise TypeError("Error! dataset not supported.")
    vocab_size = max(example_img) + 1
    out['vocab_size'] = vocab_size
    out['SEP'] = vocab_size - 3
    out['SOS'] = vocab_size - 2
    out['EOS'] = vocab_size - 1
    return out


def load_segment(seg_path):
    seg_data = np.load(seg_path)
    labels = seg_data['labels'][...]
    data = seg_data['images'][...]
    ids = seg_data['ids'][...]
    return data, labels, ids


def pad_seq(seq, max_seq_len=0):
    """
    pad zero or trim seq to fit max_seq_len
    """
    if max_seq_len:
        pad_len = max_seq_len - len(seq)
        if pad_len > 0:
            return np.concatenate([seq, np.zeros(pad_len, dtype=np.int64)])
        elif pad_len < 0:  # chop to fit
            two_last_tokens = seq[-2:]
            out = seq[:max_seq_len]
            out[-2:] = two_last_tokens
            return out.astype(np.int64)
    return seq.astype(np.int64)


def data_loader(data_path, batch_size=100, max_seq_len=0, shuffle=False):
    assert data_path.split('.')[-1] in ['txt', 'npz'], 'Error! data %s not in supported format.' % data_path
    if data_path.endswith('.txt'):  # list of data fragments
        data_dir = os.path.dirname(data_path)
        lst = pd.read_csv(data_path)
        paths = [os.path.join(data_dir, path) for path in lst['path'].tolist()]
        segids = list(range(len(paths)))
    else:  # data in a single fragment
        paths = [data_path]
        segids = [0]

    prev_seg_path = ''
    while True:
        if shuffle:
            random.shuffle(segids)
        for segid in segids:
            seg_path = paths[segid]
            if seg_path != prev_seg_path:
                prev_seg_path = seg_path  # cache current segment path
                data, labels, ids = load_segment(seg_path)
                img_nums = len(labels)
                batch_ids = list(range(img_nums//batch_size))
            if shuffle:
                random.shuffle(batch_ids)
            for batch_id in batch_ids:
                img_ids = np.arange(batch_id*batch_size, (batch_id+1)*batch_size)
                batch_labels = labels[img_ids]
                batch_imgs = []
                for img_id in img_ids:
                    img = data[ids[img_id]:ids[img_id+1]]
                    batch_imgs.append(pad_seq(img, max_seq_len))
                yield np.array(batch_imgs), batch_labels


class DataLoader(object):
    """
    DataLoader as class; easy to manage method/property
    """
    def __init__(self, data_path, batch_size=100, max_seq_len=0, shuffle=False, shuffle_stroke=False,
                 tokenizer=None):
        assert data_path.split('.')[-1] in ['txt', 'csv', 'npz'], 'Error! data %s not in supported format.' % data_path
        if data_path.endswith('.npz'):  # data in a single fragment
            self.paths = [data_path]
            self.segids = [0]
            N = len(np.load(data_path)['labels'])
        else:  # list of data fragments
            data_dir = os.path.dirname(data_path)
            lst = pd.read_csv(data_path)
            self.paths = [os.path.join(data_dir, path) for path in lst['path'].tolist()]
            self.segids = list(range(len(paths)))
            N = sum(lst['N'].tolist())

        self.shuffle = shuffle
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.batch_num = N // self.batch_size

        self.prev_seg_path = ''  # cache seg_path in load()
        self.prev_seg_path_t = ''  # cache seg_path in get_datum()

        self.tokenizer = tokenizer
        self.shuffle_stroke = shuffle_stroke
        if shuffle_stroke:
            assert tokenizer is not None, 'Error! Please provide tokenizer if shuffle_stroke=True'

    def load(self):
        while True:
            if self.shuffle:
                random.shuffle(self.segids)
            for segid in self.segids:
                seg_path = self.paths[segid]
                if seg_path != self.prev_seg_path:
                    self.prev_seg_path = seg_path  # cache current segment path
                    data, labels, ids = load_segment(seg_path)
                    img_nums = len(labels)
                    id_lst = list(range(img_nums))
                if self.shuffle:
                    random.shuffle(id_lst)
                for batch_id in range(img_nums // self.batch_size):
                    img_ids = [id_lst[i] for i in range(batch_id * self.batch_size, (batch_id + 1) * self.batch_size)]
                    batch_labels = labels[img_ids]
                    batch_imgs = []
                    for img_id in img_ids:
                        img = data[ids[img_id]:ids[img_id + 1]]
                        if self.shuffle_stroke:
                            stroke = self.tokenizer.decode(img)
                            lines = strokes_to_lines(stroke, 1.0, True)
                            random.shuffle(lines)
                            stroke_shuffle = lines_to_strokes(lines)
                            stroke_shuffle = normalise_strokes3(stroke_shuffle)
                            img = self.tokenizer.encode(stroke_shuffle)
                        batch_imgs.append(pad_seq(img, self.max_seq_len))
                    yield np.array(batch_imgs), batch_labels

    def get_datum(self, segid, img_id):
        """get specific datum; useful for test/debug"""
        seg_path = self.paths[segid]
        if self.prev_seg_path_t != seg_path:
            self.prev_seg_path_t = seg_path
            self.data, self.labels, self.ids = load_segment(seg_path)  # for reuse
        img = pad_seq(self.data[self.ids[img_id]:self.ids[img_id+1]], self.max_seq_len)
        label = self.labels[img_id].squeeze()
        return img, label

    def get_labels(self, segid):
        data, labels, ids = load_segment(self.paths[segid])
        return np.squeeze(labels)
