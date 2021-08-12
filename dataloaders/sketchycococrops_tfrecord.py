import os
import time
import json
import random
from collections import namedtuple

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2
from dataloaders.coco_crops_tfrecord import COCOCCropsTFRecord


class SketchyCOCOCropsTFRecord(BaseDataLoaderV2):
    name = 'scococrops-tf'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            scococrops_dir='/store/lribeiro/datasets/sketchycococrops-tfrecord',
            crops_dir='/store/lribeiro/datasets/cococrops-tfrecord'
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        with open(os.path.join(self.hps['scococrops_dir'], "meta.json")) as f:
            metadata = json.load(f)
        self.obj_name_to_ID = metadata['obj_name_to_ID']
        self.obj_ID_to_name = metadata['obj_ID_to_name']
        self.obj_idx_to_ID = metadata['obj_idx_to_ID']
        self.obj_ID_to_idx = metadata['obj_ID_to_idx']
        self.obj_idx_to_name = metadata['obj_idx_to_name']
        self.n_samples = metadata['n_train_samples']
        self.n_valid_samples = metadata['n_valid_samples']
        self.sketchable_objs = metadata['concrete_objs']
        self.crop_size = 96
        self.n_classes = len(self.obj_idx_to_ID)
        self.sketch_size = 96

        self.ccset = COCOCCropsTFRecord(hps)

        # prepare the inverted class index (classes that can be negative to a specific class)
        self.valid_crop_negatives = [[] for _ in self.obj_idx_to_ID]
        for idx_label_a in range(len(self.obj_idx_to_ID)):
            for idx_label_b in range(len(self.obj_idx_to_ID)):
                if idx_label_a != idx_label_b and self.obj_idx_to_name[idx_label_b] in self.sketchable_objs:
                    self.valid_crop_negatives[idx_label_a].append(self.obj_idx_to_name[idx_label_b])

        Split = namedtuple('Split', 'files dataset')
        self.splits = {}
        for split in ['train', 'valid', 'test']:
            tfrecords_pattern = os.path.join(self.hps['scococrops_dir'], "{}*.records".format(split))
            files = tf.io.matching_files(tfrecords_pattern)
            shards = tf.data.Dataset.from_tensor_slices(files)
            dataset = shards.interleave(lambda d: tf.data.TFRecordDataset(d, compression_type='GZIP'))
            self.splits[split] = Split(files, dataset)

    def get_iterator(self, split_name, batch_size, shuffle=None, prefetch=10, repeat=False):
        dataset = self.splits[split_name].dataset
        if repeat:
            dataset = dataset.repeat()
        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.map(
            utils.tfrecord.parse_sketchycoco_crop_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        cciterators = self.ccset.get_all_class_iterators(split_name, batch_size=None, shuffle=shuffle, repeat=True)
        include_n_crop, out_types = self.build_include_n_crop(cciterators)
        dataset = dataset.map(lambda *x: tf.py_function(func=include_n_crop, inp=[*x], Tout=out_types),
                              num_parallel_calls=8)
        dataset = dataset.map(self.include_shape_info, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset.prefetch(prefetch)

    def build_include_n_crop(self, cciterators):
        def include_n_crop(p_crop, label, sketch):

            # include negative crop
            n_class_name = random.choice(self.valid_crop_negatives[label])
            n_crop, n_class = next(cciterators[n_class_name])

            return sketch, p_crop, n_crop, label, n_class
        out_types = [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64]
        return include_n_crop, out_types

    def include_shape_info(self, sketch, p_crop, n_crop, p_class, n_class):
        sketch.set_shape([self.sketch_size, self.sketch_size, 1])
        p_crop.set_shape([self.crop_size, self.crop_size, 3])
        n_crop.set_shape([self.crop_size, self.crop_size, 3])
        p_class.set_shape([])
        n_class.set_shape([])
        return sketch, p_crop, n_crop, p_class, n_class

    def preprocess_image(self, images, labels, sketches):
        return (images - 0.5) * 2, labels, sketches

    def deprocess_image(self, images):
        return (images / 2) + 0.5


def main():
    qdset = SketchyCOCOCropsTFRecord()

    start_time = time.time()
    iterator = qdset.get_iterator('train', 512, shuffle=100, repeat=True)
    counter = 0
    for sample in iterator:
        counter += 1
        time_until_now = time.time() - start_time
        print("Processed {} batches in {}s ({}s/b)".format(
            counter, time_until_now, time_until_now / counter))
        time.sleep(0.2)


if __name__ == '__main__':
    main()
