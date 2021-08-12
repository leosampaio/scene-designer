import os
import time
import json
import random
from collections import namedtuple

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2
from dataloaders.quickdraw_tfrecord import QuickDrawTFRecord
from dataloaders.coco_crops_tfrecord import COCOCCropsTFRecord


class QuickDrawCOCOCropsTFRecord(BaseDataLoaderV2):
    name = 'quickdraw-cococrops-tf'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            qdraw_dir='/store/lribeiro/datasets/tfrecord_quickdraw',
            crops_dir='/store/lribeiro/datasets/cococrops-tfrecord'
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        # load the sub datasets
        self.qdset = QuickDrawTFRecord(hps)
        self.ccset = COCOCCropsTFRecord(hps)

        self.crop_size = self.ccset.crop_size
        self.sketch_size = self.qdset.sketch_size
        self.n_classes = len(self.ccset.obj_idx_to_ID)

        # prepare the inverted class index (classes that can be negative to a specific class)
        self.valid_crop_negatives = [[] for _ in self.ccset.obj_idx_to_ID]
        for idx_label_a in range(len(self.ccset.obj_idx_to_ID)):
            for idx_label_b in range(len(self.ccset.obj_idx_to_ID)):
                if idx_label_a != idx_label_b and self.ccset.obj_idx_to_name[idx_label_b] in self.ccset.sketchable_objs:
                    self.valid_crop_negatives[idx_label_a].append(self.ccset.obj_idx_to_name[idx_label_b])

    def get_iterator(self, split_name, batch_size, shuffle=None, prefetch=5, repeat=False, image_size=None):
        dataset = self.qdset.splits[split_name].dataset
        if repeat:
            dataset = dataset.repeat()
        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)
        dataset = dataset.map(
            utils.tfrecord.parse_raster_sketch_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        cciterators = self.ccset.get_all_class_iterators(split_name, batch_size=None, shuffle=200, repeat=True)
        include_crop, out_types = self.build_include_crop(cciterators)
        dataset = dataset.map(lambda *x: tf.py_function(func=include_crop, inp=[*x], Tout=out_types),
                              num_parallel_calls=4)
        dataset = dataset.map(self.include_shape_info, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        if image_size is not None:
            self.image_size = image_size
            dataset = dataset.map(self.resize)
        return dataset.prefetch(prefetch)

    def resize(self, sketch, p_crop, n_crop, p_class, n_class):
        if self.image_size != self.sketch_size:
            sketch = tf.image.resize(sketch, (self.image_size, self.image_size))
            p_crop = tf.image.resize(p_crop, (self.image_size, self.image_size))
            n_crop = tf.image.resize(n_crop, (self.image_size, self.image_size))
        return sketch, p_crop, n_crop, p_class, n_class

    def build_include_crop(self, cciterators):
        def include_crop(sketch, label):

            # convert quickdraw label to coco
            coco_class_name = self.qdset.coco_classes[label]
            p_class = self.ccset.obj_ID_to_idx[str(self.ccset.obj_name_to_ID[coco_class_name])]

            # include positive crop
            p_crop, _ = next(cciterators[coco_class_name])

            # include negative crop
            n_class_name = random.choice(self.valid_crop_negatives[p_class])
            n_crop, n_class = next(cciterators[n_class_name])

            return sketch, p_crop, n_crop, p_class, n_class
        out_types = [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64]
        return include_crop, out_types

    def include_shape_info(self, sketch, p_crop, n_crop, p_class, n_class):
        sketch.set_shape([self.sketch_size, self.sketch_size, 1])
        p_crop.set_shape([self.crop_size, self.crop_size, 3])
        n_crop.set_shape([self.crop_size, self.crop_size, 3])
        p_class.set_shape([])
        n_class.set_shape([])
        return sketch, p_crop, n_crop, p_class, n_class


def main():
    qdset = QuickDrawCOCOCropsTFRecord()

    start_time = time.time()
    iterator = qdset.get_iterator('train', 128, shuffle=200, prefetch=10, repeat=True)
    counter = 0
    for sample in iterator:
        counter += 1
        time_until_now = time.time() - start_time
        print("Processed {} batches in {}s ({}s/b)".format(
            counter, time_until_now, time_until_now / counter))
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
