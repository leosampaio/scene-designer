import os
import time
import json
import random
from collections import namedtuple

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2


class SketchyExtendedTFRecord(BaseDataLoaderV2):
    name = 'sketchy-extended-tf'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            # sketchy_dir='/scratch/sketchy-tfrecord',
            sketchy_dir='/store/lribeiro/datasets/sketchy-tfrecord',
            sketchy_imgs_dir='/store/lribeiro/datasets/sketchy-tfrecord'
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        with open(os.path.join(self.hps['sketchy_dir'], "meta.json")) as f:
            metadata = json.load(f)
        self.class_names = metadata['class_names']
        self.name_to_idx = metadata['name_to_idx']
        self.n_classes = metadata['n_classes']
        self.sketch_size, self.crop_size = 128, 128

        # load the paired stuff
        Split = namedtuple('Split', 'files dataset')
        self.sketch_splits = {}
        for split in ['train', 'valid']:
            tfrecords_pattern = os.path.join(self.hps['sketchy_dir'], "paired_{}*.records".format(split))
            files = tf.io.matching_files(tfrecords_pattern)
            shards = tf.data.Dataset.from_tensor_slices(files)
            dataset = shards.interleave(lambda d: tf.data.TFRecordDataset(d, compression_type='GZIP'))
            self.sketch_splits[split] = Split(files, dataset)

        # prepare the inverted class index (classes that can be negative to a specific class)
        self.valid_crop_negatives = [[] for _ in self.class_names]
        for idx_label_a in range(len(self.class_names)):
            for idx_label_b in range(len(self.class_names)):
                if idx_label_a != idx_label_b:
                    self.valid_crop_negatives[idx_label_a].append(self.class_names[idx_label_b])

        # load the images
        self.image_splits = {}
        for split in ['train', 'valid']:
            class_sets = []
            for class_idx in range(len(self.class_names)):
                tfrecords_path = os.path.join(self.hps['sketchy_imgs_dir'], "image_{}_c{:03}.records".format(split, class_idx))
                dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')
                class_sets.append(dataset)
            self.image_splits[split] = class_sets

    def get_iterator(self, split_name, batch_size, shuffle=None, prefetch=5, repeat=False, image_size=None):
        dataset = self.sketch_splits[split_name].dataset
        if repeat:
            dataset = dataset.repeat()
        if shuffle is not None:
            options = tf.data.Options()
            options.experimental_deterministic = False
            dataset.with_options(options)
            dataset = dataset.shuffle(shuffle)
        dataset = dataset.map(
            utils.tfrecord.parse_sketchy_record, num_parallel_calls=tf.data.AUTOTUNE)

        cciterators = self.get_all_image_iterators(split_name, batch_size=None, shuffle=200, repeat=True)
        include_crop, out_types = self.build_include_crop(cciterators)
        dataset = dataset.map(lambda *x: tf.py_function(func=include_crop, inp=[*x], Tout=out_types))
        dataset = dataset.map(self.include_shape_info_and_preprocess_image)
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

    def get_all_image_iterators(self, split_name, batch_size, shuffle=None, prefetch=2, repeat=False):
        iterators = {}
        for class_idx in range(len(self.class_names)):
            iterators[self.class_names[class_idx]] = iter(self.get_image_iterator(split_name, class_idx, batch_size,
                                                                                  repeat=repeat, prefetch=prefetch,
                                                                                  shuffle=shuffle))
        return iterators

    def get_image_iterator(self, split_name, class_idx, batch_size, shuffle=None, prefetch=2, repeat=False):
        dataset = self.image_splits[split_name][class_idx]
        if repeat:
            dataset = dataset.repeat()
        if shuffle is not None:
            options = tf.data.Options()
            options.experimental_deterministic = False
            dataset.with_options(options)
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.map(
            utils.tfrecord.parse_coco_crop_record, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(
            self.preprocess_image)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset.prefetch(prefetch)

    def preprocess_image(self, images, labels):
        return (images - 0.5) * 2, labels

    def deprocess_image(self, images):
        return (images / 2) + 0.5

    def build_include_crop(self, cciterators):
        def include_crop(image, label, sketches):

            # randomly select positive sketch
            p_crop, sketch = image, random.choice(sketches)

            # include negative crop
            n_class_name = random.choice(self.valid_crop_negatives[label])
            n_crop, n_class = next(cciterators[n_class_name])

            return sketch, p_crop, n_crop, label, n_class
        out_types = [tf.float32, tf.float32, tf.float32, tf.int64, tf.int64]
        return include_crop, out_types

    def include_shape_info_and_preprocess_image(self, sketch, p_crop, n_crop, p_class, n_class):
        sketch.set_shape([self.sketch_size, self.sketch_size, 1])
        p_crop.set_shape([self.crop_size, self.crop_size, 3])
        n_crop.set_shape([self.crop_size, self.crop_size, 3])
        p_class.set_shape([])
        n_class.set_shape([])
        return sketch, (p_crop - 0.5) * 2, n_crop, p_class, n_class


def main():
    # utils.gpu.setup_gpu([1])
    qdset = SketchyExtendedTFRecord()

    start_time = time.time()
    iterator = qdset.get_iterator('train', 32, shuffle=200, prefetch=10, repeat=True)
    counter = 0
    for sample in iterator:
        counter += 1
        time_until_now = time.time() - start_time
        print("Processed {} batches in {}s ({}s/b)".format(
            counter, time_until_now, time_until_now / counter))
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
