import os
import time
import json
import random
from collections import namedtuple

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2
from dataloaders.quickdraw_tfrecord_indexed import QuickDrawTFRecordIndexed
from dataloaders.coco_crops_tfrecord import COCOCCropsTFRecord


class QDCOCOfgTFRecord(BaseDataLoaderV2):
    name = 'qdcoco-fg-tf'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            coco_dir='/scratch/lribeiro/datasets/qdcoco-fg-03b',
            crops_dir='/store/lribeiro/datasets/cococrops-tfrecord'
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        # read all the metadata
        with open(os.path.join(self.hps['coco_dir'], "meta.json")) as f:
            metadata = json.load(f)
        self.obj_name_to_ID = metadata['obj_name_to_ID']
        self.obj_ID_to_name = metadata['obj_ID_to_name']
        self.obj_idx_to_ID = metadata['obj_idx_to_ID']
        self.obj_ID_to_idx = metadata['obj_ID_to_idx']
        self.obj_idx_to_name = metadata['obj_idx_to_name']
        self.pred_idx_to_name = metadata['pred_idx_to_name']
        self.n_samples = metadata['n_train_samples']
        self.n_test_samples = metadata['n_test_samples']
        self.n_valid_samples = metadata['n_valid_samples']
        self.sketchable_objs = metadata['concrete_objs']
        self.image_size = 256
        self.mask_size = 64
        self.attributes_dim = 35
        self.n_classes = len(self.obj_idx_to_ID)

        # load the secondary datasets
        self.ccset = COCOCCropsTFRecord(hps)
        self.crop_size = self.ccset.crop_size
        self.sketch_size = self.ccset.crop_size

        # prepare the inverted class index (classes that can be negative to a specific class)
        self.valid_crop_negatives = [[] for _ in self.obj_idx_to_ID]
        for idx_label_a in range(len(self.obj_idx_to_ID)):
            for idx_label_b in range(len(self.obj_idx_to_ID)):
                if idx_label_a != idx_label_b and self.obj_idx_to_name[idx_label_b] in self.sketchable_objs:
                    self.valid_crop_negatives[idx_label_a].append(self.obj_idx_to_name[idx_label_b])

        Split = namedtuple('Split', 'files dataset')
        self.splits = {}
        for split in ['train', 'valid', 'test']:
            tfrecords_pattern = os.path.join(self.hps['coco_dir'], "{}*.records".format(split))
            files = tf.io.matching_files(tfrecords_pattern)
            shards = tf.data.Dataset.from_tensor_slices(files)
            dataset = shards.interleave(lambda d: tf.data.TFRecordDataset(d, compression_type='GZIP'))
            self.splits[split] = Split(files, dataset)

    def get_iterator(self, split_name, batch_size, shuffle=None, prefetch=5, repeat=False):
        dataset = self.splits[split_name].dataset
        if repeat:
            dataset = dataset.repeat()

        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.map(
            utils.tfrecord.parse_qdcoco_fg_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        dataset = dataset.padded_batch(batch_size)

        cciterators = self.ccset.get_all_class_iterators(split_name, batch_size=None, shuffle=shuffle, repeat=True)
        mix_batch, out_types = self.build_mix_batch(cciterators)
        dataset = dataset.map(lambda *x: tf.py_function(func=mix_batch, inp=[*x], Tout=out_types))

        dataset = dataset.map(self.include_shape_info, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset.prefetch(prefetch)

    def preprocess_image(self, images):
        return (images - 0.5) * 2

    def deprocess_image(self, images):
        return (images / 2) + 0.5

    def build_mix_batch(self, cciterators):
        def mix_batch(n_objs, n_triples, images, objs,
                      boxes, masks, triples, attributes, ids, sketches):

            boxes = tf.RaggedTensor.from_tensor(boxes, lengths=n_objs)
            masks = tf.RaggedTensor.from_tensor(masks, lengths=n_objs)
            attributes = tf.RaggedTensor.from_tensor(attributes, lengths=n_objs)

            images = self.preprocess_image(images)

            objs_to_img = tf.TensorArray(tf.int32, size=tf.shape(objs)[0])
            for n in tf.range(tf.shape(n_objs)[0]):
                ones = tf.ones_like(objs[n], dtype=tf.int32) * n
                objs_to_img = objs_to_img.write(n, ones)
            objs_to_img = objs_to_img.stack()
            objs_to_img = tf.RaggedTensor.from_tensor(objs_to_img, lengths=n_objs).flat_values

            # we need to shift the triple references so that they remain valid
            # after flattening
            cum_img_obj_len = tf.cast(tf.concat([[0], tf.cumsum(n_objs)[:-1]], axis=0), tf.int32)
            reference_shift = tf.TensorArray(tf.int32, size=tf.shape(n_objs)[0])
            for i in tf.range(tf.shape(n_objs)[0]):
                ones = tf.ones(tf.shape(triples)[1], dtype=tf.int32) * cum_img_obj_len[i]
                reference_shift = reference_shift.write(i, ones)
            reference_shift = reference_shift.stack()
            reference_shift = tf.RaggedTensor.from_tensor(reference_shift, lengths=n_triples).flat_values
            reference_shift = tf.stack((reference_shift, tf.zeros(tf.reduce_sum(n_triples), dtype=tf.int32), reference_shift), axis=-1)

            f_triples = tf.RaggedTensor.from_tensor(triples, lengths=n_triples).flat_values
            f_triples = tf.cast(f_triples, tf.int32) + reference_shift
            f_boxes = boxes.flat_values
            f_attr = attributes.flat_values
            f_masks = masks.flat_values
            f_objs = tf.RaggedTensor.from_tensor(objs, lengths=n_objs).flat_values

            f_objs = tf.reshape(f_objs, [-1, 1])

            # select sketches
            f_sketches = tf.RaggedTensor.from_tensor(sketches, lengths=n_objs).flat_values
            sel_sketches = [None for _ in f_sketches]
            for i, obj_sketches in enumerate(f_sketches):
                sel_sketches[i] = random.choice(obj_sketches)
            sel_sketches = tf.stack(sel_sketches)

            # include negative crops
            neg_crops = [None for _ in f_objs]
            neg_labels = [None for _ in f_objs]
            for i, obj_y in enumerate(f_objs):
                neg_label = random.choice(self.valid_crop_negatives[obj_y[0]])
                neg_crops[i], neg_labels[i] = next(cciterators[neg_label])
            neg_crops = tf.stack(neg_crops)
            neg_labels = tf.stack(neg_labels)
            neg_labels = tf.reshape(neg_labels, [-1, 1])

            return images, f_objs, f_boxes, f_masks, f_triples, f_attr, objs_to_img, sel_sketches, neg_crops, neg_labels, ids
        out_types = [tf.float32, tf.int64, tf.float32, tf.float32, tf.int32, tf.int64, tf.int32, tf.float32, tf.float32, tf.int64, tf.int64]
        return mix_batch, out_types

    def include_shape_info(self, images, f_objs, f_boxes, f_masks, f_triples, f_attr, objs_to_img, sketches, neg_crops, neg_labels, ids):
        # fix size attributes
        w, h = f_boxes[:, 2] - f_boxes[:, 0], f_boxes[:, 3] - f_boxes[:, 1]
        size_attribute = tf.cast(tf.round(9 * (w * h)), tf.int32)
        new_attr = f_attr + tf.one_hot(size_attribute, 35, dtype=tf.int64)

        images.set_shape([None, self.image_size, self.image_size, 3])
        f_objs.set_shape([None, 1])
        f_boxes.set_shape([None, 4])
        f_masks.set_shape([None, self.mask_size, self.mask_size])
        f_triples.set_shape([None, 3])
        new_attr.set_shape([None, self.attributes_dim])
        objs_to_img.set_shape([None, ])
        sketches.set_shape([None, self.sketch_size, self.sketch_size, 1])
        neg_crops.set_shape([None, self.crop_size, self.crop_size, 3])
        neg_labels.set_shape([None, 1])
        ids.set_shape([None, ])
        return images, f_objs, f_boxes, f_masks, f_triples, new_attr, objs_to_img, sketches, neg_crops, neg_labels, ids


def main():
    qdset = QDCOCOfgTFRecord()

    start_time = time.time()
    iterator = qdset.get_iterator('valid', 256, shuffle=50, repeat=True)
    counter = 0
    for sample in iterator:
        counter += 1
        time_until_now = time.time() - start_time
        print("Processed {} batches in {}s ({}s/b)".format(
            counter, time_until_now, time_until_now / counter))
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
