import os
import time
import json

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2


class COCOCCropsTFRecord(BaseDataLoaderV2):
    name = 'cococrops-tfrecord'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            crops_dir='/store/lribeiro/datasets/cococrops-tfrecord',
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        with open(os.path.join(self.hps['crops_dir'], "meta.json")) as f:
            metadata = json.load(f)
        self.obj_name_to_ID = metadata['obj_name_to_ID']
        self.obj_ID_to_name = metadata['obj_ID_to_name']
        self.obj_idx_to_ID = metadata['obj_idx_to_ID']
        self.obj_ID_to_idx = metadata['obj_ID_to_idx']
        self.obj_idx_to_name = metadata['obj_idx_to_name']
        self.n_samples = metadata['n_train_samples']
        self.n_test_samples = metadata['n_test_samples']
        self.n_valid_samples = metadata['n_valid_samples']
        self.sketchable_objs = metadata['concrete_objs']
        self.crop_size = 96
        self.n_classes = len(self.obj_idx_to_ID)

        self.splits = {}
        for split in ['train', 'valid', 'test']:
            class_sets = []
            for class_idx in range(len(self.obj_idx_to_ID)):
                tfrecords_path = os.path.join(self.hps['crops_dir'], "{}-c{:03}.records".format(split, class_idx))
                dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')
                class_sets.append(dataset)
            self.splits[split] = class_sets

    def preprocess_image(self, images, labels):
        return (images - 0.5) * 2, labels

    def deprocess_image(self, images):
        return (images / 2) + 0.5

    def get_iterator(self, split_name, class_idx, batch_size, shuffle=None, prefetch=1, repeat=False):
        dataset = self.splits[split_name][class_idx]
        if repeat:
            dataset = dataset.repeat()
        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.map(
            utils.tfrecord.parse_coco_crop_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(
            self.preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset.prefetch(prefetch)

    def get_all_class_iterators(self, split_name, batch_size, shuffle=None, prefetch=2, repeat=False, make_iter=True):
        iterators = {}
        for class_idx in range(len(self.obj_idx_to_ID)):
            iterators[self.obj_idx_to_name[class_idx]] = iter(self.get_iterator(split_name, class_idx, batch_size,
                                                                                repeat=repeat, prefetch=prefetch,
                                                                                shuffle=shuffle))
        return iterators


def main():
    ccset = COCOCCropsTFRecord()

    start_time = time.time()

    counter = 0
    for class_id in range(0, len(ccset.obj_idx_to_ID)):
        iterator = ccset.get_iterator('test', class_id, 32, shuffle=50)
        for sample in iterator:
            counter += 1
            time_until_now = time.time() - start_time
            print("Processed {} batches in {}s ({}s/b)".format(
                counter, time_until_now, time_until_now / counter))
            # time.sleep(0.1)
            if counter % 100 == 0:
                print("Change class!")
                break


if __name__ == '__main__':
    main()
