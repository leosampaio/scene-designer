import os
import time
import json
from collections import namedtuple

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2


class QuickDrawTFRecord(BaseDataLoaderV2):
    name = 'quickdraw-tfrecord'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            qdraw_dir='/store/lribeiro/datasets/tfrecord_quickdraw',
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        with open(os.path.join(self.hps['qdraw_dir'], "meta.json")) as f:
            metadata = json.load(f)
        self.coco_classes = metadata['coco_classes']
        self.sketch_to_coco = metadata['sketch_to_coco']
        self.coco_classes_to_idx = metadata['coco_classes_to_idx']
        self.n_classes = len(self.coco_classes)
        self.sketch_size = 96

        Split = namedtuple('Split', 'files dataset')
        self.splits = {}
        for split in ['train', 'valid', 'test']:
            tfrecords_pattern = os.path.join(self.hps['qdraw_dir'], "{}*.records".format(split))
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
            utils.tfrecord.parse_raster_sketch_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset.prefetch(prefetch)


def main():
    utils.gpu.setup_gpu([1])
    qdset = QuickDrawTFRecord()

    start_time = time.time()
    iterator = qdset.get_iterator('train', 64, shuffle=100)
    counter = 0
    for sample in iterator:
        counter += 1
        time_until_now = time.time() - start_time
        print("Processed {} batches in {}s ({}s/b)".format(
            counter, time_until_now, time_until_now / counter))
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
