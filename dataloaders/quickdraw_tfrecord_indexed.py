import os
import time
import json

import tensorflow as tf

import utils
from core.data import BaseDataLoaderV2


class QuickDrawTFRecordIndexed(BaseDataLoaderV2):
    name = 'indexed-quickdraw'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            qdraw_dir='/store/lribeiro/datasets/tfrecord_quickdraw_indexed',
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        with open(os.path.join(self.hps['qdraw_dir'], "meta.json")) as f:
            metadata = json.load(f)
        self.coco_classes = metadata['coco_classes']
        self.sketch_to_coco = metadata['sketch_to_coco']
        self.coco_classes_to_idx = metadata['coco_classes_to_idx']
        self.sketch_size = 96

        self.splits = {}
        for split in ['train', 'valid', 'test']:
            class_sets = []
            for class_id in range(0, len(self.coco_classes)):
                tfrecords_path = os.path.join(self.hps['qdraw_dir'], "{}-c{:03}.records".format(split, class_id))
                dataset = tf.data.TFRecordDataset(tfrecords_path, compression_type='GZIP')
                class_sets.append(dataset)
            self.splits[split] = class_sets

    def get_iterator(self, split_name, class_id, batch_size, shuffle=None, prefetch=1, repeat=False):
        dataset = self.splits[split_name][class_id]
        if repeat:
            dataset = dataset.repeat()

        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.map(
            utils.tfrecord.parse_raster_sketch_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if batch_size is not None:
            dataset = dataset.batch(batch_size)
        return dataset.prefetch(prefetch)

    def get_all_class_iterators(self, split_name, batch_size, shuffle=None, prefetch=2, repeat=False, make_iter=True):
        iterators = {}
        for class_id in range(0, len(self.coco_classes)):
            iterators[self.coco_classes[class_id]] = iter(self.get_iterator(split_name, class_id, batch_size,
                                                                            repeat=repeat, prefetch=prefetch,
                                                                            shuffle=shuffle))
        return iterators


def main():
    qdset = QuickDrawTFRecordIndexed()

    start_time = time.time()

    counter = 0
    for class_id in range(0, 105):
        iterator = qdset.get_iterator('train', class_id, 256, shuffle=50)
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
