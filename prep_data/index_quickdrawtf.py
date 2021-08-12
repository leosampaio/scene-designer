import os
import time
import json
import argparse
from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils
from dataloaders.quickdraw_tfrecord import QuickDrawTFRecord

RASTER_SIZE = 96


def preprocess_sketch(sketch):
    # removes large gaps from the data
    sketch = np.minimum(sketch, 1000)
    sketch = np.maximum(sketch, -1000)

    # get bounds of sketch and use them to normalise
    min_x, max_x, min_y, max_y = utils.sketch.get_bounds(sketch)
    max_dim = max([max_x - min_x, max_y - min_y, 1])
    sketch = sketch.astype(np.float32)
    sketch[:, :2] /= max_dim

    sketch = np.squeeze(sketch)
    sketch = utils.sketch.convert_sketch_from_stroke3_to_image(sketch, RASTER_SIZE)
    return sketch


def index_to_tfrecord(set_type, datapath, coco_classes, coco_to_sketch, class_files, coco_classes_to_idx):

    for coco_class in coco_classes:

        tf_record_shard_path = os.path.join(datapath, "{}-c{:03}.records".format(set_type, coco_classes_to_idx[coco_class]))
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tf_record_shard_path, options) as writer:
            counter = 0
            for qd_class in coco_to_sketch[coco_class]:
                data = np.load(class_files[qd_class],
                               encoding='latin1', allow_pickle=True, mmap_mode='r')
                samples = data[set_type]
                labels = np.ones((len(samples),), dtype=int) * coco_classes_to_idx[coco_class]

                for sample, label in zip(samples, labels):
                    counter += 1
                    raster_sketch = preprocess_sketch(sample)
                    tf_example = utils.tfrecord.raster_sketch_example(raster_sketch, label)
                    writer.write(tf_example.SerializeToString())
                    if counter % 500 == 0:
                        print("Processed {} samples in {} ({}) class' {} set".format(counter, coco_class, qd_class, set_type))


def main():

    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/quickdraw')
    parser.add_argument('--class-relationship', type=str, default='prep_data/quickdraw/quickdraw_to_coco_v2.json')
    parser.add_argument('--target-dir', type=str, default='/store/lribeiro/datasets/tfrecord_quickdraw_indexed')
    args = parser.parse_args()

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    # load the mapping
    with open(args.class_relationship) as clrf:
        sketch_to_coco = json.load(clrf)

    class_files, coco_to_sketch, sketch_to_coco_clean = {}, {}, {}
    for class_name, mapped in sketch_to_coco.items():
        if mapped is not None:
            class_files[class_name] = "{}/{}.npz".format(args.dataset_dir, class_name)
            coco_to_sketch[mapped] = coco_to_sketch.get(mapped, []) + [class_name]
            sketch_to_coco_clean[class_name] = mapped
    coco_classes = list(set(sketch_to_coco_clean.values()))
    coco_classes_to_idx = {c: i for i, c in enumerate(coco_classes)}

    metadata = {"coco_classes": coco_classes, "sketch_to_coco": sketch_to_coco_clean,
                "coco_classes_to_idx": coco_classes_to_idx, "coco_to_sketch": coco_to_sketch}
    with open(os.path.join(args.target_dir, "meta.json"), 'w') as outfile:
        json.dump(metadata, outfile)

    index_to_tfrecord('test', args.target_dir, coco_classes, coco_to_sketch, class_files, coco_classes_to_idx)
    index_to_tfrecord('valid', args.target_dir, coco_classes, coco_to_sketch, class_files, coco_classes_to_idx)
    index_to_tfrecord('train', args.target_dir, coco_classes, coco_to_sketch, class_files, coco_classes_to_idx)


if __name__ == '__main__':
    main()
