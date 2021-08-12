import argparse
import os
import json
import random

import numpy as np
import tensorflow as tf

import utils

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


def load_data_and_save_in_chunks(set_type, coco_classes, coco_to_sketch, class_files, coco_classes_to_idx, n_chunks, datapath):
    for chunk in range(0, n_chunks):

        tf_record_shard_path = os.path.join(datapath, "{}{:03}.records".format(set_type, chunk))

        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tf_record_shard_path, options) as writer:

            # collect a bit of each class
            class_shards = []
            for i, coco_class in enumerate(coco_classes):
                for qd_class in coco_to_sketch[coco_class]:
                    data = np.load(class_files[qd_class],
                                   encoding='latin1', allow_pickle=True, mmap_mode='r')
                    print("Loading chunk {}/{} from class {} ({}) (class {}/{})...".format(
                        chunk + 1, n_chunks, coco_class, qd_class, i + 1, len(coco_classes)))

                    n_samples = len(data[set_type]) // n_chunks
                    start = n_samples * chunk
                    end = start + n_samples
                    samples = data[set_type][start:end]
                    labels = np.ones((n_samples,), dtype=int) * coco_classes_to_idx[coco_class]
                    class_shards.append((samples, labels))

            # shuffle the shards
            random.shuffle(class_shards)

            # take one sample from each shard at a time, mixing all of them
            for cur_index in range(n_samples):
                for samples, labels in class_shards:
                    raster_sketch = preprocess_sketch(samples[cur_index])
                    tf_example = utils.tfrecord.raster_sketch_example(raster_sketch, labels[cur_index])
                    writer.write(tf_example.SerializeToString())
                print("Saved {}/{} samples from each class shard".format(cur_index, n_samples))


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/quickdraw')
    parser.add_argument('--class-relationship', type=str, default='prep_data/quickdraw/quickdraw_to_coco_v2.json')
    parser.add_argument('--n-chunks', type=int, default=50)
    parser.add_argument('--target-dir', default='/store/lribeiro/datasets/tfrecord_quickdraw')

    args = parser.parse_args()
    random.seed(14)

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

    load_data_and_save_in_chunks('valid', coco_classes, coco_to_sketch, class_files, coco_classes_to_idx, 1, args.target_dir)
    load_data_and_save_in_chunks('test', coco_classes, coco_to_sketch, class_files, coco_classes_to_idx, 1, args.target_dir)
    load_data_and_save_in_chunks('train', coco_classes, coco_to_sketch, class_files, coco_classes_to_idx, 5, args.target_dir)


if __name__ == '__main__':
    main()
