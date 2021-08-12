import argparse
import os
import json
import random
import glob

import imageio
import skimage.transform as sk_transform
import numpy as np
import tensorflow as tf

import utils

RASTER_SIZE = 96
CROP_SIZE = 96


def preprocess_sketch(sketch_file):
    sketch = imageio.imread(sketch_file)
    sketch = sk_transform.resize(sketch, (RASTER_SIZE, RASTER_SIZE))
    if sketch.shape[-1] == 3:
        sketch = sketch[..., 0]
    if len(sketch.shape) == 2:
        sketch = np.reshape(sketch,
                            (RASTER_SIZE, RASTER_SIZE, 1))
    return sketch


def preprocess_image(image_file):
    image = imageio.imread(image_file)
    image = sk_transform.resize(image, (CROP_SIZE, CROP_SIZE))
    if len(image.shape) == 2:
        image = np.reshape(image,
                           (CROP_SIZE, CROP_SIZE, 1))
    if image.shape[-1] == 1:
        image = np.concatenate([image, image, image], -1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    return image


def load_data_and_save(set_type, image_class_to_files, sketch_class_to_files, metadata, target_dir):

    # save all the sketches
    tf_record_path = os.path.join(target_dir, "sketch_{}001.records".format(set_type))
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tf_record_path, options) as writer:

        # collect all class shards and n_samples
        class_shards = []
        for i, (class_name, class_files) in enumerate(sketch_class_to_files[set_type].items()):
            n_samples = len(class_files)
            labels = np.ones((n_samples,), dtype=int) * metadata['name_to_idx'][class_name]
            class_shards.append((class_files, labels))

        # take one sample from each shard at a time, mixing all of them
        for cur_index in range(n_samples):
            for class_files, labels in class_shards:
                raster_sketch = preprocess_sketch(class_files[cur_index])
                tf_example = utils.tfrecord.raster_sketch_example(raster_sketch, labels[cur_index])
                writer.write(tf_example.SerializeToString())
            print("Saved {}/{} samples from each class shard".format(cur_index, n_samples))

    # save all the images indexed by class
    for i, (class_name, class_files) in enumerate(image_class_to_files[set_type].items()):
        tf_record_shard_path = os.path.join(target_dir, "image_{}_c{:03}.records".format(set_type, metadata['name_to_idx'][class_name]))
        options = tf.io.TFRecordOptions(compression_type="GZIP")
        with tf.io.TFRecordWriter(tf_record_shard_path, options) as writer:
            counter = 0
            for sample in class_files:
                if not sample.endswith('jpg'):
                    continue
                counter += 1
                image = preprocess_image(sample)
                tf_example = utils.tfrecord.coco_crop_example(image, metadata['name_to_idx'][class_name])
                writer.write(tf_example.SerializeToString())
                if counter % 100 == 0:
                    print("Processed {} samples in {} class' {} set".format(counter, class_name, set_type))


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/quickdraw_extended')
    parser.add_argument('--target-dir', default='/store/lribeiro/datasets/tfrecord_quickdraw_extended')

    args = parser.parse_args()
    random.seed(14)

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    sketch_dir = os.path.join(args.dataset_dir, 'QuickDraw_sketches_final')
    image_dir = os.path.join(args.dataset_dir, 'QuickDraw_images_final')

    qdextended_classes = [os.path.basename(f) for f in glob.glob(os.path.join(sketch_dir, '*'))]
    name_to_idx = {c: i for i, c in enumerate(qdextended_classes)}
    all_sketch_class_to_files = {c: glob.glob(os.path.join(sketch_dir, c, '*')) for c in qdextended_classes}
    all_image_class_to_files = {c: glob.glob(os.path.join(image_dir, c, '*')) for c in qdextended_classes}

    sketch_class_to_files, image_class_to_files = {}, {}
    sketch_class_to_files['valid'] = {c: f[:100] for c, f in all_sketch_class_to_files.items()}
    image_class_to_files['valid'] = {c: f[:100] for c, f in all_image_class_to_files.items()}
    sketch_class_to_files['train'] = {c: f[100:] for c, f in all_sketch_class_to_files.items()}
    image_class_to_files['train'] = {c: f[100:] for c, f in all_image_class_to_files.items()}

    metadata = {"class_names": qdextended_classes, "name_to_idx": name_to_idx,
                "n_classes": len(qdextended_classes)}
    with open(os.path.join(args.target_dir, "meta.json"), 'w') as outfile:
        json.dump(metadata, outfile)

    load_data_and_save('valid', image_class_to_files, sketch_class_to_files, metadata, args.target_dir)
    load_data_and_save('train', image_class_to_files, sketch_class_to_files, metadata, args.target_dir)


if __name__ == '__main__':
    main()
