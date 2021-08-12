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

RASTER_SIZE = 128
CROP_SIZE = 128


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


def load_data_and_save(set_type, image_class_to_files, ext_image_class_to_files, sketch_dir, metadata, target_dir, extended=True):

    # save all the sketches
    tf_record_path = os.path.join(target_dir, "paired_{}001.records".format(set_type))
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tf_record_path, options) as writer:

        # collect all class shards and n_samples
        class_shards = []
        for i, (class_name, class_files) in enumerate(image_class_to_files[set_type].items()):
            n_samples = len(class_files)
            labels = np.ones((n_samples,), dtype=int) * metadata['name_to_idx'][class_name]
            class_shards.append((class_files, labels, class_name))

        # take one sample from each shard at a time, mixing all of them
        for cur_index in range(n_samples):
            for class_files, labels, class_name in class_shards:
                # load the image
                image_file = class_files[cur_index]
                image = preprocess_image(image_file)

                # and find the corresponding sketches
                sketch_files = glob.glob(os.path.join(sketch_dir, class_name, "{}-*.png".format(os.path.basename(image_file)[:-4])))
                sketches = []
                for sketch_file in sketch_files:
                    raster_sketch = preprocess_sketch(sketch_file)
                    sketches.append(raster_sketch)
                if len(sketches) > 0:
                    tf_example = utils.tfrecord.sketchy_example(image, np.array(sketches), labels[cur_index])
                    writer.write(tf_example.SerializeToString())
            print("Saved {}/{} samples from each class shard".format(cur_index, n_samples))

    # save all the images indexed by class
    if extended:
        for i, (class_name, class_files) in enumerate(ext_image_class_to_files[set_type].items()):
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
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/Sketchy')
    parser.add_argument('--target-dir', default='/store/lribeiro/datasets/sketchy-tfrecord')
    parser.add_argument('--filter-data-dir', default='')
    parser.add_argument("--no-extended", action="store_true")

    args = parser.parse_args()
    random.seed(14)

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    sketch_dir = os.path.join(args.dataset_dir, '256x256/sketch/tx_000000000000/')
    image_dir = os.path.join(args.dataset_dir, '256x256/photo/tx_000000000000/')
    extended_image_dir = os.path.join(args.dataset_dir, 'EXTEND_image_sketchy')  # extended sketchy

    if args.filter_data_dir == '':
        sketchy_classes = [os.path.basename(f) for f in glob.glob(os.path.join(image_dir, '*'))]
        name_to_idx = {c: i for i, c in enumerate(sketchy_classes)}
        folder_map = {}
    else:
        with open(os.path.join(args.filter_data_dir, "meta.json")) as f:
            metadata = json.load(f)
            sketchy_classes = metadata['class_names']
            name_to_idx = metadata['name_to_idx']
        folder_map = {'car': 'car_(sedan)'}
    for c_name in sketchy_classes:
        try:
            _ = folder_map[c_name]
        except KeyError:
            folder_map[c_name] = c_name

    all_ext_image_class_to_files = {c: glob.glob(os.path.join(extended_image_dir, folder_map[c], '*')) for c in sketchy_classes}
    all_image_class_to_files = {c: glob.glob(os.path.join(image_dir, folder_map[c], '*')) for c in sketchy_classes}

    image_class_to_files = {}
    image_class_to_files['valid'] = {c: f[:10] for c, f in all_image_class_to_files.items()}
    image_class_to_files['train'] = {c: f[10:] for c, f in all_image_class_to_files.items()}
    ext_image_class_to_files = {}
    ext_image_class_to_files['valid'] = {c: f[:10] for c, f in all_ext_image_class_to_files.items()}
    ext_image_class_to_files['train'] = {c: f[10:] for c, f in all_ext_image_class_to_files.items()}

    metadata = {"class_names": sketchy_classes, "name_to_idx": name_to_idx,
                "n_classes": len(sketchy_classes)}
    with open(os.path.join(args.target_dir, "meta.json"), 'w') as outfile:
        json.dump(metadata, outfile)

    load_data_and_save('valid', image_class_to_files, ext_image_class_to_files, sketch_dir, metadata, args.target_dir, not args.no_extended)
    load_data_and_save('train', image_class_to_files, ext_image_class_to_files, sketch_dir, metadata, args.target_dir, not args.no_extended)


if __name__ == '__main__':
    main()
