import argparse
import os
import json
import random
import glob

import imageio
import skimage.transform as sk_transform
import skimage.morphology as sk_morph
import numpy as np
import tensorflow as tf

import utils

RASTER_SIZE = 96
CROP_SIZE = 96


def preprocess_sketch(sketch_file):
    sketch = imageio.imread(sketch_file) / 255.0
    sketch = sk_transform.resize(sketch, (RASTER_SIZE, RASTER_SIZE))
    if sketch.shape[-1] == 3:
        sketch = sketch[..., 0]
    if len(sketch.shape) == 2:
        sketch = np.reshape(sketch,
                            (RASTER_SIZE, RASTER_SIZE, 1))
    sketch = sketch < 0.1
    sketch = sk_morph.skeletonize(sketch)
    sketch = (sketch < 200).astype(np.float32)
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


def load_data_and_save(set_type, ext_image_class_to_files, edges_dir, metadata, target_dir):

    # save all the sketches
    tf_record_path = os.path.join(target_dir, "paired_{}001.records".format(set_type))
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tf_record_path, options) as writer:

        # collect all class shards and n_samples
        class_shards = []
        for i, (class_name, class_files) in enumerate(ext_image_class_to_files[set_type].items()):
            n_samples = len(class_files)
            labels = np.ones((n_samples,), dtype=int) * metadata['name_to_idx'][class_name]
            class_shards.append((class_files, labels, class_name))

        # take one sample from each shard at a time, mixing all of them
        for cur_index in range(n_samples):
            for class_files, labels, class_name in class_shards:
                try:
                    # load the image
                    image_file = class_files[cur_index]
                    image = preprocess_image(image_file)
                except IndexError:
                    continue

                # and find the corresponding sketches
                sketch_files = glob.glob(os.path.join(edges_dir, class_name, "{}.png".format(os.path.basename(image_file)[:-4])))
                sketches = []
                for sketch_file in sketch_files:
                    raster_sketch = preprocess_sketch(sketch_file)
                    sketches.append(raster_sketch)
                if len(sketches) > 0:
                    tf_example = utils.tfrecord.sketchy_example(image, np.array(sketches), labels[cur_index])
                    writer.write(tf_example.SerializeToString())
            print("Saved {}/{} samples from each class shard".format(cur_index, n_samples))


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/Sketchy')
    parser.add_argument('--edges-dir', default='/scratch/lribeiro/datasets/ext-sketchy-p2s')
    parser.add_argument('--target-dir', default='/scratch/lribeiro/datasets/sketchy-edges-tfrecord')

    args = parser.parse_args()
    random.seed(14)

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    extended_image_dir = os.path.join(args.dataset_dir, 'EXTEND_image_sketchy')  # extended sketchy

    sketchy_classes = [os.path.basename(f) for f in glob.glob(os.path.join(extended_image_dir, '*'))]
    name_to_idx = {c: i for i, c in enumerate(sketchy_classes)}
    all_ext_image_class_to_files = {c: glob.glob(os.path.join(extended_image_dir, c, '*')) for c in sketchy_classes}

    ext_image_class_to_files = {}
    ext_image_class_to_files['valid'] = {c: f[:10] for c, f in all_ext_image_class_to_files.items()}
    ext_image_class_to_files['train'] = {c: f[10:] for c, f in all_ext_image_class_to_files.items()}

    metadata = {"class_names": sketchy_classes, "name_to_idx": name_to_idx,
                "n_classes": len(sketchy_classes)}
    with open(os.path.join(args.target_dir, "meta.json"), 'w') as outfile:
        json.dump(metadata, outfile)

    # load_data_and_save('valid', ext_image_class_to_files, args.edges_dir, metadata, args.target_dir)
    load_data_and_save('train', ext_image_class_to_files, args.edges_dir, metadata, args.target_dir)


if __name__ == '__main__':
    main()
