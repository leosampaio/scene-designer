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

import models
import dataloaders

# load up a coco dataset
# get indexed quickdraw iterators
# load up the model we are going to use
# create new dataset
# for each sample, for each object, get 256 sketches
# extract features, do sbir and get best 5 and save with dataset


def load_data_and_save(set_type, n_chunks, model, cocoset, target_dir, masked=False):
    if set_type == 'train':
        n_samples = cocoset.n_samples
    elif set_type == 'test':
        n_samples = cocoset.n_test_samples
    elif set_type == 'valid':
        n_samples = cocoset.n_valid_samples

    chunk_size = n_samples // n_chunks

    coco_iter = cocoset.get_simple_iterator(set_type, shuffle=None, prefetch=5, repeat=False)
    sketch_iters = cocoset.qdset.get_all_class_iterators(set_type, batch_size=1024, shuffle=None, repeat=True)
    image_counter, chunk_counter = 0, 0
    for batch in coco_iter:
        n_objs, n_triples, image, objs, boxes, masks, triples, attributes, ids = batch

        if image_counter % chunk_size == 0:
            tf_record_shard_path = os.path.join(target_dir, "{}{:03}.records".format(set_type, chunk_counter))
            options = tf.io.TFRecordOptions(compression_type="GZIP")
            writer = tf.io.TFRecordWriter(tf_record_shard_path, options)
            chunk_counter += 1

        sketches = get_sketches_for_sample(model, cocoset, sketch_iters, image, objs, boxes, masks, masked=masked)
        tf_example = utils.tfrecord.qdcoco_fg_example(image.numpy(), objs.numpy(),
                                                      boxes.numpy(), masks.numpy(), triples.numpy(),
                                                      attributes.numpy(), ids.numpy(), sketches)
        writer.write(tf_example.SerializeToString())

        image_counter += 1
        if image_counter % 5 == 0:
            print("Processed {}/{} images from set {}".format(image_counter, n_samples, set_type))


def get_sketches_for_sample(model, cocoset, sketch_iters, image, objs, boxes, masks, masked=False):
    crops = utils.bbox.crop_bbox_batch([image], boxes, np.zeros(len(boxes)), cocoset.crop_size)
    if masked:
        c_masks = tf.cast(tf.expand_dims(masks, axis=-1), tf.float32)
        c_masks = tf.image.resize(c_masks, (tf.shape(crops)[1], tf.shape(crops)[2]),
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        crops = tf.multiply(crops, c_masks)
    sel_sketches = []
    obj_reps = model.common_project_and_norm(model.obj_encoder.encoder(crops))
    for obj, obj_rep in zip(objs, obj_reps):
        sketches, _ = next(sketch_iters[cocoset.obj_idx_to_name[obj]])
        skt_features = model.common_project_and_norm(model.skt_encoder.encoder(sketches))
        sel_sketches.append(collect_sketches_and_extract_data(obj_rep, sketches, skt_features))
    return tf.stack(sel_sketches).numpy()


def collect_sketches_and_extract_data(obj_crop, sketches, skt_features):
    _, _, _, _, rank_mat = utils.sbir.simple_sbir(np.array([obj_crop]), skt_features, return_mat=True)
    return tf.gather(sketches, rank_mat[:, :5])[0]


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--target-dir', default='/scratch/lribeiro/datasets/qdcoco-fg-tf')
    parser.add_argument('--n-chunks', type=int, default=5)

    parser.add_argument("--model", default=None, help="Model that we are going to train")
    parser.add_argument("--id", default="0", help="experiment signature")
    parser.add_argument("--data-loader", default='stroke3-distributed',
                        help="Data loader that will provide data for model")
    parser.add_argument("-o", "--output-dir", default="", help="output directory")
    parser.add_argument('-p', "--hparams", default=None,
                        help="Parameters to override")
    parser.add_argument('-dp', "--data-hparams", default=None,
                        help="Data Parameters")
    parser.add_argument("--masked", action="store_true")
    parser.add_argument("-g", "--gpu", default=0, type=int, nargs='+', help="GPU ID to run on", )
    args = parser.parse_args()

    # load up the model with same code as evaluate-metrics
    Model = models.get_model_by_name(args.model)
    DataLoader = dataloaders.get_dataloader_by_name(args.data_loader)
    hps = utils.hparams.combine_hparams_into_one(Model.default_hparams(), DataLoader.default_hparams())
    utils.hparams.load_config(hps, Model.get_config_filepath(args.output_dir, args.id))
    if args.hparams:
        hps.parse(args.hparams)
    utils.gpu.setup_gpu(args.gpu)
    data_hps = DataLoader.parse_hparams(args.data_hparams)
    dataset = DataLoader(data_hps)
    model = Model(hps, dataset, args.output_dir, args.id)
    model.restore_checkpoint_if_exists('latest')

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)
    metafile = os.path.join(args.target_dir, 'meta.json')
    cocoset = dataloaders.get_dataloader_by_name('coco-tfrecord')()
    with open(os.path.join(cocoset.hps['coco_dir'], "meta.json")) as f:
        metadata = json.load(f)
    with open(metafile, 'w') as outfile:
        json.dump(metadata, outfile)

    load_data_and_save('valid', args.n_chunks, model, cocoset, args.target_dir, masked=args.masked)
    load_data_and_save('test', args.n_chunks, model, cocoset, args.target_dir, masked=args.masked)
    load_data_and_save('train', args.n_chunks, model, cocoset, args.target_dir, masked=args.masked)

if __name__ == '__main__':
    main()
