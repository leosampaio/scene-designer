import argparse
import json
import os

import imageio
from collections import defaultdict
import skimage.transform as sk_transform
import pycocotools.mask as coco_mask_utils
import numpy as np
import tensorflow as tf

import utils


def default_hparams():
    hps = utils.hparams.HParams(
        image_size=256,
        min_object_size=0.05,
        excluded_meta_file='prep_data/coco/mats_objs_sketchables_v2.json',
        crop_size=96,
    )
    return hps


def load_all_data_and_save_in_chunks(set_name, image_ids, image_dir, id_to_filename, id_to_size,
                                     id_to_objects, target_dir, hps, meta, masked=False):

    # load all images, saving one big chunk at a time
    image_counter = 0
    crops_counter = 0

    options = tf.io.TFRecordOptions(compression_type="GZIP")
    tfrecord_paths = [os.path.join(target_dir, "{}-c{:03}.records".format(set_name, c)) for c in range(len(meta['obj_idx_to_ID']))]
    writers = [tf.io.TFRecordWriter(tfp, options) for tfp in tfrecord_paths]
    for img_id in image_ids:
        img = imageio.imread(os.path.join(image_dir, id_to_filename[img_id]))
        size = id_to_size[img_id]
        objs = id_to_objects[img_id]
        c_crops, c_objs_y = utils.coco.preprocess_for_boxes(img, size, objs, hps, meta, masked=masked)

        if c_crops is not None:
            for crop, y in zip(c_crops, c_objs_y):
                tf_example = utils.tfrecord.coco_crop_example(crop.numpy(), y)
                writers[y].write(tf_example.SerializeToString())
                crops_counter += 1
            image_counter += 1
        if image_counter % 100 == 0:
            print("Saved {} crops from {} images from {} set".format(crops_counter, image_counter, set_name))
    return crops_counter


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/coco-stuff')
    parser.add_argument('--target-dir', default='/store/lribeiro/datasets/cococrops-tfrecord')
    parser.add_argument('--val-size', type=int, default=2048)
    parser.add_argument("--masked", action="store_true")
    parser.add_argument('--hparams', type=str)

    args = parser.parse_args()

    hps = default_hparams()
    if args.hparams is not None:
        hps = hps.parse(args.hparams)
    hps = dict(hps.values())

    # get all the full paths
    train_image_dir = os.path.join(args.dataset_dir, 'images/train2017')
    val_image_dir = os.path.join(args.dataset_dir, 'images/val2017')
    train_instances_json = os.path.join(args.dataset_dir, 'annotations/instances_train2017.json')
    train_stuff_json = os.path.join(args.dataset_dir, 'annotations/stuff_train2017.json')
    val_instances_json = os.path.join(args.dataset_dir, 'annotations/instances_val2017.json')
    val_stuff_json = os.path.join(args.dataset_dir, 'annotations/stuff_val2017.json')
    meta_filename = os.path.join(args.target_dir, "meta.json")

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    # load up all train metadata
    print("Loading train metadata...")
    (object_idx_to_name, object_name_to_idx, objects_list, total_objs,
     train_image_ids,
     train_image_id_to_filename,
     train_image_id_to_size,
     train_image_id_to_objects) = utils.coco.prepare_and_load_metadata(train_instances_json, train_stuff_json)
    train_n_images = len(train_image_ids)

    # load up all valid metadata
    print("Loading validation metadata...")
    (_, _, _, _,
     valid_image_ids,
     valid_image_id_to_filename,
     valid_image_id_to_size,
     valid_image_id_to_objects) = utils.coco.prepare_and_load_metadata(val_instances_json, val_stuff_json)

    # break valid and train into two sets
    test_image_ids = valid_image_ids[args.val_size:]
    valid_image_ids = valid_image_ids[:args.val_size]
    test_n_images, valid_n_images = len(test_image_ids), len(valid_image_ids)

    with open(hps['excluded_meta_file']) as emf:
        materials_metadata = json.load(emf)
        concrete_objs = materials_metadata["objects"]
        allowed_materials = materials_metadata["materials"]
        fully_excluded_objs = materials_metadata["fully_excluded"]

    object_id_to_idx = {ident: i for i, ident in enumerate(objects_list)}

    # include info about the extra __image__ object
    object_id_to_idx[0] = len(objects_list)
    objects_list = np.append(objects_list, 0)

    print("Saving metadata...")
    PREDICATES_VALUES = ['left of', 'right of', 'above', 'below', 'inside', 'surrounding']
    pred_idx_to_name = ['__in_image__'] + PREDICATES_VALUES
    pred_name_to_idx = {name: idx for idx, name in enumerate(pred_idx_to_name)}
    meta = {
        'obj_name_to_ID': object_name_to_idx,
        'obj_ID_to_name': object_idx_to_name,
        'obj_idx_to_ID': objects_list.tolist(),
        'obj_ID_to_idx': object_id_to_idx,
        'obj_idx_to_name': [object_idx_to_name[objects_list[i]] for i in range(len(objects_list))],
        'train_total_objs': total_objs,
        'n_train_samples': train_n_images,
        'n_valid_samples': valid_n_images,
        'n_test_samples': test_n_images,
        'concrete_objs': concrete_objs,
        'allowed_materials': allowed_materials,
        'fully_excluded_objs': fully_excluded_objs,
        'pred_idx_to_name': pred_idx_to_name,
        'pred_name_to_idx': pred_name_to_idx
    }
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    # validation
    meta['n_valid_samples'] = load_all_data_and_save_in_chunks(
        'valid',
        image_ids=valid_image_ids,
        image_dir=val_image_dir,
        id_to_filename=valid_image_id_to_filename,
        id_to_size=valid_image_id_to_size,
        id_to_objects=valid_image_id_to_objects,
        target_dir=args.target_dir,
        meta=meta,
        hps=hps,
        masked=args.masked)
    print("Saved {} crops for valid set".format(meta['n_valid_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    # test
    meta['n_test_samples'] = load_all_data_and_save_in_chunks(
        'test',
        image_ids=test_image_ids,
        image_dir=val_image_dir,
        id_to_filename=valid_image_id_to_filename,
        id_to_size=valid_image_id_to_size,
        id_to_objects=valid_image_id_to_objects,
        target_dir=args.target_dir,
        meta=meta,
        hps=hps,
        masked=args.masked)
    print("Saved {} crops for test set".format(meta['n_test_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    # finally, the train set
    meta['n_train_samples'] = load_all_data_and_save_in_chunks(
        'train',
        image_ids=train_image_ids,
        image_dir=train_image_dir,
        id_to_filename=train_image_id_to_filename,
        id_to_size=train_image_id_to_size,
        id_to_objects=train_image_id_to_objects,
        target_dir=args.target_dir,
        meta=meta,
        hps=hps,
        masked=args.masked)
    print("Saved {} crops for train set".format(meta['n_train_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)


if __name__ == '__main__':
    main()
