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
        min_objects_per_image=1,
        max_objects_per_image=8,
        mask_size=64,
        include_image_obj=False,
        excluded_meta_file='prep_data/coco/mats_objs_sketchables_v2.json',
    )
    return hps


def load_all_data_and_save_in_chunks(n_chunks, chunk_size, image_ids,
                                     image_dir, id_to_filename, id_to_size,
                                     id_to_objects, base_filename, hps, meta):

    # load all images, saving one big chunk at a time
    image_counter = 0
    for cur_chunk in range(0, n_chunks):
        start_id = cur_chunk * chunk_size
        end_id = cur_chunk * chunk_size + chunk_size if cur_chunk + chunk_size < len(image_ids) else len(image_ids)

        tf_record_shard_path = base_filename.format(cur_chunk)
        options = tf.io.TFRecordOptions(compression_type="GZIP")

        print("Saving chunk {}/{} in {}...".format(cur_chunk + 1, n_chunks, tf_record_shard_path))
        with tf.io.TFRecordWriter(tf_record_shard_path, options) as writer:

            for i in range(start_id, end_id):
                img_id = image_ids[i]
                img = imageio.imread(os.path.join(image_dir, id_to_filename[img_id]))
                size = id_to_size[img_id]
                objs = id_to_objects[img_id]
                img, c_objs, c_boxes, c_masks, c_triples, c_attributes = utils.coco.preprocess(
                    img, size, objs, hps, meta)

                if img is not None:
                    tf_example = utils.tfrecord.coco_scene_graph_example(
                        img,
                        c_objs,
                        c_boxes,
                        np.array(c_masks),
                        np.array(c_triples),
                        c_attributes.astype(np.int32),
                        img_id)
                    writer.write(tf_example.SerializeToString())
                    image_counter += 1
    return image_counter


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--target-dir')
    parser.add_argument('--n-chunks', type=int, default=5)
    parser.add_argument('--test-n-chunks', type=int, default=1)
    parser.add_argument('--valid-n-chunks', type=int, default=1)
    parser.add_argument('--val-size', type=int, default=1024)
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
    train_basename = os.path.join(args.target_dir, "train_{:03}.npz")
    valid_basename = os.path.join(args.target_dir, "valid_{:03}.npz")
    test_basename = os.path.join(args.target_dir, "test_{:03}.npz")
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
        n_chunks=args.valid_n_chunks,
        chunk_size=valid_n_images // args.valid_n_chunks,
        image_ids=valid_image_ids,
        image_dir=val_image_dir,
        id_to_filename=valid_image_id_to_filename,
        id_to_size=valid_image_id_to_size,
        id_to_objects=valid_image_id_to_objects,
        base_filename=valid_basename,
        meta=meta,
        hps=hps)
    print("Saved {} images for valid set".format(meta['n_valid_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    # test
    meta['n_test_samples'] = load_all_data_and_save_in_chunks(
        n_chunks=args.test_n_chunks,
        chunk_size=test_n_images // args.test_n_chunks,
        image_ids=test_image_ids,
        image_dir=val_image_dir,
        id_to_filename=valid_image_id_to_filename,
        id_to_size=valid_image_id_to_size,
        id_to_objects=valid_image_id_to_objects,
        base_filename=test_basename,
        meta=meta,
        hps=hps)
    print("Saved {} images for test set".format(meta['n_test_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    # finally, the train set
    meta['n_train_samples'] = load_all_data_and_save_in_chunks(
        n_chunks=args.n_chunks,
        chunk_size=train_n_images // args.n_chunks,
        image_ids=train_image_ids,
        image_dir=train_image_dir,
        id_to_filename=train_image_id_to_filename,
        id_to_size=train_image_id_to_size,
        id_to_objects=train_image_id_to_objects,
        base_filename=train_basename,
        meta=meta,
        hps=hps)
    print("Saved {} images for train set".format(meta['n_train_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)


if __name__ == '__main__':
    main()
