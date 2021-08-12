import argparse
import json
import os
import re
import glob

import imageio
import skimage.transform as sk_transform
import numpy as np
import tensorflow as tf

import utils


def default_hparams():
    hps = utils.hparams.HParams(
        image_size=256,
        min_object_size=0.05,
        min_objects_per_image=1,
        max_objects_per_image=8,
        include_image_obj=False,
        mask_size=64,
        excluded_meta_file='prep_data/coco/mats_objs_sketchables_v2.json',
        crop_size=96,
    )
    return hps


def load_all_data_and_save_in_chunks(set_name, image_ids, image_dir, id_to_filename, id_to_size,
                                     id_to_objects, target_dir, obj_ids_in_sketchycoco, ids_to_sketch_file, hps, meta):

    # load all images, saving one big chunk at a time
    image_counter = 0

    tf_record_shard_path = os.path.join(target_dir, "{}-{:03}.records".format(set_name, 0))
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter(tf_record_shard_path, options) as writer:
        for img_id in image_ids:
            img = imageio.imread(os.path.join(image_dir, id_to_filename[img_id]))
            size = id_to_size[img_id]
            objs = id_to_objects[img_id]
            img, c_objs, c_boxes, c_masks, c_triples, c_attributes, obj_ids = utils.coco.preprocess(
                img, size, objs, hps, meta, filter_obj_ids=obj_ids_in_sketchycoco, return_ids=True)
            if img is not None:
                sketches = load_sketches(obj_ids, ids_to_sketch_file)
                tf_example = utils.tfrecord.sketchycoco_scene_graph_example(
                    img,
                    c_objs,
                    c_boxes,
                    np.array(c_masks),
                    np.array(c_triples),
                    c_attributes.astype(np.int32),
                    img_id,
                    sketches)
                writer.write(tf_example.SerializeToString())
                image_counter += 1
            if image_counter % 100 == 0 and image_counter != 0:
                print("Saved {}/{} images from {} set".format(image_counter, len(image_ids), set_name))
    return image_counter


def load_all_sketchycoco_object_ids_and_sketches(objects_folder, split):
    all_the_files = [f for f in glob.glob(os.path.join(objects_folder, "GT", "{}/*/*".format(split))) if f.endswith('png')]
    object_ids = [int(re.findall(r'\d+', os.path.basename(f))[0]) for f in all_the_files]
    sketch_files = [f for f in glob.glob(os.path.join(objects_folder, "Sketch", "{}/*/*".format(split))) if f.endswith('png')]
    ids_to_sketch_file = {int(re.findall(r'\d+', os.path.basename(f))[0]): f for f in sketch_files}
    return object_ids, ids_to_sketch_file


def load_all_image_ids(image_folder, split):
    all_the_files = [f for f in glob.glob(os.path.join(image_folder, "GT", "{}/*".format(split))) if f.endswith('png')]
    object_ids = [int(re.findall(r'\d+', os.path.basename(f))[0]) for f in all_the_files]
    return object_ids


def load_sketches(ids, ids_to_sketch_file):
    sketches = np.zeros((len(ids), 96, 96, 1))
    for i, idd in enumerate(ids):
        skt = imageio.imread(ids_to_sketch_file[idd])
        sketches[i] = preprocess_sketch(skt, 96)
    return sketches


def preprocess_sketch(image, image_size):
    scaled = sk_transform.resize(image, (image_size, image_size))
    if scaled.shape[-1] == 3:
        scaled = scaled[..., 0]
    if len(scaled.shape) == 2:
        scaled = np.reshape(scaled,
                            (image_size, image_size, 1))
    return scaled


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir', default='/store/shared/datasets/coco-stuff')
    parser.add_argument('--target-dir', default='/scratch/lribeiro/datasets/sketchycoco-tf')
    parser.add_argument('--sketchycoco-dir', default='/store/shared/datasets/SketchyCOCO')
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
    sketchycoco_objs_dir = os.path.join(args.sketchycoco_dir, "Object")
    sketchycoco_imgs_dir = os.path.join(args.sketchycoco_dir, "Scene")
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

    valid_n_images = len(valid_image_ids)

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
        'concrete_objs': concrete_objs,
        'allowed_materials': allowed_materials,
        'fully_excluded_objs': fully_excluded_objs,
        'pred_idx_to_name': pred_idx_to_name,
        'pred_name_to_idx': pred_name_to_idx
    }
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    obj_ids_valid, ids_to_sketch_file_valid = load_all_sketchycoco_object_ids_and_sketches(sketchycoco_objs_dir, 'val')
    obj_ids_train, ids_to_sketch_file_train = load_all_sketchycoco_object_ids_and_sketches(sketchycoco_objs_dir, 'train')
    valid_filtered_img_ids = load_all_image_ids(sketchycoco_imgs_dir, 'valInTrain')
    test_filtered_img_ids = load_all_image_ids(sketchycoco_imgs_dir, 'val')
    train_filtered_img_ids = load_all_image_ids(sketchycoco_imgs_dir, 'trainInTrain')

    # test
    # meta['n_test_samples'] = load_all_data_and_save_in_chunks(
    #     'test',
    #     image_ids=test_filtered_img_ids,
    #     image_dir=val_image_dir,
    #     id_to_filename=valid_image_id_to_filename,
    #     id_to_size=valid_image_id_to_size,
    #     id_to_objects=valid_image_id_to_objects,
    #     target_dir=args.target_dir,
    #     obj_ids_in_sketchycoco=obj_ids_valid,
    #     ids_to_sketch_file=ids_to_sketch_file_valid,
    #     meta=meta,
    #     hps=hps)
    # print("Saved {} crops for test set".format(meta['n_test_samples']))
    # with open(meta_filename, 'w') as outfile:
    #     json.dump(meta, outfile)

    # valid
    meta['n_valid_samples'] = load_all_data_and_save_in_chunks(
        'valid',
        image_ids=valid_filtered_img_ids,
        image_dir=train_image_dir,
        id_to_filename=train_image_id_to_filename,
        id_to_size=train_image_id_to_size,
        id_to_objects=train_image_id_to_objects,
        target_dir=args.target_dir,
        obj_ids_in_sketchycoco=obj_ids_train + obj_ids_valid,
        ids_to_sketch_file={**ids_to_sketch_file_train, **ids_to_sketch_file_valid},
        meta=meta,
        hps=hps)
    print("Saved {} crops for valid set".format(meta['n_valid_samples']))
    with open(meta_filename, 'w') as outfile:
        json.dump(meta, outfile)

    # finally, the train set
    # meta['n_train_samples'] = load_all_data_and_save_in_chunks(
    #     'train',
    #     image_ids=train_filtered_img_ids,
    #     image_dir=train_image_dir,
    #     id_to_filename=train_image_id_to_filename,
    #     id_to_size=train_image_id_to_size,
    #     id_to_objects=train_image_id_to_objects,
    #     target_dir=args.target_dir,
    #     obj_ids_in_sketchycoco=obj_ids_train,
    #     ids_to_sketch_file=ids_to_sketch_file_train,
    #     meta=meta,
    #     hps=hps)
    # print("Saved {} crops for train set".format(meta['n_train_samples']))
    # with open(meta_filename, 'w') as outfile:
    #     json.dump(meta, outfile)


if __name__ == '__main__':
    main()
