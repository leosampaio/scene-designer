import glob
import argparse
import json
import os
import sys
import pickle

import imageio
from collections import defaultdict
import skimage.transform as sk_transform
import pycocotools.mask as coco_mask_utils
import numpy as np
import tensorflow as tf

import utils


def load_metadata_from_json(instances_json, stuff_json):

    with open(instances_json, 'r') as f:
        instances_data = json.load(f)

    stuff_data = None
    if stuff_json is not None and stuff_json != '':
        with open(stuff_json, 'r') as f:
            stuff_data = json.load(f)
    return instances_data, stuff_data


def create_vocab(instances_data_categories, stuff_data_categories):
    object_idx_to_name = {}
    object_name_to_idx = {}
    for category_data in instances_data_categories:
        category_id = category_data['id']
        category_name = category_data['name']
        object_idx_to_name[category_id] = category_name
        object_name_to_idx[category_name] = category_id
    for category_data in stuff_data_categories:
        category_id = category_data['id']
        category_name = category_data['name']
        object_idx_to_name[category_id] = category_name
        object_name_to_idx[category_name] = category_id
    return object_idx_to_name, object_name_to_idx


def create_images_and_objects_dataset(instances_data, stuff_data):
    image_ids, image_id_to_filename, image_id_to_size = [], {}, {}
    image_id_to_objects = defaultdict(list)
    for image_data in instances_data['images']:
        image_id = image_data['id']
        filename = image_data['file_name']
        width = image_data['width']
        height = image_data['height']
        image_ids.append(image_id)
        image_id_to_filename[image_id] = filename
        image_id_to_size[image_id] = (width, height)

    # Add object data from instances
    for object_data in instances_data['annotations']:
        image_id_to_objects[object_data['image_id']].append(object_data)

    # Add object data from stuff
    image_ids_with_stuff = set()
    new_image_ids = set()
    for object_data in stuff_data['annotations']:
        image_ids_with_stuff.add(object_data['image_id'])
        image_id_to_objects[object_data['image_id']].append(object_data)
    total_objs = 0
    for image_id in image_ids:
        num_objs = len(image_id_to_objects[image_id])
        new_image_ids.add(image_id)
        total_objs += num_objs

    image_ids = list(new_image_ids)

    objects_list = set()
    for image_id in image_ids:
        for object in image_id_to_objects[image_id]:
            object_class = object['category_id']
            objects_list.add(object_class)
    return list(objects_list), total_objs, image_ids, image_id_to_filename, image_id_to_size, image_id_to_objects


def prepare_and_load_metadata(instances_json, stuff_json):

    # load the metadata
    instances_data, stuff_data = load_metadata_from_json(instances_json, stuff_json)

    # separate category metadata
    object_idx_to_name, object_name_to_idx = create_vocab(
        instances_data['categories'], stuff_data['categories'])

    # break down data by image sample
    results = create_images_and_objects_dataset(
        instances_data, stuff_data)
    (objects_list, total_objs, image_ids, image_id_to_filename,
        image_id_to_size, image_id_to_objects) = results

    # category labels start at 1, so use 0 for __image__
    object_name_to_idx['__image__'] = 0
    object_idx_to_name[0] = '__image__'

    return (object_idx_to_name, object_name_to_idx, objects_list, total_objs,
            image_ids, image_id_to_filename,
            image_id_to_size, image_id_to_objects)


def should_keep_object(size, obj, hps, meta, materials=False):
    _, _, w, h = obj['bbox']
    W, H = size
    box_area = (w * h) / (W * H)
    box_ok = box_area > hps['min_object_size']
    object_name = meta['obj_ID_to_name'][obj['category_id']]
    if not materials:
        category_ok = object_name in meta['concrete_objs'] or object_name in meta['allowed_materials'] or object_name in meta['fully_excluded_objs']
    else:
        category_ok = object_name in meta['excluded_materials']
    is_not_other = object_name != 'other'
    return box_ok and category_ok and is_not_other


def should_keep_image(img, size, objects, min_objs, max_objs, meta):
    n_objs = len(objects)
    for o in objects:
        if meta['obj_ID_to_name'][o['category_id']] in meta['fully_excluded_objs']:
            return False  # images with this object are not allowed
    return (min_objs <= n_objs <= max_objs)


def preprocess_image(image, image_size):
    scaled = sk_transform.resize(image, (image_size, image_size))
    if len(scaled.shape) == 2:
        scaled = np.reshape(np.array([scaled, scaled, scaled]),
                            (image_size, image_size, 3))
    return scaled


def normalize_bounding_boxes(bbox, WW, HH):
    x, y, w, h = bbox
    x0 = x / WW
    y0 = y / HH
    x1 = (x + w) / WW
    y1 = (y + h) / HH
    return [x0, y0, x1, y1]


def extract_normalized_mask(segmentation, image_size, bbox, mask_size):
    WW, HH = image_size
    x, y, w, h = bbox

    # this will give a numpy array of shape (HH, WW)
    mask = seg_to_mask(segmentation, WW, HH)

    # crop the mask according to the bounding box, being careful to
    # ensure that we don't crop a zero-area region
    mx0, mx1 = int(round(x)), int(round(x + w))
    my0, my1 = int(round(y)), int(round(y + h))
    mx1 = max(mx0 + 1, mx1)
    my1 = max(my0 + 1, my1)
    mask = mask[my0:my1, mx0:mx1]
    mask = sk_transform.resize(
        255.0 * mask, (mask_size, mask_size),
        mode='constant', anti_aliasing=True)
    mask = (mask > 128).astype(np.int64)
    return mask


def compute_scene_graph_for_image(image_size, c_objs, hps, meta, include_image_obj=False):

    # create arrays to represent the current (c) image objects
    c_objs_y, c_boxes, c_masks = [], [], []

    # create the location and size attributes
    # we add 1 to account for the __image__ object
    WW, HH = image_size
    n_objs = len(c_objs) + 1 if include_image_obj else len(c_objs)

    # collect masks, ids and boxes
    for i, object_data in enumerate(c_objs):

        # get label of object
        c_objs_y.append(object_data['category_id'])

        x0, y0, x1, y1 = normalize_bounding_boxes(object_data['bbox'], WW, HH)
        c_boxes.append([x0, y0, x1, y1])

        mask = extract_normalized_mask(object_data['segmentation'], image_size,
                                       object_data['bbox'], hps['mask_size'])
        c_masks.append(mask)

    # add the dummy __image__ object
    if include_image_obj:
        c_objs_y.append(meta['obj_name_to_ID']['__image__'])
        c_boxes.append([0, 0, 1, 1])
        c_masks.append(np.ones((hps['mask_size'], hps['mask_size'])))

    c_objs_y = np.array(c_objs_y)
    triples, attributes = utils.graph.compute_scene_graph_for_image(c_boxes, meta['pred_name_to_idx'])

    # add __in_image__ triples
    if include_image_obj:
        n_objs = c_objs_y.shape[0]
        in_image = meta['pred_name_to_idx']['__in_image__']
        for i in range(n_objs - 1):
            triples.append([i, in_image, n_objs - 1])

    return c_objs_y, c_boxes, c_masks, triples, attributes


def seg_to_mask(seg, width=1.0, height=1.0):
    """
    Tiny utility for decoding segmentation masks using the pycocotools API.
    """
    if type(seg) == list:
        rles = coco_mask_utils.frPyObjects(seg, height, width)
        rle = coco_mask_utils.merge(rles)
    elif type(seg['counts']) == list:
        rle = coco_mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return coco_mask_utils.decode(rle)


def filter_objs(size, objs, hps, meta, filter_obj_ids=None, materials=False):
    c_objs, obj_ids = [], []
    for obj in objs:
        if obj['id'] in obj_ids:
            print("Repeated object {} found!".format(obj['id']))
            continue
        if filter_obj_ids is not None and obj['id'] not in filter_obj_ids:
            continue
        if should_keep_object(size, obj, hps, meta, materials):
            c_objs.append(obj)
            obj_ids.append(obj['id'])
    return c_objs


def preprocess(img, size, objs, hps, meta, filter_obj_ids=None, return_ids=False, return_materials=False):

    # create arrays to represent the current (c) image objects
    c_objs = filter_objs(size, objs, hps, meta, filter_obj_ids)
    if return_materials:
        bg_objs = filter_objs(size, objs, hps, meta, filter_obj_ids, materials=True)

    if not should_keep_image(img, size, c_objs, hps['min_objects_per_image'], hps['max_objects_per_image'], meta):
        n_nones = 6
        if return_ids:
            n_nones += 1
        elif return_materials:
            n_nones += 3
        return [None] * n_nones

    # preprocess image size
    WW, HH = size
    preprocessed_image = preprocess_image(img, hps['scene_size'])

    # compute all metadata to build a scene graph
    c_objs_y, c_boxes, c_masks, triples, attributes = compute_scene_graph_for_image(size, c_objs, hps, meta)

    # convert objs y from ids to idx's on the list
    c_objs_y = [meta['obj_ID_to_idx'][y] for y in c_objs_y]

    if return_materials:
        bg_c_objs_y, bg_c_boxes, bg_c_masks, _, _ = compute_scene_graph_for_image(size, bg_objs, hps, meta, include_image_obj=True)
        bg_c_objs_y = [meta['obj_ID_to_idx'][y] for y in bg_c_objs_y]

    return_vals = [preprocessed_image, c_objs_y, np.array(c_boxes), c_masks, triples, attributes]
    if return_ids:
        ids = [c['id'] for c in c_objs]
        return_vals += [ids]
    if return_materials:
        return_vals += [bg_c_objs_y, bg_c_boxes, bg_c_masks]
    return return_vals


def preprocess_for_boxes(img, size, objs, hps, meta, filter_obj_ids=None, return_ids=False, masked=False):

    c_objs = filter_objs(size, objs, hps, meta, filter_obj_ids)

    if not should_keep_image(img, size, c_objs, 1, 10, meta):
        if return_ids:
            return [None] * 3
        else:
            return [None] * 2

    preprocessed_image = preprocess_image(img, hps['scene_size'])

    # crop all the objects out of the image
    c_crops, c_objs_y = extract_obj_crops(preprocessed_image, size, c_objs, hps, meta, masked)

    # convert objs y from ids to idx's on the list
    c_objs_y = [meta['obj_ID_to_idx'][y] for y in c_objs_y]

    if return_ids:
        ids = [c['id'] for c in c_objs]
        return c_crops, c_objs_y, ids
    else:
        return c_crops, c_objs_y


def extract_obj_crops(image, image_size, c_objs, hps, meta, masked=False):

    # create arrays to represent the current (c) image objects
    c_objs_y, c_boxes, c_masks = [], [], []
    WW, HH = image_size
    for i, object_data in enumerate(c_objs):

        c_objs_y.append(object_data['category_id'])
        c_boxes.append(normalize_bounding_boxes(object_data['bbox'], WW, HH))

        if masked:
            mask = extract_normalized_mask(object_data['segmentation'], image_size,
                                           object_data['bbox'], hps['crop_size'])
            c_masks.append(mask)

    expanded_image = np.tile(np.expand_dims(image, axis=0), (len(c_objs), 1, 1, 1))
    c_crops = utils.bbox.crop_bbox(expanded_image, np.array(c_boxes), hps['crop_size'])

    if masked:
        masks = tf.cast(tf.expand_dims(c_masks, axis=-1), tf.float32)
        c_crops = tf.multiply(c_crops, masks)

    return c_crops, c_objs_y
