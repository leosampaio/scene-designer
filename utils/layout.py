import tensorflow as tf
import numpy as np
import json

import utils


def masks_to_layout(vecs, boxes, masks, obj_to_img, H, W, n_imgs, pooling='sum', test_mode=False):
    """
    Inputs:
    - vecs: Tensor of shape (O, D) giving vectors
    - boxes: Tensor of shape (O, 4) giving bounding boxes in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space
    - masks: Tensor of shape (O, M, M) giving binary masks for each object
    - obj_to_img: LongTensor of shape (O,) mapping objects to images
    - H, W: Size of the output image.

    Returns:
    - out: Tensor of shape (N, D, H, W)
    """
    if W is None:
        W = H

    X, Y = _boxes_to_grid(boxes, H, W)

    # multiply the vectors by the masks, making each mask
    # an image with the depth of the vectors' dimensionality
    if test_mode:
        masks = tf.cast(masks > 0.6, tf.float32)
    img_in = tf.expand_dims(tf.expand_dims(vecs, axis=1), axis=1)
    masks = tf.expand_dims(masks, axis=-1)
    img_in = img_in * masks

    # refit the valued masks into their places on the image
    sampled = utils.bbox.bilinear_sampler(img_in, X, Y)
    if test_mode:
        clean_mask_sampled = utils.bbox.bilinear_sampler(masks, X, Y)
    else:
        clean_mask_sampled = None

    out = _pool_samples(sampled, clean_mask_sampled, obj_to_img, n_imgs, pooling=pooling)
    return out


def _boxes_to_grid(boxes, H, W):
    """
    Input:
    - boxes: FloatTensor of shape (O, 4) giving boxes in the [x0, y0, x1, y1]
      format in the [0, 1] coordinate space
    - H, W: Scalars giving size of output

    Returns:
    - grid x, y: tensors of shape (O, H, W,) suitable for passing to bilinear_sampler
    """
    n_objs = tf.shape(boxes)[0]

    boxes = tf.reshape(boxes, (-1, 4, 1, 1))

    # All these are (n_objs, 1, 1)
    x0, y0 = boxes[:, 0], boxes[:, 1]
    ww, hh = boxes[:, 2] - x0, boxes[:, 3] - y0
    ww, hh = tf.clip_by_value(ww, 1e-4, 1.), tf.clip_by_value(hh, 1e-4, 1.)

    X = tf.linspace(0., 1., num=W)
    X = tf.tile(tf.reshape(X, (1, 1, W)), (n_objs, H, 1))
    Y = tf.linspace(0., 1., num=H)
    Y = tf.tile(tf.reshape(Y, (1, H, 1)), (n_objs, 1, W))

    X = (X - x0) / ww
    Y = (Y - y0) / hh

    # Right now grid is in [0, 1] space; transform to [-1, 1]
    X = (X * 2) - 1
    Y = (Y * 2) - 1

    return X, Y


def _pool_samples(samples, clean_mask_sampled, obj_to_img, n_imgs, pooling='sum'):
    """
    Input:
    - samples: FloatTensor of shape (O, D, H, W)
    - obj_to_img: LongTensor of shape (O,) with each element in the range
      [0, N) mapping elements of samples to output images

    Output:
    - pooled: FloatTensor of shape (N, D, H, W)
    """
    O = tf.shape(samples)[0]
    H, W, D = samples.shape[1:]

    # gather all objects for image and sum up their masks
    if clean_mask_sampled is None:
        out = tf.scatter_nd(updates=samples, indices=tf.reshape(obj_to_img, (-1, 1)), shape=(n_imgs, H, W, D))
    else:
        out = []
        mass = tf.reduce_sum(samples, axis=[1, 2, 3])
        for i in tf.range(n_imgs):
            indices = tf.squeeze(tf.where(obj_to_img == i), axis=-1)
            cur_mass = mass[obj_to_img == i]
            argsort = tf.argsort(cur_mass)
            result = tf.zeros((H, W, D), dtype=samples.dtype)
            result_clean = tf.zeros((H, W), dtype=samples.dtype)
            for j in argsort:
                masked_mask = tf.cast(result_clean == 0, tf.float32) * tf.cast(clean_mask_sampled[indices[j], ..., 0] > 0.5, tf.float32)
                result_clean += masked_mask
                result += samples[indices[j]] * tf.expand_dims(masked_mask, axis=-1)
            out.append(result)
        out = tf.identity(out)
    return out


def get_clean_mask(boxes, masks, obj_to_img, H, W, n_imgs, redistribute=True):

    X, Y = _boxes_to_grid(boxes, H, W)
    masks = tf.expand_dims(masks, axis=-1)

    clean_mask_sampled = utils.bbox.bilinear_sampler(masks, X, Y)
    out = tf.scatter_nd(updates=clean_mask_sampled, indices=tf.reshape(obj_to_img, (-1, 1)), shape=(n_imgs, H, W, 1))
    if redistribute:
        out = tf.gather(out, obj_to_img)

    return out


def get_sketch_composition(sketches, boxes, obj_to_img, H, W, n_imgs, avoid_overlap=False):

    if avoid_overlap:
        new_boxes = []
        boxes = np.array([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=np.float32)
        for img_id in np.unique(obj_to_img):
            cur_boxes = boxes[obj_to_img == img_id, ...]
            if np.sum(obj_to_img == img_id) == 1:
                new_boxes.append(cur_boxes[0])
                continue
            for a in range(len(cur_boxes)):
                box_a = cur_boxes[a]
                overlap_center_x, overlap_center_y = [], []
                for b in range(len(cur_boxes)):
                    box_b = cur_boxes[b]
                    box_b_min_x, box_b_max_x = box_b[0], box_b[2]
                    box_b_min_y, box_b_max_y = box_b[1], box_b[3]

                    box_a_min_x, box_a_max_x = box_a[0], box_a[2]
                    box_a_min_y, box_a_max_y = box_a[1], box_a[3]

                    # check if overlapping in each axis
                    # xmax1 >= xmin2 and xmax2 >= xmin1
                    overlapping_x = box_a_max_x > box_b_min_x and box_b_max_x > box_a_min_x
                    overlapping_y = box_a_max_y > box_b_min_y and box_b_max_y > box_a_min_y
                    if overlapping_x and overlapping_y:
                        center_b_x, center_b_y = (box_b_max_x + box_b_min_x) / 2, (box_b_max_y + box_b_min_y) / 2
                        overlap_center_x.append(center_b_x)
                        overlap_center_y.append(center_b_y)
                if len(overlap_center_x) > 0:
                    overlap_center_x = np.mean(overlap_center_x)
                    overlap_center_y = np.mean(overlap_center_y)
                    center_a_x, center_a_y = (box_a_max_x + box_a_min_x) / 2, (box_a_max_y + box_a_min_y) / 2
                    a_size_x, a_size_y = (box_a_max_x - box_a_min_x), (box_a_max_y - box_a_min_y)
                    delta_x, delta_y = center_a_x - overlap_center_x, center_a_y - overlap_center_y
                    delta_xn, delta_yn = np.sign(delta_x), np.sign(delta_y)
                    change_x = a_size_x / 4 - np.abs(delta_x)
                    change_y = a_size_y / 4 - np.abs(delta_y)
                    new_x0, new_x1 = box_a[0] + delta_xn * change_x, box_a[2] + delta_xn * change_x
                    new_y0, new_y1 = box_a[1] + delta_yn * change_y, box_a[3] + delta_yn * change_y
                    if change_x > 0 and (0 <= new_x1 <= 1.1):
                        cur_boxes[a][0] = np.clip(new_x0, 0., 1.)
                        cur_boxes[a][2] = np.clip(new_x1, 0., 1.)
                    if change_y > 0 and (0 < new_y1 <= 1.1):
                        cur_boxes[a][1] = np.clip(new_y0, 0., 1.)
                        cur_boxes[a][3] = np.clip(new_y1, 0., 1.)
            for box in cur_boxes:
                new_boxes.append(box)
        boxes = np.array(new_boxes, dtype=np.float32)
    X, Y = _boxes_to_grid(boxes, H, W)
    inv_skt = [s if s.mean() > 0.01 else np.ones_like(s) for s in sketches]
    inv_skt = -1 * (np.array(inv_skt) - 1)
    sampled_sketches = utils.bbox.bilinear_sampler(inv_skt, X, Y)
    out = tf.scatter_nd(updates=sampled_sketches, indices=tf.reshape(obj_to_img, (-1, 1)), shape=(n_imgs, H, W, 1))
    out = tf.clip_by_value(out, 0., 1.)
    out = -1 * (out - 1)
    return out


def get_single_sketch_composition(sketches, boxes, image_size, avoid_overlap=False, three_channels=True):
    obj_to_img = np.zeros((len(sketches),), dtype=np.int32)
    res = get_sketch_composition(sketches, boxes, obj_to_img, image_size, image_size, 1, avoid_overlap=avoid_overlap)
    if three_channels:
        res = np.concatenate((res, res, res),axis=-1)
    return res[0]


def get_sketch_composition_with_images(obj_imgs, sketches, boxes_o, boxes_s, obj_to_img_o, obj_to_img_s, H, W, n_imgs):
    if sketches is not None:
        X, Y = utils.layout._boxes_to_grid(boxes_s, H, W)
        inv_skt = [s if s.mean() > 0.01 else np.ones_like(s) for s in sketches]
        inv_skt = -1 * (np.array(inv_skt) - 1)
        sampled_sketches = utils.bbox.bilinear_sampler(inv_skt, X, Y)
        s_out = tf.scatter_nd(updates=sampled_sketches, indices=tf.reshape(obj_to_img_s, (-1, 1)), shape=(n_imgs, H, W, 1))
        s_out = tf.clip_by_value(s_out, 0., 1.)
        s_out = np.concatenate((s_out, s_out, s_out), axis=-1)
    else:
        s_out = 0

    X, Y = utils.layout._boxes_to_grid(boxes_o, H, W)
    inv_obj = -1 * (obj_imgs - 1)
    sampled_objs = utils.bbox.bilinear_sampler(inv_obj, X, Y)
    o_out = tf.scatter_nd(updates=sampled_objs, indices=tf.reshape(obj_to_img_o, (-1, 1)), shape=(n_imgs, H, W, 3))
    o_out = -1 * (o_out - 1)
    out = o_out - s_out
    return tf.clip_by_value(out, 0., 1.)


def get_single_sketch_composition_with_images(obj_imgs, sketches, boxes_o, boxes_s, image_size):
    if sketches is not None:
        obj_to_img_s = np.zeros((len(sketches),), dtype=np.int32)
    else:
        obj_to_img_s = None
    obj_to_img_o = np.zeros((len(obj_imgs),), dtype=np.int32)
    return get_sketch_composition_with_images(
        obj_imgs, sketches, boxes_o, boxes_s, obj_to_img_o, obj_to_img_s, image_size, image_size, 1)[0]


def one_hot_to_rgb(one_hot):
    if one_hot_to_rgb.colours is None:
        one_hot_to_rgb.colours = tf.random.uniform((tf.shape(one_hot)[-1], 3), minval=0, maxval=256, dtype=tf.int32)
        one_hot_to_rgb.colours = tf.cast(one_hot_to_rgb.colours, tf.float32)
    one_hot_3d = tf.einsum('abcd,de->abce', one_hot, one_hot_to_rgb.colours)
    one_hot_3d *= (1.0 / tf.reduce_max(one_hot_3d))
    return one_hot_3d
one_hot_to_rgb.colours = None


def fix_holes_in_layouts(layouts, obj_idx_to_name, obj_idx_list, background_list, dummy_bg=255, offset=1):
    for x in range(layouts.shape[0]):
        last_background = dummy_bg
        for i in range(layouts.shape[1]):
            for j in range(layouts.shape[2]):
                current_obj = np.argmax(layouts[x, i, j])
                if np.sum(layouts[x, i, j]) != 1:  # overlaps
                    layouts[x, i, j, current_obj] = 1
                if current_obj == dummy_bg:
                    layouts[x, i, j, dummy_bg] = 0
                    layouts[x, i, j, last_background] = 1
                elif obj_idx_to_name[obj_idx_list[current_obj + offset]] in background_list:
                    last_background = current_obj
    return layouts


def combine_layouts(background, objects):
    out = np.copy(background)
    out[np.sum(objects, axis=-1) > 0.5] = 0
    return out + objects


def convert_layout_to_gaugan(layout, obj_idx_list):
    categorical = np.argmax(layout[0], axis=-1)
    for x in range(categorical.shape[0]):
        for y in range(categorical.shape[1]):
            i = categorical[x, y]
            categorical[x, y] = obj_idx_list[i] - 1 if i != len(obj_idx_list) - 1 else 255
    return categorical


def masks_to_gaugan_instance_layout(objs, boxes, masks, image_size):
    obj_to_img = np.zeros((len(objs),), dtype=np.int32)
    inst_one_hot = tf.one_hot(list(range(len(objs))), len(objs))
    inst_layout = utils.layout.masks_to_layout(
        inst_one_hot, boxes, masks, obj_to_img, image_size, image_size, 1, test_mode=True).numpy()
    inst_layout = np.argmax(inst_layout[0], axis=-1)
    return inst_layout
