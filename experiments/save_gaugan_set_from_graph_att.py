import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio
import json

import utils
from core.experiments import Experiment


class SaveGauganCompatibleLayoutsFromGraphBasedMaskGenerator(Experiment):
    name = "save-gaugan-layouts"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            target_dir='',
            excluded_file='prep_data/coco/excluded_scoco_with_bg.json',
        )
        return hps

    def prepare_all_directories(self, set_type):
        self.sketch_label_dir = os.path.join(self.hps['target_dir'], "{}_sketch_label".format(set_type))
        if not os.path.isdir(self.sketch_label_dir):
            os.mkdir(self.sketch_label_dir)

        self.sketch_inst_dir = os.path.join(self.hps['target_dir'], "{}_sketch_inst".format(set_type))
        if not os.path.isdir(self.sketch_inst_dir):
            os.mkdir(self.sketch_inst_dir)

        self.image_label_dir = os.path.join(self.hps['target_dir'], "{}_image_label".format(set_type))
        if not os.path.isdir(self.image_label_dir):
            os.mkdir(self.image_label_dir)

        self.image_inst_dir = os.path.join(self.hps['target_dir'], "{}_image_inst".format(set_type))
        if not os.path.isdir(self.image_inst_dir):
            os.mkdir(self.image_inst_dir)

        self.gt_label_dir = os.path.join(self.hps['target_dir'], "{}_gt_label".format(set_type))
        if not os.path.isdir(self.gt_label_dir):
            os.mkdir(self.gt_label_dir)

        self.gt_inst_dir = os.path.join(self.hps['target_dir'], "{}_gt_inst".format(set_type))
        if not os.path.isdir(self.gt_inst_dir):
            os.mkdir(self.gt_inst_dir)

        self.images_dir = os.path.join(self.hps['target_dir'], "{}_images".format(set_type))
        if not os.path.isdir(self.images_dir):
            os.mkdir(self.images_dir)

        self.sketches_dir = os.path.join(self.hps['target_dir'], "{}_sketches".format(set_type))
        if not os.path.isdir(self.sketches_dir):
            os.mkdir(self.sketches_dir)

    def generate_masks_for_set(self, set_type, imgs_per_file=False):
        if set_type == 'train':
            n_samples = self.model.dataset.n_samples
        elif set_type == 'test':
            n_samples = self.model.dataset.n_test_samples
        elif set_type == 'valid':
            n_samples = self.model.dataset.n_valid_samples

        self.img_counter_id = 0

        self.prepare_all_directories(set_type)

        all_objs, all_bboxes, all_masks, all_s_bboxes, all_s_masks, all_o_bboxes, all_o_masks, all_skts, all_imgs = self.reset_arrays()
        batch_iterator = self.model.dist_eval_dataset(split_name=set_type, stop_at_end_of_split=True)
        counter, file_counter = 0, 0
        for batch in batch_iterator:

            (imgs, objs, boxes, masks, objs_to_img, objs_s,
             boxes_s, masks_s, triples_s, attr_s, objs_to_img_s,
             sketches, n_crops, neg_labels) = batch

            out = self.forward(imgs, objs_s, triples_s, masks_s, boxes_s, attr_s, objs_to_img_s, sketches, n_crops)
            (img_reps, img_tri_reps, img_skt_reps,
             img_skt_tri_reps, img_rep_recon,
             skt_rep_recon,
             obj_vecs, skt_obj_vecs,
             n_img_tri_reps,
             masks_pred, boxes_pred,
             skt_masks_pred, skt_boxes_pred) = out

            # img_reps, img_skt_reps, imgs = self.reduce_concat(img_reps, img_skt_reps, imgs)
            img_s = self.model.dataset.hps['image_size']
            batchs = self.model.hps['batch_size'] // self.model.strategy.num_replicas_in_sync if self.model.hps['distribute'] else self.model.hps['batch_size']

            def fun(skts, boxes, obj_to_img):
                return utils.layout.get_sketch_composition(skts.numpy(), boxes, obj_to_img, img_s, img_s, batchs)
            skt_comp = self.model.reduce_lambda_concat(fun, sketches, boxes_s, objs_to_img_s)

            objs_bg_per_image, boxes_bg_per_image, masks_bg_per_image = self.gather_objs_attrs(objs, boxes, masks, objs_to_img, batchs)
            objs_gt_per_image, boxes_gt_per_image, masks_gt_per_image = self.gather_objs_attrs(objs_s, boxes_s, masks_s, objs_to_img_s, batchs)
            _, boxes_s_per_image, masks_s_per_image = self.gather_objs_attrs(objs_s, skt_boxes_pred, skt_masks_pred, objs_to_img_s, batchs)
            _, boxes_o_per_image, masks_o_per_image = self.gather_objs_attrs(objs_s, boxes_pred, masks_pred, objs_to_img_s, batchs)

            all_skts = np.concatenate((skt_comp, skt_comp, skt_comp), axis=-1)
            all_imgs = self.model.dataset.deprocess_image(imgs)
            self.save_masks_for_set(set_type, file_counter,
                                    objs_bg_per_image, objs_gt_per_image,
                                    boxes_bg_per_image, masks_bg_per_image,
                                    boxes_gt_per_image, masks_gt_per_image,
                                    boxes_s_per_image, masks_s_per_image,
                                    boxes_o_per_image, masks_o_per_image,
                                    all_skts, all_imgs)
            all_objs, all_bboxes, all_masks, all_s_bboxes, all_s_masks, all_o_bboxes, all_o_masks, all_skts, all_imgs = self.reset_arrays()

            counter += self.model.hps['batch_size']
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break

    def reset_arrays(self):
        return [], [], [], [], [], [], [], [], []

    def save_masks_for_set(self, set_type, file_cursor,
                           objs_bg_per_image, objs_gt_per_image,
                           boxes_bg_per_image, masks_bg_per_image,
                           boxes_gt_per_image, masks_gt_per_image,
                           boxes_s_per_image, masks_s_per_image,
                           boxes_o_per_image, masks_o_per_image,
                           all_skts, all_imgs):
        for (bg_objs, gt_objs, bg_boxes, bg_masks, gt_boxes, gt_masks, s_boxes, s_masks, o_boxes, o_masks, sketch, img) in zip(objs_bg_per_image, objs_gt_per_image,
                                                                                                                               boxes_bg_per_image, masks_bg_per_image,
                                                                                                                               boxes_gt_per_image, masks_gt_per_image,
                                                                                                                               boxes_s_per_image, masks_s_per_image,
                                                                                                                               boxes_o_per_image, masks_o_per_image,
                                                                                                                               all_skts, all_imgs):

            filepath_jpg = os.path.join("{:012}.jpg".format(self.img_counter_id))
            self.img_counter_id += 1

            id_objs = [self.model.dataset.obj_idx_list[i] - 1 if i != self.model.num_objs - 1 else 255 for i in tf.squeeze(gt_objs, axis=1)]
            one_hot_obj = tf.one_hot(id_objs, 256)
            inst_one_hot = tf.one_hot(list(range(len(gt_objs))), len(gt_objs))
            obj_to_img = tf.zeros(len(gt_objs), dtype=tf.int32)

            id_objs_bg = [self.model.dataset.obj_idx_list[i] - 1 if i != self.model.num_objs - 1 else 255 for i in tf.squeeze(bg_objs, axis=1)]
            one_hot_obj_bg = tf.one_hot(id_objs_bg, 256)
            obj_to_img_bg = tf.zeros(len(bg_objs), dtype=tf.int32)

            H, W = self.model.dataset.hps['image_size'], self.model.dataset.hps['image_size']
            bg_label_layout = utils.layout.masks_to_layout(one_hot_obj_bg, bg_boxes, bg_masks, obj_to_img_bg, H, W, 1, test_mode=True).numpy()

            s_label_layout = utils.layout.masks_to_layout(one_hot_obj, s_boxes, s_masks, obj_to_img, H, W, 1, test_mode=True).numpy()
            s_inst_layout = utils.layout.masks_to_layout(inst_one_hot, s_boxes, s_masks, obj_to_img, H, W, 1, test_mode=True).numpy()
            o_label_layout = utils.layout.masks_to_layout(one_hot_obj, o_boxes, o_masks, obj_to_img, H, W, 1, test_mode=True).numpy()
            o_inst_layout = utils.layout.masks_to_layout(inst_one_hot, o_boxes, o_masks, obj_to_img, H, W, 1, test_mode=True).numpy()
            label_layout = utils.layout.masks_to_layout(one_hot_obj, gt_boxes, gt_masks, obj_to_img, H, W, 1, test_mode=True).numpy()
            inst_layout = utils.layout.masks_to_layout(inst_one_hot, gt_boxes, gt_masks, obj_to_img, H, W, 1, test_mode=True).numpy()

            with open(self.hps['excluded_file']) as emf:
                materials_metadata = json.load(emf)
                background_list = materials_metadata["materials"]
            bg_label_layout = self.fix_holes_in_layouts(bg_label_layout, background_list)

            s_label_layout = self.combine_layouts(bg_label_layout, s_label_layout)
            o_label_layout = self.combine_layouts(bg_label_layout, o_label_layout)
            label_layout = self.combine_layouts(bg_label_layout, label_layout)

            s_label_layout, s_inst_layout, o_label_layout, o_inst_layout, label_layout, inst_layout = self.convert_all_layouts(
                s_label_layout, s_inst_layout, o_label_layout, o_inst_layout, label_layout, inst_layout)
            self.save_all_layouts(
                filepath_jpg, s_label_layout, s_inst_layout, o_label_layout, o_inst_layout, label_layout, inst_layout, img, sketch)

    def save_all_layouts(self, filepath_jpg, s_label_layout, s_inst_layout,
                         o_label_layout, o_inst_layout, label_layout, inst_layout, img, sketch):
        filepath_png = filepath_jpg.replace("jpg", "png")
        skio.imsave(os.path.join(self.sketch_label_dir, filepath_png), s_label_layout)
        skio.imsave(os.path.join(self.sketch_inst_dir, filepath_png), s_inst_layout)
        skio.imsave(os.path.join(self.image_label_dir, filepath_png), o_label_layout)
        skio.imsave(os.path.join(self.image_inst_dir, filepath_png), o_inst_layout)
        skio.imsave(os.path.join(self.gt_label_dir, filepath_png), label_layout)
        skio.imsave(os.path.join(self.gt_inst_dir, filepath_png), inst_layout)
        skio.imsave(os.path.join(self.images_dir, filepath_jpg), (img.numpy() * 255).astype(np.uint8))
        skio.imsave(os.path.join(self.sketches_dir, filepath_jpg), (sketch * 255).astype(np.uint8))

    def fix_holes_in_layouts(self, layouts, background_list):
        the_dummy_bg = 255
        for x in range(layouts.shape[0]):
            last_background = the_dummy_bg
            for i in range(layouts.shape[1]):
                for j in range(layouts.shape[2]):
                    current_obj = np.argmax(layouts[x, i, j])
                    if np.sum(layouts[x, i, j]) != 1:  # overlaps
                        layouts[x, i, j, current_obj] = 1
                    if current_obj == the_dummy_bg:
                        layouts[x, i, j, the_dummy_bg] = 0
                        layouts[x, i, j, last_background] = 1
                    elif self.model.dataset.obj_idx_to_name[current_obj + 1] in background_list:
                        last_background = current_obj
        return layouts

    def combine_layouts(self, background, objects):
        out = np.copy(background)
        out[np.sum(objects, axis=-1) > 0.5] = 0
        return out + objects

    def one_hot_to_categorical_layout(self, layout):
        categorical = np.argmax(layout, axis=-1)
        return categorical

    def convert_all_layouts(self, *args):
        res = []
        for a in args:
            res.append(
                np.reshape(
                    self.one_hot_to_categorical_layout(a), (self.model.dataset.hps['image_size'], self.model.dataset.hps['image_size'])
                ).astype(np.uint8)
            )
        return res

    def gather_objs_attrs(self, objs, boxes, masks, obj_to_img, n_imgs):
        objs_per_image = []
        boxes_per_image = []
        masks_per_image = []
        for img_id in tf.range(n_imgs):
            boxes, masks, objs = boxes[obj_to_img == img_id], masks[obj_to_img == img_id], objs[obj_to_img == img_id]
            objs_per_image.append(objs.numpy())
            boxes_per_image.append(boxes.numpy())
            masks_per_image.append(masks.numpy())
        return (objs_per_image, boxes_per_image, masks_per_image)

    def forward(self, imgs, objs_s, triples_s, masks_s, boxes_s, attr_s, objs_to_img_s, sketches, n_crops):
        return self.model.strategy.experimental_run_v2(
            self.model.forward, args=(imgs, objs_s, triples_s, masks_s, boxes_s, attr_s, objs_to_img_s, sketches, n_crops))

    def compute(self, model=None):
        self.model = model

        if not os.path.isdir(self.hps['target_dir']):
            os.mkdir(self.hps['target_dir'])
        self.generate_masks_for_set('valid')
        # self.generate_masks_for_set('test')
        # self.generate_masks_for_set('train', imgs_per_file=3000)
