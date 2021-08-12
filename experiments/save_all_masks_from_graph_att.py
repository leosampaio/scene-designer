import os
import glob

import numpy as np
import tensorflow as tf

import utils
from core.experiments import Experiment


class SaveMasksFromGraphBasedMaskGenerator(Experiment):
    name = "save-masks"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            target_dir=''
        )
        return hps

    def generate_masks_for_set(self, set_type, imgs_per_file=False):
        if set_type == 'train':
            n_samples = self.model.dataset.n_samples
        elif set_type == 'test':
            n_samples = self.model.dataset.n_test_samples
        elif set_type == 'valid':
            n_samples = self.model.dataset.n_valid_samples

        all_objs, all_bboxes, all_masks, all_s_bboxes, all_s_masks, all_o_bboxes, all_o_masks, all_skts, all_imgs = self.reset_arrays()
        batch_iterator = self.model.dist_eval_dataset(split_name=set_type, stop_at_end_of_split=True)
        file_cursor = self.model.dataset.splits[set_type].file_cursor
        counter, previous_div, file_counter = 0, 0, 0
        for batch in batch_iterator:
            if imgs_per_file:
                should_save = previous_div > (counter % imgs_per_file)
                previous_div = counter % imgs_per_file
            else:
                should_save = file_cursor != self.model.dataset.splits[set_type].file_cursor - 1
            if should_save:
                all_skts = np.concatenate(all_skts, axis=0)
                all_imgs = self.model.dataset.deprocess_image(np.concatenate(all_imgs, axis=0))
                self.save_masks_for_set(set_type, file_counter, all_objs, all_bboxes, all_masks, all_s_bboxes, all_s_masks, all_o_bboxes, all_o_masks, all_skts, all_imgs)
                all_objs, all_bboxes, all_masks, all_s_bboxes, all_s_masks, all_o_bboxes, all_o_masks, all_skts, all_imgs = self.reset_arrays()
                file_counter += 1
            file_cursor = self.model.dataset.splits[set_type].file_cursor - 1

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
            (objs_per_image, boxes_per_image, masks_per_image,
                boxes_s_per_image, masks_s_per_image, boxes_o_per_image,
                masks_o_per_image) = self.gather_objs_attrs(objs, boxes, masks, skt_boxes_pred, skt_masks_pred, boxes_pred, masks_pred, objs_s, objs_to_img, objs_to_img_s, imgs.shape[0])
            all_objs += objs_per_image
            all_bboxes += boxes_per_image
            all_masks += masks_per_image
            all_s_bboxes += boxes_s_per_image
            all_s_masks += masks_s_per_image
            all_o_bboxes += boxes_o_per_image
            all_o_masks += masks_o_per_image

            img_s = self.model.dataset.hps['image_size']
            batchs = self.model.hps['batch_size'] // self.model.strategy.num_replicas_in_sync if self.model.hps['distribute'] else self.model.hps['batch_size']

            def fun(skts, boxes, obj_to_img):
                return utils.layout.get_sketch_composition(skts.numpy(), boxes, obj_to_img, img_s, img_s, batchs)
            skt_comp = self.model.reduce_lambda_concat(fun, sketches, boxes_s, objs_to_img_s)

            all_skts.append(np.concatenate((skt_comp, skt_comp, skt_comp), axis=-1))
            all_imgs.append(imgs)
            counter += self.model.hps['batch_size']
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break

    def reset_arrays(self):
        return [], [], [], [], [], [], [], [], []

    def save_masks_for_set(self, set_type, file_cursor, all_objs, all_bboxes,
                           all_masks, all_s_bboxes, all_s_masks, all_o_bboxes,
                           all_o_masks, all_skts, all_imgs):
        filepath = os.path.join(self.hps['target_dir'], "{}_{:03}.npz".format(set_type, file_cursor))
        print("[DATA] Saving {}...".format(filepath))
        np.savez(filepath,
                 imgs=all_imgs,
                 objs=all_objs,
                 boxes=all_bboxes,
                 masks=all_masks,
                 boxes_s=all_s_bboxes,
                 masks_s=all_s_masks,
                 boxes_o=all_o_bboxes,
                 masks_o=all_o_masks,
                 sketches=all_skts)

    def gather_objs_attrs(self, objs, boxes, masks, boxes_s, masks_s, boxes_o, masks_o, objs_s, obj_to_img, obj_to_img_s, n_imgs):
        objs_per_image = []
        boxes_per_image = []
        masks_per_image = []
        boxes_s_per_image = []
        masks_s_per_image = []
        boxes_o_per_image = []
        masks_o_per_image = []
        for img_id in tf.range(n_imgs):

            cursor = 0
            boxes_gt, masks_gt = boxes[obj_to_img == img_id], masks[obj_to_img == img_id]
            objs_gt = objs[obj_to_img == img_id]
            cur_objs_s = objs_s[obj_to_img_s == img_id]
            cur_boxes_s, cur_masks_s = boxes_s[obj_to_img_s == img_id], masks_s[obj_to_img_s == img_id]
            cur_boxes_o, cur_masks_o = boxes_o[obj_to_img_s == img_id], masks_o[obj_to_img_s == img_id]
            all_boxes_s, all_masks_s, all_boxes_o, all_masks_o = [], [], [], []
            for obj, box_gt, mask_gt in zip(objs_gt, boxes_gt, masks_gt):
                if self.model.dataset.coco_classes_mask[obj]:
                    all_boxes_s.append(cur_boxes_s[cursor].numpy())
                    all_masks_s.append(cur_masks_s[cursor].numpy())
                    all_boxes_o.append(cur_boxes_o[cursor].numpy())
                    all_masks_o.append(cur_masks_o[cursor].numpy())
                    if cur_objs_s[cursor] != obj:
                        print("[ERROR] Objects don't match (GT is {} and Sketchable is {})".format(obj, cur_objs_s[cursor]))
                    cursor += 1
                else:
                    all_boxes_s.append(box_gt.numpy())
                    all_masks_s.append(mask_gt.numpy())
                    all_boxes_o.append(box_gt.numpy())
                    all_masks_o.append(mask_gt.numpy())

            objs_per_image.append(objs_gt.numpy())
            boxes_per_image.append(boxes_gt.numpy())
            masks_per_image.append(masks_gt.numpy())
            boxes_s_per_image.append(all_boxes_s)
            masks_s_per_image.append(all_masks_s)
            boxes_o_per_image.append(all_boxes_o)
            masks_o_per_image.append(all_masks_o)

        return (objs_per_image, boxes_per_image, masks_per_image,
                boxes_s_per_image, masks_s_per_image, boxes_o_per_image,
                masks_o_per_image)

    def forward(self, imgs, objs_s, triples_s, masks_s, boxes_s, attr_s, objs_to_img_s, sketches, n_crops):
        return self.model.strategy.experimental_run_v2(
            self.model.forward, args=(imgs, objs_s, triples_s, masks_s, boxes_s, attr_s, objs_to_img_s, sketches, n_crops))

    def compute(self, model=None):
        self.model = model

        if not os.path.isdir(self.hps['target_dir']):
            os.mkdir(self.hps['target_dir'])
        self.generate_masks_for_set('valid')
        self.generate_masks_for_set('test')
        # self.generate_masks_for_set('train', imgs_per_file=3000)
