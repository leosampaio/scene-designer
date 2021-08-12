import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio
import json

import utils
from core.experiments import Experiment


class CollectLayoutsWithIDS(Experiment):
    name = "collect-crops-with-ids"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
        )
        return hps

    def prepare_all_directories(self, set_type):
        dataset_name = self.model.dataset.name.replace('-', '_')
        model_name = self.model.experiment_id.replace('-', '_')
        target_dir = os.path.join(self.hps['results_dir'], "{}_{}_{}_generation".format(model_name, dataset_name, set_type))

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        self.crops_dir = os.path.join(target_dir, "crops")
        if not os.path.isdir(self.crops_dir):
            os.mkdir(self.crops_dir)

    def generate_masks_for_set(self, set_type, imgs_per_file=False):
        if set_type == 'train':
            n_samples = self.model.dataset.n_samples
        elif set_type == 'test':
            n_samples = self.model.dataset.n_test_samples
        elif set_type == 'valid':
            n_samples = self.model.dataset.n_valid_samples

        self.prepare_all_directories(set_type)

        gcrops, img_gcrops, gt_gcrops, mixed_gcrops, all_objs = {}, {}, {}, {}, {}
        batch_iterator = self.model.dataset.batch_iterator(set_type, 1, stop_at_end_of_split=True)
        counter = 0
        for batch in batch_iterator:

            (imgs, objs_bg, boxes_bg, masks_bg, objs_to_img_bg, objs,
             boxes, masks, triples, attr, objs_to_img,
             sketches, n_crops, neg_labels, identifier) = batch
            sketches = np.array(sketches, dtype=np.float32)
            crops = utils.bbox.crop_bbox_batch(imgs, boxes, objs_to_img, self.model.dataset.hps['crop_size'])

            # generate all of it
            # sketches
            _, obj_co_vecs = self.model.inference_mixed_representation(sketches, None, boxes)
            masks_pred, boxes_pred = self.model.inference_generation(obj_co_vecs)
            ggan_layout, inst_layout = self.model.make_layout(
                boxes_pred, masks_pred, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
            gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
            gcrops[str(identifier[0])] = utils.bbox.crop_bbox_batch(gen_image[None], boxes_pred, objs_to_img, 224).numpy()

            # images-based
            _, obj_co_vecs = self.model.inference_mixed_representation(None, crops, boxes)
            masks_pred, boxes_pred = self.model.inference_generation(obj_co_vecs)
            ggan_layout, inst_layout = self.model.make_layout(
                boxes_pred, masks_pred, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
            img_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
            img_gcrops[str(identifier[0])] = utils.bbox.crop_bbox_batch(img_gen_image[None], boxes_pred, objs_to_img, 224).numpy()

            # from ground truth layout
            ggan_layout, inst_layout = self.model.make_layout(
                boxes, masks, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
            gtlayout_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
            gt_gcrops[str(identifier[0])] = utils.bbox.crop_bbox_batch(gtlayout_gen_image[None], boxes, objs_to_img, 224).numpy()

            # mixed domains
            if len(objs) > 1:
                half_of_objs = len(boxes) // 2
                img_skt_reps, obj_co_vecs = self.model.inference_mixed_representation(sketches[:half_of_objs], crops[half_of_objs:], boxes)
                masks_pred, boxes_pred = self.model.inference_generation(obj_co_vecs)
                ggan_layout, inst_layout = self.model.make_layout(
                    boxes_pred, masks_pred, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
                mixed_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
                mixed_gcrops[str(identifier[0])] = utils.bbox.crop_bbox_batch(mixed_gen_image[None], boxes_pred, objs_to_img, 224).numpy()

            all_objs[str(identifier[0])] = objs

            counter += 1
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break

        # np.savez(os.path.join(self.crops_dir, "sketch_gen.npz"), **gcrops)
        np.savez(os.path.join(self.crops_dir, "image_gen.npz"), **img_gcrops)
        np.savez(os.path.join(self.crops_dir, "gt_gen.npz"), **gt_gcrops)
        np.savez(os.path.join(self.crops_dir, "mixed_gen.npz"), **mixed_gcrops)
        np.savez(os.path.join(self.crops_dir, "objs.npz"), **all_objs)

    def compute(self, model=None):
        self.model = model

        self.generate_masks_for_set('test')
        self.generate_masks_for_set('valid')
