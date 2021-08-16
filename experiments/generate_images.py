import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio
import json

import utils
from core.experiments import Experiment


class GenerateImages(Experiment):
    name = "generate-images"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            flipped_sketches=False,
            generate_from_image=False,
            generate_from_mixed=False,
            generate_from_gt=False,
        )
        return hps

    def prepare_all_directories(self, set_type):
        dataset_name = self.model.dataset.name.replace('-', '_')
        model_name = self.model.experiment_id.replace('-', '_')
        target_dir = os.path.join(self.hps['results_dir'], "{}_{}_{}_generation".format(model_name, dataset_name, set_type))

        if not os.path.isdir(target_dir):
            os.mkdir(target_dir)

        self.images_dir = os.path.join(target_dir, "images")
        if not os.path.isdir(self.images_dir):
            os.mkdir(self.images_dir)

        self.sketches_dir = os.path.join(target_dir, "sketches")
        if not os.path.isdir(self.sketches_dir):
            os.mkdir(self.sketches_dir)

        self.flip_sketches_dir = os.path.join(target_dir, "sketches_flipped")
        if not os.path.isdir(self.flip_sketches_dir):
            os.mkdir(self.flip_sketches_dir)

        self.mixed_domain_dir = os.path.join(target_dir, "mixed")
        if not os.path.isdir(self.mixed_domain_dir):
            os.mkdir(self.mixed_domain_dir)

        self.gen_dir = os.path.join(target_dir, "generated_from_sketch")
        if not os.path.isdir(self.gen_dir):
            os.mkdir(self.gen_dir)

        self.img_gen_dir = os.path.join(target_dir, "generated_from_image")
        if not os.path.isdir(self.img_gen_dir):
            os.mkdir(self.img_gen_dir)

        self.flipped_gen_dir = os.path.join(target_dir, "generated_from_flipped")
        if not os.path.isdir(self.flipped_gen_dir):
            os.mkdir(self.flipped_gen_dir)

        self.mixed_gen_dir = os.path.join(target_dir, "generated_from_mixed_domains")
        if not os.path.isdir(self.mixed_gen_dir):
            os.mkdir(self.mixed_gen_dir)

        self.gt_gen_dir = os.path.join(target_dir, "generated_from_gt_layout")
        if not os.path.isdir(self.gt_gen_dir):
            os.mkdir(self.gt_gen_dir)

        self.sidebyside_dir = os.path.join(target_dir, "sketch_side_by_side")
        if not os.path.isdir(self.sidebyside_dir):
            os.mkdir(self.sidebyside_dir)

        self.sidebyside_layout_dir = os.path.join(target_dir, "sketch_side_by_side_layout")
        if not os.path.isdir(self.sidebyside_layout_dir):
            os.mkdir(self.sidebyside_layout_dir)

        self.flipped_sidebyside_dir = os.path.join(target_dir, "flipped_side_by_side")
        if not os.path.isdir(self.flipped_sidebyside_dir):
            os.mkdir(self.flipped_sidebyside_dir)

        self.mixed_sidebyside_dir = os.path.join(target_dir, "mixed_side_by_side")
        if not os.path.isdir(self.mixed_sidebyside_dir):
            os.mkdir(self.mixed_sidebyside_dir)

        self.bg_layout_dir = os.path.join(target_dir, "bg_layout")
        if not os.path.isdir(self.bg_layout_dir):
            os.mkdir(self.bg_layout_dir)

    def generate_masks_for_set(self, set_type, imgs_per_file=False):

        self.prepare_all_directories(set_type)

        batch_iterator = self.model.dataset.get_iterator('valid', shuffle=None, repeat=False, prefetch=1)
        counter = 0
        for batch in batch_iterator:

            (imgs, objs, boxes, masks, triples, attributes,
                objs_to_img, sketches, identifier, objs_bg, boxes_bg, masks_bg) = batch
            sketches = np.array(sketches, dtype=np.float32)
            attributes = attributes.numpy()
            boxes = boxes.numpy()

            # preprocess sketches and deprocess images
            img_s = self.model.dataset.image_size[0]
            sketch_comp = utils.layout.get_single_sketch_composition(sketches, boxes, img_s)
            # image = self.model.dataset.deprocess_image(np.array(imgs))[0]
            if self.hps['flipped_sketches']:
                flip_sketches = tf.image.flip_left_right(sketches).numpy()
                flip_sketch_comp = utils.layout.get_single_sketch_composition(flip_sketches, boxes, img_s)
            crops = utils.bbox.crop_bbox_batch(imgs, boxes, objs_to_img, self.model.dataset.hps['crop_size'])

            # save stuff
            filepath_jpg = os.path.join("{:012}.jpg".format(identifier[0]))
            filepath_png = filepath_jpg.replace("jpg", "png")
            skio.imsave(os.path.join(self.images_dir, filepath_jpg), (image * 255).astype(np.uint8))
            skio.imsave(os.path.join(self.sketches_dir, filepath_png), (sketch_comp * 255).astype(np.uint8))
            if self.hps['flipped_sketches']:
                skio.imsave(os.path.join(self.flip_sketches_dir, filepath_png), (flip_sketch_comp * 255).astype(np.uint8))

            # generate all of it
            # sketches
            img_skt_reps, obj_co_vecs = self.model.inference_mixed_representation(sketches, None, boxes)
            ggan_layout, inst_layout = self.model.inference_layout_generation(
                obj_co_vecs, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
            rgb_layout = self.model.inference_layout_generation(
                obj_co_vecs, objs, boxes_bg, masks_bg, objs_bg, rgb=True, gaugan=False)
            gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
            skio.imsave(os.path.join(self.gen_dir, filepath_jpg), (gen_image * 255).astype(np.uint8))
            skio.imsave(os.path.join(self.sidebyside_dir, filepath_jpg),
                        (np.concatenate([sketch_comp, gen_image], axis=1) * 255).astype(np.uint8))
            skio.imsave(os.path.join(self.sidebyside_layout_dir, filepath_jpg),
                        (np.concatenate([sketch_comp, rgb_layout], axis=1) * 255).astype(np.uint8))

            # images-based
            if self.hps['generate_from_image']:
                img_skt_reps, obj_co_vecs = self.model.inference_mixed_representation(None, crops, boxes)
                ggan_layout, inst_layout = self.model.inference_layout_generation(
                    obj_co_vecs, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
                img_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
                skio.imsave(os.path.join(self.img_gen_dir, filepath_jpg), (img_gen_image * 255).astype(np.uint8))

            # flipped sketches
            if self.hps['flipped_sketches']:
                img_skt_reps, obj_co_vecs = self.model.inference_mixed_representation(flip_sketches, None, boxes)
                ggan_layout, inst_layout = self.model.inference_layout_generation(
                    obj_co_vecs, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
                flip_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
                skio.imsave(os.path.join(self.flipped_gen_dir, filepath_jpg), (flip_gen_image * 255).astype(np.uint8))
                skio.imsave(os.path.join(self.flipped_sidebyside_dir, filepath_jpg),
                            (np.concatenate([sketch_comp, gen_image, flip_sketch_comp, flip_gen_image], axis=1) * 255).astype(np.uint8))

            # from ground truth layout
            if self.hps['generate_from_gt']:
                ggan_layout, inst_layout = self.model.make_layout(
                    boxes, masks, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
                gtlayout_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
                skio.imsave(os.path.join(self.gt_gen_dir, filepath_jpg), (gtlayout_gen_image * 255).astype(np.uint8))

            # save the bg layout
            bg_layout = self.model.make_bg_layout(boxes_bg, masks_bg, objs_bg, rgb=True)
            skio.imsave(os.path.join(self.bg_layout_dir, filepath_png), (bg_layout.numpy() * 255).astype(np.uint8))

            # mixed domains
            if self.hps['generate_from_mixed'] and len(objs) > 1:
                half_of_objs = len(boxes) // 2
                dep_crops = self.model.dataset.deprocess_image(np.array(crops))
                mixed_comp = utils.layout.get_single_sketch_composition_with_images(
                    dep_crops[half_of_objs:], sketches[:half_of_objs], boxes[half_of_objs:], boxes[:half_of_objs], img_s).numpy()
                skio.imsave(os.path.join(self.mixed_domain_dir, filepath_png), (mixed_comp * 255).astype(np.uint8))

                img_skt_reps, obj_co_vecs = self.model.inference_mixed_representation(sketches[:half_of_objs], crops[half_of_objs:], boxes)
                ggan_layout, inst_layout = self.model.inference_layout_generation(
                    obj_co_vecs, objs, boxes_bg, masks_bg, objs_bg, rgb=False, gaugan=True)
                mixed_gen_image = utils.gaugan.generate(ggan_layout, inst_layout)
                skio.imsave(os.path.join(self.mixed_gen_dir, filepath_jpg), (mixed_gen_image * 255).astype(np.uint8))
                skio.imsave(os.path.join(self.mixed_sidebyside_dir, filepath_jpg),
                            (np.concatenate([mixed_comp, mixed_gen_image], axis=1) * 255).astype(np.uint8))

            counter += 1
            print("[Processing] {} images done!".format(counter))

    def compute(self, model=None):
        self.model = model

        self.generate_masks_for_set('test')
        # self.generate_masks_for_set('valid')
