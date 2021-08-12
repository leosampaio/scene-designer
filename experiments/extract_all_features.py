import os
import glob

import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE

from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import ticker
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

import utils
import metrics
from core.experiments import Experiment


class ExtractFeatures(Experiment):
    name = "extract-features"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            save_images=False,
        )
        return hps

    def gather_sketches_for(self, set_type):
        dataset_name = self.model.dataset.name.replace('-', '_')
        model_name = self.model.experiment_id.replace('-', '_')
        filename = os.path.join(self.hps['results_dir'], "{}_{}_{}_feats.npz".format(model_name, dataset_name, set_type))

        if set_type == 'train':
            n_samples = self.model.dataset.n_samples
        elif set_type == 'test':
            n_samples = self.model.dataset.n_test_samples
        elif set_type == 'valid':
            n_samples = self.model.dataset.n_valid_samples

        if self.model.dataset.name == 'coco-scene-triplet-2graphs' or self.model.dataset.name == 'sketchycoco-2graphs':
            has_background = True
            all_objs_bg, all_masks_bg, all_boxes_bg = [], [], []
        else:
            has_background = False

        self.img_counter_id = 0

        all_skts, all_imgs, all_reps, all_skt_reps, all_objs, all_ovecs, all_svecs = [], [], [], [], [], [], []
        all_masks, all_boxes = [], []
        batch_iterator = self.model.dist_eval_dataset(split_name=set_type, stop_at_end_of_split=False)
        counter = 0
        for batch in batch_iterator:
            if has_background:
                (imgs, objs_bg, boxes_bg, masks_bg, objs_to_img_bg, objs,
                 boxes, masks, triples, attr, objs_to_img,
                 sketches, n_crops, neg_labels, ids) = batch
            else:
                (imgs, objs, boxes, masks, triples, attr, objs_to_img,
                 sketches, n_crops, n_labels, sims) = batch

            img_s = self.model.dataset.hps['image_size']
            batchs = self.model.hps['batch_size']

            skt_comp = utils.layout.get_sketch_composition(
                sketches.numpy(), boxes.numpy(), objs_to_img, img_s, img_s, batchs, avoid_overlap=True)

            img_reps, img_skt_reps, obj_co_vecs, skt_co_vecs = self.model.inference_representation(
                imgs, objs, triples, masks, boxes, attr, objs_to_img, sketches)

            (objs_per_image, boxes_per_image, masks_per_image, o_vecs_per_image, s_vecs_per_image) = self.gather_objs_attrs(
                objs, boxes, masks, obj_co_vecs, skt_co_vecs, objs_to_img, imgs.shape[0])

            if has_background:
                objs_bg_per_image, boxes_bg_per_image, masks_bg_per_image = self.gather_bg_objs_attrs(
                    objs_bg, boxes_bg, masks_bg, objs_to_img_bg, imgs.shape[0])
                all_objs_bg += objs_bg_per_image
                all_masks_bg += masks_bg_per_image
                all_boxes_bg += boxes_bg_per_image

            all_skts.append(np.concatenate((skt_comp, skt_comp, skt_comp), axis=-1))
            all_reps.append(img_reps)
            all_skt_reps.append(img_skt_reps)
            all_imgs.append(imgs)
            all_objs += objs_per_image
            all_masks += masks_per_image
            all_boxes += boxes_per_image
            all_ovecs += o_vecs_per_image
            all_svecs += s_vecs_per_image

            counter += self.model.hps['batch_size']
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break

        all_skts = np.concatenate(all_skts, axis=0)
        all_reps = np.concatenate(all_reps, axis=0)
        all_skt_reps = np.concatenate(all_skt_reps, axis=0)
        all_imgs = self.model.dataset.deprocess_image(np.concatenate(all_imgs, axis=0))

        data = {'img_reps': all_reps[:n_samples],
                'skt_reps': all_skt_reps[:n_samples],
                'objs': all_objs[:n_samples],
                'masks': all_masks[:n_samples],
                'boxes': all_boxes[:n_samples]}
        if self.hps['save_images']:
            data['images'] = all_imgs[:n_samples]
            data['sketches'] = all_skts[:n_samples]
        if has_background:
            data['bg_objs'] = all_objs_bg
            data['bg_masks'] = all_masks_bg
            data['bg_boxes'] = all_boxes_bg
        np.savez(filename, **data)

    def gather_objs_attrs(self, objs, boxes, masks, obj_co_vecs, skt_co_vecs, obj_to_img, n_imgs):
        objs_per_image = []
        boxes_per_image = []
        masks_per_image = []
        o_vecs_per_image = []
        s_vecs_per_image = []
        for img_id in tf.range(n_imgs):

            boxes_gt, masks_gt = boxes[obj_to_img == img_id], masks[obj_to_img == img_id]
            objs_gt = objs[obj_to_img == img_id]
            ovecs, svecs = obj_co_vecs[obj_to_img == img_id], skt_co_vecs[obj_to_img == img_id]

            objs_per_image.append(objs_gt.numpy())
            boxes_per_image.append(boxes_gt.numpy())
            masks_per_image.append(masks_gt.numpy())
            o_vecs_per_image.append(ovecs.numpy())
            s_vecs_per_image.append(svecs.numpy())

        return (objs_per_image, boxes_per_image, masks_per_image, o_vecs_per_image, s_vecs_per_image)

    def gather_bg_objs_attrs(self, objs, boxes, masks, obj_to_img, n_imgs):
        objs_per_image = []
        boxes_per_image = []
        masks_per_image = []
        for img_id in tf.range(n_imgs):
            boxes_gt, masks_gt = boxes[obj_to_img == img_id], masks[obj_to_img == img_id]
            objs_gt = objs[obj_to_img == img_id]
            objs_per_image.append(objs_gt.numpy())
            boxes_per_image.append(boxes_gt.numpy())
            masks_per_image.append(masks_gt.numpy())

        return objs_per_image, boxes_per_image, masks_per_image

    def compute(self, model=None):
        self.model = model
        if not os.path.isdir(self.hps['results_dir']):
            os.mkdir(self.hps['results_dir'])

        self.gather_sketches_for('valid')
        self.gather_sketches_for('test')
