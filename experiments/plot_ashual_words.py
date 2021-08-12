import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio
import json
import matplotlib.pyplot as plt

import utils
from core.experiments import Experiment


class PlotWordMaps(Experiment):
    name = "plot-ashual-words"
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

        self.ashual_words_dir = os.path.join(target_dir, "ashual_words")
        if not os.path.isdir(self.ashual_words_dir):
            os.mkdir(self.ashual_words_dir)

    def generate_masks_for_set(self, set_type, imgs_per_file=False):
        if set_type == 'train':
            n_samples = self.model.dataset.n_samples
        elif set_type == 'test':
            n_samples = self.model.dataset.n_test_samples
        elif set_type == 'valid':
            n_samples = self.model.dataset.n_valid_samples

        self.prepare_all_directories(set_type)

        font_dict = [12 + 3 * i for i in range(10)]
        batch_iterator = self.model.dataset.batch_iterator(set_type, 1, stop_at_end_of_split=True)
        counter = 0
        for batch in batch_iterator:

            (imgs, objs_bg, boxes_bg, masks_bg, objs_to_img_bg, objs,
             boxes, masks, triples, attr, objs_to_img,
             sketches, n_crops, neg_labels, identifier) = batch

            filepath_png = os.path.join("{:012}.png".format(identifier[0]))

            plt.figure(figsize=(8, 8))
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            plt.tick_params(top=False, bottom=False, left=False, right=False,
                            labelleft=False, labelbottom=False)

            objs = np.concatenate([objs_bg[0], objs[0]])
            boxes = np.concatenate([boxes_bg, boxes])

            plt.grid(color='black', lw=0.2)
            for obj, box in zip(objs, boxes):
                category = self.model.dataset.obj_idx_to_name[self.model.dataset.obj_idx_list[obj]]
                if category != '__image__':
                    center_x = (box[0] + box[2]) / 2
                    center_y = 1 - (box[1] + box[3]) / 2
                    w, h = box[2] - box[0], box[3] - box[1]
                    size = int(round((9) * (w * h)))
                    plt.text(
                        center_x, center_y,
                        category,
                        fontsize=font_dict[size],
                        horizontalalignment='center',
                        verticalalignment='center')

            plt.savefig(os.path.join(self.ashual_words_dir, filepath_png), bbox_inches='tight')
            plt.clf()

            counter += 1
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break

    def compute(self, model=None):
        self.model = model

        self.generate_masks_for_set('test')
        self.generate_masks_for_set('valid')
