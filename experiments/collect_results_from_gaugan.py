import os
import glob

import numpy as np
import skimage.io as skio
import imageio

import utils
import metrics
from core.experiments import Experiment


class SaveGauganCompatibleLayoutsFromGraphBasedMaskGenerator(Experiment):
    name = "collect-gaugan-results"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='',
            images_dir='',
            exp_id=''
        )
        return hps

    def prepare_all_directories(self, set_type):
        self.sketch_label_dir = os.path.join(self.hps['results_dir'], "{}_skt/coco_pretrained/test_latest/images/input_label/".format(self.hps['exp_id']))
        self.sketch_results_dir = os.path.join(self.hps['results_dir'], "{}_skt/coco_pretrained/test_latest/images/synthesized_image".format(self.hps['exp_id']))
        self.image_label_dir = os.path.join(self.hps['results_dir'], "{}_img/coco_pretrained/test_latest/images/input_label/".format(self.hps['exp_id']))
        self.image_results_dir = os.path.join(self.hps['results_dir'], "{}_img/coco_pretrained/test_latest/images/synthesized_image".format(self.hps['exp_id']))
        self.gt_label_dir = os.path.join(self.hps['results_dir'], "{}_gt/coco_pretrained/test_latest/images/input_label/".format(self.hps['exp_id']))
        self.gt_results_dir = os.path.join(self.hps['results_dir'], "{}_gt/coco_pretrained/test_latest/images/synthesized_image".format(self.hps['exp_id']))
        self.images_dir = os.path.join(self.hps['images_dir'], "{}_images".format(set_type))
        self.sketches_dir = os.path.join(self.hps['images_dir'], "{}_sketches".format(set_type))

    def build_plot_for_set(self, set_type):
        self.prepare_all_directories(set_type)

        s_layouts, o_layouts, label_layouts, imgs, sketches, s_gimgs, o_gimgs, gimgs = [], [], [], [], [], [], [], []

        image_files = [os.path.basename(f) for f in glob.glob("{}/*".format(self.images_dir))]
        for jpg_file in image_files[:200]:
            png_file = jpg_file.replace("jpg", "png")
            s_layouts.append(imageio.imread(os.path.join(self.sketch_label_dir, png_file))/255.0)
            s_gimgs.append(imageio.imread(os.path.join(self.sketch_results_dir, png_file))/255.0)
            o_layouts.append(imageio.imread(os.path.join(self.image_label_dir, png_file))/255.0)
            o_gimgs.append(imageio.imread(os.path.join(self.image_results_dir, png_file))/255.0)
            label_layouts.append(imageio.imread(os.path.join(self.gt_label_dir, png_file))/255.0)
            gimgs.append(imageio.imread(os.path.join(self.gt_results_dir, png_file))/255.0)
            imgs.append(imageio.imread(os.path.join(self.images_dir, jpg_file))/255.0)
            sketches.append(imageio.imread(os.path.join(self.sketches_dir, jpg_file))/255.0)

        image_metric = metrics.build_metric_by_name('images-from-scene-graph-big-valid', self.hps.values())
        image_metric.computation_worker([np.array(imgs), np.array(gimgs), np.array(o_gimgs), np.array(sketches), np.array(s_gimgs)])

        layout_metric = metrics.build_metric_by_name('layout-from-scene-graph-big-valid', self.hps.values())
        layout_metric.computation_worker([np.array(label_layouts), np.array(o_layouts), np.array(s_layouts)])

        metrics_list = {'generated-images': image_metric, 'generated-layouts': layout_metric}
        self.model.plot_and_send_notification_for(metrics_list)
        self.model.clean_up_tmp_dir()

    def compute(self, model=None):
        self.model = model
        self.build_plot_for_set('valid')
        # self.build_plot_for_set('test')
