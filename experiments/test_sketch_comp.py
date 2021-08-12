import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio

import utils
import metrics
from core.experiments import Experiment


class SaveGauganCompatibleLayoutsFromGraphBasedMaskGenerator(Experiment):
    name = "test-sketch-comp"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            target_dir=''
        )
        return hps

    def gather_sketches_for(self, set_type, imgs_per_file=False):
        n_samples = 375

        self.img_counter_id = 0

        all_skts, all_nover_skts = [], []
        batch_iterator = self.model.dist_eval_dataset(split_name=set_type, stop_at_end_of_split=True)
        counter = 0
        for batch in batch_iterator:

            (imgs, objs, boxes, masks, triples, attr, objs_to_img,
             sketches, n_crops, n_labels, sims) = batch

            img_s = self.model.dataset.hps['image_size']
            batchs = self.model.hps['batch_size'] // self.model.strategy.num_replicas_in_sync if self.model.hps['distribute'] else self.model.hps['batch_size']

            def fun(skts, boxes, obj_to_img):
                return utils.layout.get_sketch_composition(skts.numpy(), boxes, obj_to_img, img_s, img_s, batchs)
            def fun_nonoverlapping(skts, boxes, obj_to_img):
                return utils.layout.get_sketch_composition(skts.numpy(), boxes, obj_to_img, img_s, img_s, batchs, avoid_overlap=True)
            skt_comp = self.model.reduce_lambda_concat(fun, sketches, boxes, objs_to_img)
            skt_comp_nover = self.model.reduce_lambda_concat(fun_nonoverlapping, sketches, boxes, objs_to_img)

            all_skts.append(np.concatenate((skt_comp, skt_comp, skt_comp), axis=-1))
            all_nover_skts.append(np.concatenate((skt_comp_nover, skt_comp_nover, skt_comp_nover), axis=-1))

            counter += self.model.hps['batch_size']
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break
        all_skts = np.concatenate(all_skts, axis=0)
        all_nover_skts = np.concatenate(all_nover_skts, axis=0)

        image_metric = metrics.build_metric_by_name('images-from-scene-graph-big-valid', self.hps.values())
        image_metric.computation_worker([np.array(all_skts[:125]), np.array(all_nover_skts[:125]), np.array(all_skts[125:250]), np.array(all_nover_skts[125:250]), np.array(all_nover_skts[250:375])])
        metrics_list = {'generated-images': image_metric}
        self.model.plot_and_send_notification_for(metrics_list)
        self.model.clean_up_tmp_dir()

    def compute(self, model=None):
        self.model = model
        # self.gather_sketches_for('valid')
        self.gather_sketches_for('test')
        # self.gather_sketches_for('train', imgs_per_file=3000)
