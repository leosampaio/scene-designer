import os
import glob

import numpy as np

import utils
import metrics
from core.experiments import Experiment


class SBIRonCAGFeaturesFileWithExtSet(Experiment):
    name = "sbir-on-cag-features-extended-set"
    requires_model = False

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup_orig/scratch_4weeks/tb0035/projects/scene_sketch/',
            dataset='sketchycoco',
            labelsdir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            labelsfile='44b_sketchycoco_valid_feats.npz',
            save_images=False,
            save_images_two_plus=True,
            save_images_single=False,
            extended_set='/vol/vssp/cvpnobackup_orig/scratch_4weeks/tb0035/projects/scene_sketch/sketchycoco/extended_image.npy',
        )
        return hps

    def compute(self, model=None):
        labels_filename = os.path.join(self.hps['labelsdir'], self.hps['labelsfile'])
        if self.hps['dataset'] == 'sketchycoco':
            split = 'valid'
        else:
            split = 'test'

        skt_filename = os.path.join(self.hps['results_dir'], self.hps['dataset'], '{}_sketch.npy'.format(split))
        img_filename = os.path.join(self.hps['results_dir'], self.hps['dataset'], '{}_image.npy'.format(split))
        skt_reps = np.load(skt_filename, allow_pickle=True)
        img_reps = np.load(img_filename, allow_pickle=True)
        data = np.load(labels_filename, allow_pickle=True)
        labels = data['objs']
        extended_reps = np.load(self.hps['extended_set'])

        if self.hps['dataset'] == 'qdcoco':
            skt_reps = skt_reps[::5]

        mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps, np.concatenate([img_reps, extended_reps]))

        message = "mAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            mAP, top1, top5, top10)
        exp_id = "{}_CAG_extended".format(self.hps['dataset'])
        self.notifyier.notify_with_message(message, exp_id)

        two_or_more = []
        for i, y in enumerate(labels):
            if len(y) > 1:
                two_or_more.append(i)
        mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps[two_or_more], np.concatenate([img_reps[two_or_more], extended_reps]))
        message = "Images with 2+ objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            len(two_or_more), mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, exp_id)