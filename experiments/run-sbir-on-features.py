import os
import glob

import numpy as np
import skimage.io as skio

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


class SBIRonFeaturesFile(Experiment):
    name = "sbir-on-features-file"
    requires_model = False

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            featsfile='coco_scene_triplet_valid_feats.npz',
            save_images=False,
            save_images_two_plus=True,
            save_images_single=False,
            save_separate=False
        )
        return hps

    def compute(self, model=None):
        filename = os.path.join(self.hps['results_dir'], self.hps['featsfile'])
        data = np.load(filename, allow_pickle=True)

        mAP, top1, top5, top10 = utils.sbir.simple_sbir(data['skt_reps'], data['img_reps'])

        message = "mAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, self.hps['featsfile'])

        two_or_more = []
        for i, y in enumerate(data['objs']):
            if len(y) > 1:
                two_or_more.append(i)
        mAP, top1, top5, top10 = utils.sbir.simple_sbir(data['skt_reps'][two_or_more], data['img_reps'][two_or_more])
        message = "Images with 2+ objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            len(two_or_more), mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, self.hps['featsfile'])

        indexed_by_n_objs = {i: [] for i in range(1, 9)}
        for i, y in enumerate(data['objs']):
            indexed_by_n_objs[len(y)].append(i)

        for i in range(1, 8):
            if len(indexed_by_n_objs[i]) > 10:
                mAP, top1, top5, top10 = utils.sbir.simple_sbir(data['skt_reps'][indexed_by_n_objs[i]], data['img_reps'][indexed_by_n_objs[i]])
                message = "Only images with {} objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
                    i, len(indexed_by_n_objs[i]), mAP, top1, top5, top10)
                self.notifyier.notify_with_message(message, self.hps['featsfile'])

        if self.hps['save_images']:

            if self.hps['save_images_single']:
                skts_r, imgs_r, skts, imgs = data['skt_reps'][indexed_by_n_objs[1]], data['img_reps'][indexed_by_n_objs[1]], data['sketches'][indexed_by_n_objs[1]], data['images'][indexed_by_n_objs[1]]
                sel_top = 15
                images_dir = os.path.join(self.hps['results_dir'], "{}_single_sep_sbir".format(self.hps['featsfile'][:-4]))
                paint_correct = False
                dpi = 250
            elif not self.hps['save_images_two_plus']:
                skts_r, imgs_r, skts, imgs = data['skt_reps'], data['img_reps'], data['sketches'], data['images']
                images_dir = os.path.join(self.hps['results_dir'], "{}_sbir".format(self.hps['featsfile'][:-4]))
                sel_top = 5
                paint_correct = True
                dpi = 150
            else:
                skts_r, imgs_r, skts, imgs = data['skt_reps'][two_or_more], data['img_reps'][two_or_more], data['sketches'][two_or_more], data['images'][two_or_more]
                images_dir = os.path.join(self.hps['results_dir'], "{}_sbir".format(self.hps['featsfile'][:-4]))
                sel_top = 5
                paint_correct = True
                dpi = 150

            if not os.path.isdir(images_dir):
                os.mkdir(images_dir)

            mAP, top1, top5, top10, rank_mat = utils.sbir.simple_sbir(skts_r, imgs_r, return_mat=True)
            correct = [0.535, 0.980, 0.307]
            for i, (sketch, images) in enumerate(zip(skts, imgs[rank_mat[..., 0:sel_top], ...])):
                outfile = os.path.join(images_dir, "{:06}.png".format(i))
                plt.axis('off')
                images_strip = np.concatenate(images, axis=1)
                correct_image = np.where(rank_mat[i, 0:sel_top] == i)[0]
                if len(correct_image) == 1 and paint_correct:
                    x = correct_image[0]
                    images_strip[:, 256 * x:(256 * x + 15), :] = correct
                    images_strip[:, (256 * (x + 1) - 15):(256 * (x + 1)), :] = correct
                    images_strip[0:15, (256 * x):(256 * (x + 1)), :] = correct
                    images_strip[-15:256, (256 * x):(256 * (x + 1)), :] = correct
                if self.hps['save_separate']:
                    skio.imsave(os.path.join(images_dir, "{:06}_query.png".format(i)), (sketch * 255).astype(np.uint8))
                    for j, image in enumerate(images):
                        skio.imsave(os.path.join(images_dir, "{:06}_{:02}.png".format(i, j)), (image * 255).astype(np.uint8))
                else:
                    plt.imshow(np.concatenate([sketch, images_strip], axis=1))
                    plt.savefig(outfile, dpi=dpi, bbox_inches='tight')
                if i % 20 == 0:
                    print("Did save top {} for {}/{} images".format(sel_top, i, len(imgs_r)))
