import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio

import utils
import metrics
from core.experiments import Experiment


class SBIRonSketchyFeaturesFile(Experiment):
    name = "sbir-on-sketchy-features"
    requires_model = False

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            dataset='sketchycoco',
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            labelsfile='44b_sketchycoco_test_feats.npz',
            sketchfile='sketchy_sketchycoco_test_feats_sketches.npz',
            imagefile='sketchy_sketchycoco_test_feats_images.npz',
            ext_set='',
            save_images=False,
            save_images_two_plus=True,
            save_images_single=False,
            extended_set='',
            save_separate=False,
        )
        return hps

    def compute(self, model=None):
        labels_filename = os.path.join(self.hps['results_dir'], self.hps['labelsfile'])
        skt_filename = os.path.join(self.hps['results_dir'], self.hps['sketchfile'])
        img_filename = os.path.join(self.hps['results_dir'], self.hps['imagefile'])
        skt_reps = np.load(skt_filename, allow_pickle=True)['arr_0']
        img_reps = np.load(img_filename, allow_pickle=True)['arr_0']
        data = np.load(labels_filename, allow_pickle=True)
        labels = data['objs']

        if self.hps['ext_set'] != '':
            ext_filename = os.path.join(self.hps['results_dir'], self.hps['ext_set'])
            ext_reps = np.load(ext_filename, allow_pickle=True)['arr_0']
        else:
            ext_reps = None

        if ext_reps is not None:
            mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps, np.concatenate([img_reps, ext_reps]))
        else:
            mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps, img_reps)

        message = "mAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            mAP, top1, top5, top10)
        exp_id = self.hps['sketchfile']
        self.notifyier.notify_with_message(message, exp_id)

        two_or_more = []
        for i, y in enumerate(labels):
            if len(y) > 1:
                two_or_more.append(i)
        if ext_reps is not None:
            mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps[two_or_more], np.concatenate([img_reps[two_or_more], ext_reps]))
        else:
            mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps[two_or_more], img_reps[two_or_more])
        message = "Images with 2+ objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            len(two_or_more), mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, exp_id)

        indexed_by_n_objs = {i: [] for i in range(1, 9)}
        for i, y in enumerate(labels):
            indexed_by_n_objs[len(y)].append(i)

        for i in range(1, 8):
            if len(indexed_by_n_objs[i]) > 10:
                mAP, top1, top5, top10 = utils.sbir.simple_sbir(skt_reps[indexed_by_n_objs[i]], img_reps[indexed_by_n_objs[i]])
                message = "Only images with {} objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
                    i, len(indexed_by_n_objs[i]), mAP, top1, top5, top10)
                self.notifyier.notify_with_message(message, exp_id)

        if self.hps['save_images']:

            if self.hps['save_images_single']:
                skts_r, imgs_r, skts, imgs = skt_reps[indexed_by_n_objs[1]], img_reps[
                    indexed_by_n_objs[1]], data['sketches'][indexed_by_n_objs[1]], data['images'][indexed_by_n_objs[1]]
                sel_top = 15
                images_dir = os.path.join(self.hps['results_dir'], "{}_single_sbir".format(exp_id))
                paint_correct = False
                dpi = 250
            elif not self.hps['save_images_two_plus']:
                skts_r, imgs_r, skts, imgs = skt_reps, img_reps, data['sketches'], data['images']
                images_dir = os.path.join(self.hps['results_dir'], "{}_sbir".format(exp_id))
                sel_top = 5
                paint_correct = True
                dpi = 150
            else:
                skts_r, imgs_r, skts, imgs = skt_reps[two_or_more], img_reps[two_or_more], data['sketches'][two_or_more], data['images'][two_or_more]
                images_dir = os.path.join(self.hps['results_dir'], "{}_sbir".format(exp_id))
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
