import os
import glob

import numpy as np
import tensorflow as tf

import utils
import metrics
from core.experiments import Experiment


class SBIRonFlippedFeaturesFile(Experiment):
    name = "sbir-on-flipped-features-file"
    requires_model = False

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            featsfile='coco_scene_triplet_valid_feats.npz',
            flip_featsfile='',
            save_images=False,
            save_images_two_plus=True,
        )
        return hps

    def compute(self, model=None):
        filename = os.path.join(self.hps['results_dir'], self.hps['featsfile'])
        flipped_filename = os.path.join(self.hps['results_dir'], self.hps['flip_featsfile'])
        data = np.load(filename, allow_pickle=True)
        flipped_data = np.load(flipped_filename, allow_pickle=True)

        # we want to know:
        # how well the general performance is with all the flipped data
        # how it performs on flipped data only
        # how many times the unflipped shows up before the flipped?

        # mixed
        mixed_imgs = np.concatenate([flipped_data['img_reps'], data['img_reps']], axis=0)
        mAP, top1, top5, top10 = utils.sbir.simple_sbir(flipped_data['skt_reps'], mixed_imgs)

        self.notifyier.notify_with_message("*Mixed Image Representations*", self.hps['flip_featsfile'])
        message = "mAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, self.hps['featsfile'])

        two_or_more = []
        for i, y in enumerate(data['objs']):
            if len(y) > 1:
                two_or_more.append(i)
        mixed_imgs = np.concatenate([flipped_data['img_reps'][two_or_more], data['img_reps'][two_or_more]], axis=0)
        mAP, top1, top5, top10 = utils.sbir.simple_sbir(flipped_data['skt_reps'][two_or_more], mixed_imgs)
        message = "Images with 2+ objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            len(two_or_more), mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, self.hps['featsfile'])

        indexed_by_n_objs = {i: [] for i in range(1, 9)}
        for i, y in enumerate(data['objs']):
            indexed_by_n_objs[len(y)].append(i)

        for i in range(1, 8):
            if len(indexed_by_n_objs[i]) > 10:
                mixed_imgs = np.concatenate([flipped_data['img_reps'][indexed_by_n_objs[i]], data['img_reps'][indexed_by_n_objs[i]]], axis=0)
                mAP, top1, top5, top10 = utils.sbir.simple_sbir(flipped_data['skt_reps'][indexed_by_n_objs[i]], mixed_imgs)
                message = "Only images with {} objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
                    i, len(indexed_by_n_objs[i]), mAP, top1, top5, top10)
                self.notifyier.notify_with_message(message, self.hps['featsfile'])

        if self.hps['save_images']:
            images_dir = os.path.join(self.hps['results_dir'], "{}_sbir".format(self.hps['flip_featsfile'][:-4]))
            if not os.path.isdir(images_dir):
                os.mkdir(images_dir)

            if not self.hps['save_images_two_plus']:
                skts_r, imgs_r, skts, imgs, flip_img_r = flipped_data['skt_reps'], data['img_reps'], flipped_data['sketches'], data['images'], flipped_data['img_reps']
            else:
                skts_r, imgs_r, skts, imgs, flip_img_r = data['skt_reps'][two_or_more], data['img_reps'][two_or_more], flipped_data['sketches'][two_or_more], data['images'][two_or_more], flipped_data['img_reps'][two_or_more]

            all_img_reps = np.concatenate([flip_img_r, imgs_r], axis=0)
            all_images = np.concatenate([tf.image.flip_left_right(imgs), imgs], axis=0)
            mAP, top1, top5, top10, rank_mat = utils.sbir.simple_sbir(skts_r, all_img_reps, return_mat=True)
            correct = [0.92941176, 0.45098039, 0.4627451]
            correct_darker = [0.85941176, 0.37098039, 0.3927451]
            for i, (sketch, images) in enumerate(zip(skts, all_images[rank_mat[..., 0:5], ...])):
                outfile = os.path.join(images_dir, "{:06}.png".format(i))
                plt.axis('off')
                images_strip = np.concatenate(images, axis=1)
                correct_image = np.where(rank_mat[i, 0:5] == i)[0]
                if len(correct_image) == 1:
                    x = correct_image[0]
                    images_strip[:, 256 * x:(256 * x + 10), :] = correct
                    images_strip[:, (256 * (x + 1) - 10):(256 * (x + 1)), :] = correct
                    images_strip[0:10, (256 * x):(256 * (x + 1)), :] = correct
                    images_strip[-10:256, (256 * x):(256 * (x + 1)), :] = correct
                    images_strip[-5:256, (256 * x):(256 * (x + 1)), :] = correct_darker
                plt.imshow(np.concatenate([sketch, images_strip], axis=1))
                plt.savefig(outfile, dpi=150, bbox_inches='tight')
                if i % 20 == 0:
                    print("Did save top 5 for {}/{} images".format(i, len(imgs_r)))

        # only on flipped images
        self.notifyier.notify_with_message("*Flipped Image Representations*", self.hps['flip_featsfile'])
        mAP, top1, top5, top10 = utils.sbir.simple_sbir(flipped_data['skt_reps'], flipped_data['img_reps'])
        message = "mAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, self.hps['flip_featsfile'])

        two_or_more = []
        for i, y in enumerate(data['objs']):
            if len(y) > 1:
                two_or_more.append(i)
        mAP, top1, top5, top10 = utils.sbir.simple_sbir(flipped_data['skt_reps'][two_or_more], flipped_data['img_reps'][two_or_more])
        message = "Images with 2+ objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            len(two_or_more), mAP, top1, top5, top10)
        self.notifyier.notify_with_message(message, self.hps['flip_featsfile'])

        indexed_by_n_objs = {i: [] for i in range(1, 9)}
        for i, y in enumerate(flipped_data['objs']):
            indexed_by_n_objs[len(y)].append(i)

        for i in range(1, 8):
            if len(indexed_by_n_objs[i]) > 10:
                mAP, top1, top5, top10 = utils.sbir.simple_sbir(flipped_data['skt_reps'][indexed_by_n_objs[i]], flipped_data['img_reps'][indexed_by_n_objs[i]])
                message = "Only images with {} objs ({}): \nmAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
                    i, len(indexed_by_n_objs[i]), mAP, top1, top5, top10)
                self.notifyier.notify_with_message(message, self.hps['flip_featsfile'])
