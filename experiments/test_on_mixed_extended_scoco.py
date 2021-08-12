import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio

import utils
from core.experiments import Experiment


class TestMixedExtendedCOCO(Experiment):
    name = "test-mixed-extended-scoco"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            extended_set='',
        )
        return hps

    def test_extended_scoco(self):
        self.img_counter_id = 0

        all_skt_reps, all_reps = [], []

        e_imgs = self.extra_images['imgs']
        e_objs = self.extra_images['objs']
        e_masks = self.extra_images['masks']
        e_boxes = self.extra_images['boxes']
        e_triples = self.extra_images['triples']
        e_attributes = self.extra_images['attributes']

        all_reps, all_skt_reps = [], []
        batch_iterator = self.model.dataset.batch_iterator('valid', 1, stop_at_end_of_split=True)
        counter = 0
        for batch in batch_iterator:
            (imgs, objs, boxes, masks, triples, attr, objs_to_img, sketches, _, _, _) = batch
            sketches = np.array(sketches, dtype=np.float32)

            crops = utils.bbox.crop_bbox_batch(imgs, boxes, objs_to_img, self.model.dataset.hps['crop_size'])
            img_reps, _ = self.model.inference_mixed_representation(None, crops, boxes)

            half_of_objs = len(boxes) // 2
            img_skt_reps, _ = self.model.inference_mixed_representation(sketches[:half_of_objs], crops[half_of_objs:], boxes)

            all_reps.append(img_reps)
            all_skt_reps.append(img_skt_reps)

            counter += 1
            print("[Processing] {}/{} images done!".format(counter, self.model.dataset.n_valid_samples))
            if counter >= self.model.dataset.n_valid_samples:
                break

        all_reps = np.concatenate(all_reps, axis=0)[:self.model.dataset.n_valid_samples]
        all_skt_reps = np.concatenate(all_skt_reps, axis=0)[:self.model.dataset.n_valid_samples]

        counter = 0
        ext_reps, ext_images = [], []
        for img, objs, boxes, masks, triples, attr in zip(e_imgs, e_objs, e_boxes, e_masks, e_triples, e_attributes):
            objs_to_img = np.zeros(len(objs), dtype=np.int32)
            img = (np.expand_dims(img.astype(np.float32), axis=0) - 0.5) * 2
            triples = tf.convert_to_tensor(triples, dtype=tf.int32)
            attr = tf.convert_to_tensor(attr, dtype=tf.int32)

            img_reps, _ = self.model.inference_image_representation(imgs, triples, boxes, attr, objs_to_img)

            ext_reps.append(img_reps)
            ext_images.append(img)

            counter += 1
            print("[Processing] {}/{} images done!".format(counter, 5000))

        ext_reps = np.concatenate(ext_reps, axis=0)
        all_reps = np.concatenate([all_reps, ext_reps], axis=0)

        mAP, top1, top5, top10, recall = utils.sbir.simple_sbir(all_skt_reps, all_reps, return_recall=True)

        message = "mAP: {} \nrecall@1: {} \nrecall@5: {} \nrecall@10: {} \n".format(
            mAP, top1, top5, top10)
        print(message)
        print("recall@50: {}".format(recall[49]))
        print("recall@100: {}".format(recall[99]))

    def compute(self, model=None):
        self.model = model
        self.extra_images = np.load(self.hps['extended_set'], allow_pickle=True)

        self.test_extended_scoco()
