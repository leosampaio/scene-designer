import os
import glob

import numpy as np

import utils
import metrics
from core.experiments import Experiment


class ExtractMixedFeatures(Experiment):
    name = "extract-mixed-features"
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
        filename = os.path.join(self.hps['results_dir'], "{}_{}_{}_mixed_feats.npz".format(model_name, dataset_name, set_type))

        if set_type == 'train':
            n_samples = self.model.dataset.n_samples
        elif set_type == 'test':
            n_samples = self.model.dataset.n_test_samples
        elif set_type == 'valid':
            n_samples = self.model.dataset.n_valid_samples

        self.img_counter_id = 0

        if self.model.dataset.name == 'coco-scene-triplet-2graphs' or self.model.dataset.name == 'sketchycoco-2graphs':
            has_background = True
            all_objs_bg, all_masks_bg, all_boxes_bg = [], [], []
        else:
            has_background = False

        all_reps, all_skt_reps, all_imgs, all_objs, all_boxes = [], [], [], [], []
        batch_iterator = self.model.dataset.batch_iterator(set_type, 1, stop_at_end_of_split=True)
        counter = 0
        for batch in batch_iterator:
            if has_background:
                (imgs, objs_bg, boxes_bg, masks_bg, objs_to_img_bg, objs,
                 boxes, masks, triples, attr, objs_to_img,
                 sketches, n_crops, neg_labels, ids) = batch
            else:
                (imgs, objs, boxes, masks, triples, attr, objs_to_img,
                 sketches, n_crops, n_labels, sims) = batch
            sketches = np.array(sketches, dtype=np.float32)

            crops = utils.bbox.crop_bbox_batch(imgs, boxes, objs_to_img, self.model.dataset.hps['crop_size'])
            img_reps, _ = self.model.inference_mixed_representation(None, crops, boxes)

            half_of_objs = len(boxes) // 2
            img_skt_reps, _ = self.model.inference_mixed_representation(sketches[:half_of_objs], crops[half_of_objs:], boxes)

            if has_background:
                all_objs_bg.append(objs_bg)
                all_masks_bg.append(masks_bg)
                all_boxes_bg.append(boxes_bg)

            all_imgs.append(imgs)
            all_objs.append(objs)
            all_boxes.append(boxes)

            all_reps.append(img_reps)
            all_skt_reps.append(img_skt_reps)

            counter += 1
            print("[Processing] {}/{} images done!".format(counter, n_samples))
            if counter >= n_samples:
                break

        all_reps = np.concatenate(all_reps, axis=0)
        all_skt_reps = np.concatenate(all_skt_reps, axis=0)

        data = {'img_reps': all_reps[:n_samples],
                'skt_reps': all_skt_reps[:n_samples],
                'objs': all_objs,
                'boxes': all_boxes}
        if self.hps['save_images']:
            data['images'] = all_imgs[:n_samples]
        if has_background:
            data['bg_objs'] = all_objs_bg
            data['bg_masks'] = all_masks_bg
            data['bg_boxes'] = all_boxes_bg
        np.savez(filename, **data)

    def compute(self, model=None):
        self.model = model
        if not os.path.isdir(self.hps['results_dir']):
            os.mkdir(self.hps['results_dir'])

        self.gather_sketches_for('test')
        self.gather_sketches_for('valid')