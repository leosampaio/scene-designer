import tensorflow as tf
import numpy as np
import os
import skimage.transform as sk_transform
from scipy import stats as sk_stats
from concurrent.futures import ThreadPoolExecutor
import tensorflow_addons as tfa
from utils.sbir import sbir

import utils


class GraphAttentionRepresentationMetrics(object):

    def compute_common_embedding_from_appearence_net_on_validation(self, n=10000):

        obj_feats, skt_feats, all_y, skt_y = [], [], [], []
        batch_iterator = self.eval_iterators.get(n, self.dist_eval_dataset(split_name='valid', n=n, shuffled=True, seeded=True))
        self.eval_iterators[n] = batch_iterator
        counter = 0
        for batch in batch_iterator:
            (imgs, objs, boxes, masks, triples, attr, objs_to_img,
             sketches, n_crops, n_labels) = batch
            obj_enc, obj_encoded_n, skt_enc, skt_rep, obj_rep, obj_rep_n = self.strategy.experimental_run_v2(
                self.multidomain_encoder.forward, args=(obj_imgs, obj_imgs_n, sketches, masks, masks_n))

            obj_rep, skt_rep, objs = self.reduce_concat(obj_rep, skt_rep, objs)

            obj_feats.append(obj_rep)
            skt_feats.append(skt_rep)
            skt_y.append(objs)
            all_y.append(objs)
            counter += self.hps['batch_size'] * 4
            if counter > n:
                break
        obj_feats = np.concatenate(obj_feats, axis=0)
        skt_feats = np.concatenate(skt_feats, axis=0)
        skt_y = np.concatenate(skt_y, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return obj_feats, skt_feats, all_y, skt_y

    def compute_embedding_classification_predictions_on_validation_set(self, n=10000):
        obj_pred_ys, skt_pred_ys, all_y, skt_y = [], [], [], []
        batch_iterator = self.eval_iterators.get(n, self.dist_eval_dataset(split_name='valid', n=n, shuffled=True, seeded=True))
        self.eval_iterators[n] = batch_iterator
        counter = 0
        for batch in batch_iterator:
            (imgs, objs, boxes, masks, triples, attr, objs_to_img,
             sketches, n_crops, n_labels) = batch
            obj_enc, obj_encoded_n, skt_enc, skt_rep, obj_rep, obj_rep_n = self.strategy.experimental_run_v2(
                self.multidomain_encoder.forward, args=(obj_imgs, obj_imgs_n, sketches, masks, masks_n))

            obj_pred_y = self.strategy.experimental_run_v2(
                self.multidomain_encoder.obj_encoder.embedding_classifier, args=(obj_enc,))
            skt_pred_y = self.strategy.experimental_run_v2(
                self.multidomain_encoder.skt_encoder.embedding_classifier, args=(skt_enc,))

            obj_pred_y, skt_pred_y, objs = self.reduce_concat(obj_pred_y, skt_pred_y, objs)

            obj_pred_ys.append(obj_pred_y)
            skt_pred_ys.append(skt_pred_y)
            skt_y.append(objs)
            all_y.append(objs)
            counter += self.hps['batch_size'] * 4
            if counter > n:
                break
        obj_pred_ys = np.concatenate(obj_pred_ys, axis=0)
        skt_pred_ys = np.concatenate(skt_pred_ys, axis=0)
        skt_y = np.concatenate(skt_y, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return obj_pred_ys, skt_pred_ys, all_y, skt_y

    def compute_SBIR_on_validation_set(self):
        return self.compute_SBIR_on_set('valid')

    def compute_SBIR_on_big_validation_set(self):
        return self.compute_SBIR_on_set('valid')

    def compute_SBIR_on_test_set(self):
        return self.compute_SBIR_on_set('test')

    def compute_SBIR_on_set(self, set_type, queries=64):
        all_reps, all_skt_reps, all_skts, all_imgs = [], [], [], []
        for batch in self.dataset.get_iterator(set_type, self.hps['batch_size'], shuffle=None, repeat=False):
            imgs, objs, boxes, masks, triples, attr, objs_to_img, sketches, _, _, _ = batch
            img_reps, img_skt_reps, obj_co_vecs, skt_co_vecs = self.inference_representation(
                imgs, objs, triples, masks, boxes, attr, objs_to_img, sketches)

            img_s = self.dataset.image_size
            skt_comp = utils.layout.get_sketch_composition(sketches.numpy(), boxes, objs_to_img, img_s, img_s, self.hps['batch_size'])

            all_skts.append(np.concatenate((skt_comp, skt_comp, skt_comp), axis=-1))
            all_reps.append(img_reps)
            all_skt_reps.append(img_skt_reps)
            all_imgs.append(imgs)
        all_skts = np.concatenate(all_skts, axis=0)
        all_reps = np.concatenate(all_reps, axis=0)
        all_skt_reps = np.concatenate(all_skt_reps, axis=0)
        all_imgs = self.dataset.deprocess_image(np.concatenate(all_imgs, axis=0))

        mAP, top1, top5, top10, rank_mat = utils.sbir.simple_sbir(all_skt_reps, all_reps, return_mat=True)
        return all_skts[:queries], all_imgs[rank_mat[:queries, 0:5]], all_imgs[:queries], mAP, top1, top5, top10

    def gather_objs_attrs(self, objs, attrs, boxes, obj_to_img, n_imgs):
        objs_per_image = []
        attrs_per_image = []
        boxes_per_image = []
        for img_id in tf.range(n_imgs):
            objs_per_image.append(objs[obj_to_img == img_id])
            attrs_per_image.append(attrs[obj_to_img == img_id])
            boxes_per_image.append(boxes[obj_to_img == img_id])

        return objs_per_image, attrs_per_image, boxes_per_image

    def precompute_similarities(self):
        precomp_sims = []
        for i in range(25):
            cont_x_i, cont_y_i = (i % 5) * (1.0 / 5.0), (i // 5) * (1.0 / 5.0)
            precomp_sims_i = []
            for j in range(25):
                cont_x_j, cont_y_j = (j % 5) * (1.0 / 5.0), (j // 5) * (1.0 / 5.0)
                dist = np.sqrt(np.power(cont_x_i - cont_x_j, 2) + np.power(cont_y_i - cont_y_j, 2))
                sim = 1.0 / (1.0 + dist)
                precomp_sims_i.append(sim)
            precomp_sims.append(precomp_sims_i)
        return precomp_sims

    def compute_similarity_from_a_to_all(self, a, all_attrs, all_objs, all_bboxes, similarity_matrix, precomp_sims):
        attrs_a, objs_a, bboxes_a = all_attrs[a], all_objs[a], all_bboxes[a]
        for b in range(a, len(all_attrs)):
            attrs_b, objs_b, bboxes_b = all_attrs[b], all_objs[b], all_bboxes[b]
            obj_sims = []
            for attr_a, obj_a, bbox_a in zip(attrs_a, objs_a, bboxes_a):
                matching_sims = [0.]
                # center_index_a = np.argmax(attr_a[10:])
                for attr_b, obj_b, bbox_b in zip(attrs_b, objs_b, bboxes_b):
                    if obj_a == obj_b:
                        # center_index_b = np.argmax(attr_b[10:])
                        # matching_sims.append(precomp_sims[center_index_a][center_index_b])
                        matching_sims.append(2 - self.giou(bbox_a, bbox_b))
                obj_sims.append(np.amax(matching_sims))
            if len(attrs_a) < len(attrs_b):
                obj_sims += [0] * (len(attrs_b) - len(attrs_a))
            similarity_matrix[a, b] = np.mean(obj_sims)
            similarity_matrix[b, a] = np.mean(obj_sims)
        print("Computed Similarity for {}/{}".format(a, len(all_attrs)))

    def compute_similarity_matrix(self, attributes, objs, bboxes):
        similarity_matrix = np.zeros((len(objs), len(objs)), dtype=np.float32)
        precomp_sims = self.precompute_similarities()

        def map_fun(a):
            self.compute_similarity_from_a_to_all(a, attributes, objs, bboxes, similarity_matrix, precomp_sims)
        with ThreadPoolExecutor(max_workers=8) as executor:
            for a in range(len(objs)):
                executor.submit(map_fun, (a))
        return similarity_matrix

    def compute_mask_generation_on_validation_set(self):
        return self.compute_mask_generation_on_set('valid', n=8)

    def compute_mask_generation_on_big_validation_set(self):
        return self.compute_mask_generation_on_set('valid', n=64)

    def compute_mask_generation_on_test_set(self):
        return self.compute_mask_generation_on_set('test', n=8)

    def compute_mask_generation_on_set(self, set_type, n=8):
        all_masks, all_masks_pred, all_skts, all_masks_sketch = [], [], [], []
        for batch in self.dataset.get_iterator(set_type, self.hps['batch_size'], shuffle=None, repeat=False):
            imgs, objs, boxes, masks, triples, attr, objs_to_img, sketches, n_crops, n_labels, ids = batch

            out = self.forward(imgs, objs, triples, masks, boxes, attr, objs_to_img, sketches, n_crops)
            (img_reps, img_tri_reps, img_skt_reps, img_skt_tri_reps,
             obj_vecs, skt_obj_vecs, n_img_tri_reps, masks_pred, boxes_pred,
             skt_masks_pred, skt_boxes_pred) = out

            sketches, masks, masks_pred, skt_masks_pred = self.reduce_concat(sketches, masks, masks_pred, skt_masks_pred)

            all_masks.append(masks)
            all_masks_pred.append(masks_pred)
            all_masks_sketch.append(skt_masks_pred)
            all_skts.append(sketches)
        all_masks = tf.concat(all_masks, axis=0)
        all_masks_pred = tf.concat(all_masks_pred, axis=0)
        all_masks_sketch = tf.concat(all_masks_sketch, axis=0)
        all_skts = np.concatenate(all_skts, axis=0)

        all_masks = tf.expand_dims(all_masks, axis=-1)
        all_masks_pred = tf.expand_dims(all_masks_pred, axis=-1)
        all_masks_sketch = tf.expand_dims(all_masks_sketch, axis=-1)

        all_masks = tf.image.resize(all_masks, (self.dataset.sketch_size, self.dataset.sketch_size), method='nearest')
        all_masks_pred = tf.image.resize(all_masks_pred, (self.dataset.sketch_size, self.dataset.sketch_size), method='nearest')
        all_masks_sketch = tf.image.resize(all_masks_sketch, (self.dataset.sketch_size, self.dataset.sketch_size), method='nearest')

        all_masks = all_masks.numpy()
        all_masks_pred = all_masks_pred.numpy()
        all_masks_sketch = all_masks_sketch.numpy()

        return all_masks, all_masks_pred, all_skts, all_masks_sketch

    def compute_rep_SBIR_on_validation_set(self):
        return self.compute_rep_SBIR_on_set('valid')

    def compute_rep_SBIR_on_big_validation_set(self):
        return self.compute_rep_SBIR_on_set('valid')

    def compute_rep_SBIR_on_test_set(self):
        return self.compute_rep_SBIR_on_set('test')

    def compute_rep_SBIR_on_set(self, set_type, queries=64):
        all_objs, all_reps, all_skt_reps, all_skts, all_imgs = [], [], [], [], []
        for batch in self.dataset.get_iterator(set_type, 32, shuffle=None, repeat=False):
            imgs, objs, boxes, masks, triples, attr, objs_to_img, sketches, n_crops, n_labels, ids = batch

            g_crops = utils.bbox.crop_bbox_batch(imgs, boxes, objs_to_img, self.dataset.hps['crop_size'])
            out = self.multidomain_encoder.forward(g_crops, n_crops, sketches, masks, masks)
            t_obj_enc, t_obj_enc_n, t_skt_enc, t_skt_rep, t_obj_rep, t_obj_rep_n = out

            all_skts.append(np.concatenate((sketches, sketches, sketches), axis=-1))
            all_reps.append(t_obj_rep)
            all_skt_reps.append(t_skt_rep)
            all_imgs.append(g_crops)
            all_objs.append(objs)
        all_skts = np.concatenate(all_skts, axis=0)
        all_reps = np.concatenate(all_reps, axis=0)
        all_skt_reps = np.concatenate(all_skt_reps, axis=0)
        all_objs = np.concatenate(all_objs, axis=0)
        all_imgs = self.dataset.deprocess_image(np.concatenate(all_imgs, axis=0))

        mAP, top1, top5, top10, rank_mat = utils.sbir.simple_sbir(all_skt_reps, all_reps, return_mat=True)
        return all_skts[:queries], all_imgs[rank_mat[:queries, 0:5]], all_imgs[:queries], mAP, top1, top5, top10
