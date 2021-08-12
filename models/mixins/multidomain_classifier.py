import tensorflow as tf
import numpy as np

import utils
import dataloaders


class ObjCropClassifierMetricMixin(object):

    def compute_embedding_from_appearence_net_on_validation(self):
        features, all_y = [], []
        batch_iterator = self.dataset.get_n_samples_batch_iterator_from(
            'valid', n=2048, batch_size=self.hps['batch_size'], shuffled=True, seeded=True)
        for batch in batch_iterator:
            imgs, objs, boxes, masks = batch
            obj_encoding = self.forward(imgs, objs, masks)
            features.append(obj_encoding)
            all_y.append(objs)
        features = np.concatenate(features, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return features, all_y

    def compute_embedding_classification_predictions_on_validation_set(self):
        obj_ys, all_y = [], []
        batch_iterator = self.dataset.get_n_samples_batch_iterator_from(
            'valid', n=2048, batch_size=self.hps['batch_size'], shuffled=True, seeded=True)
        for batch in batch_iterator:
            imgs, objs, boxes, masks = batch
            obj_encoding = self.forward(imgs, objs, masks)
            obj_ys.append(self.embedding_classifier(obj_encoding))
            all_y.append(objs)
        obj_ys = np.concatenate(obj_ys, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return obj_ys, None, all_y, None

    def compute_data_sanity_check(self):
        all_imgs = []
        batch_iterator = self.dataset.get_n_samples_batch_iterator_from(
            'valid', n=2048, batch_size=self.hps['batch_size'], shuffled=True, seeded=True)
        chosen_obj = 0
        for batch in batch_iterator:
            imgs, objs, boxes, masks = batch
            for obj, img in zip(objs, imgs):
                if obj == chosen_obj:
                    all_imgs.append(img)

        return np.array([self.dataset.deprocess_image(x) for x in all_imgs])


class SketchClassifierMetricMixin(object):

    def compute_embedding_from_appearence_net_on_validation(self):
        features, all_y = [], []
        batch_iterator = self.dataset.get_n_samples_batch_iterator_from(
            'valid', n=2048, batch_size=self.hps['batch_size'], shuffled=True, seeded=True)
        for batch in batch_iterator:
            skts, y = batch
            skt_encoding = self.forward(skts)
            features.append(skt_encoding)
            all_y.append(y)
        features = np.concatenate(features, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return features, all_y

    def compute_embedding_classification_predictions_on_validation_set(self):
        obj_ys, all_y = [], []
        batch_iterator = self.dataset.get_n_samples_batch_iterator_from(
            'valid', n=2048, batch_size=self.hps['batch_size'], shuffled=True, seeded=True)
        for batch in batch_iterator:
            skts, y = batch
            skt_encoding = self.forward(skts)
            obj_ys.append(self.embedding_classifier(skt_encoding))
            all_y.append(y)
        obj_ys = np.concatenate(obj_ys, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return None, obj_ys, None, all_y

    def compute_data_sanity_check(self):
        all_skts = []
        batch_iterator = self.dataset.get_n_samples_batch_iterator_from(
            'valid', n=2048, batch_size=self.hps['batch_size'], shuffled=True, seeded=True)
        chosen_obj = 0
        for batch in batch_iterator:
            skts, ys = batch
            for obj, skt in zip(ys, skts):
                if obj == chosen_obj:
                    all_skts.append(skt)

        return np.array(all_skts)


class MultidomainRepresentationMetricMixin(object):

    def compute_common_embedding_from_appearence_net_on_validation(self):
        obj_feats, skt_feats, all_y, skt_y = [], [], [], []
        counter, n = 0, 2500
        for batch in self.dataset.get_iterator('valid', self.hps['batch_size'], shuffle=None, prefetch=None, repeat=False, image_size=self.hps['image_size']):
            sketches, p_crops, n_crops, p_class, n_class = batch
            obj_enc, obj_enc_n, skt_enc, skt_rep, obj_rep, obj_rep_n = self.forward(p_crops, n_crops, sketches)
            obj_feats.append(obj_rep)
            skt_feats.append(skt_rep)
            skt_y.append(p_class)
            all_y.append(p_class)
            counter += 1
            if counter % 5 == 0:
                print("Processed {} batches, {} samples".format(counter, self.hps['batch_size'] * counter))
            if self.hps['batch_size'] * counter >= n:
                break
        obj_feats = np.concatenate(obj_feats, axis=0)
        skt_feats = np.concatenate(skt_feats, axis=0)
        skt_y = np.concatenate(skt_y, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return obj_feats, skt_feats, all_y, skt_y

    def compute_embedding_classification_predictions_on_validation_set(self):
        obj_pred_ys, skt_pred_ys, all_y, skt_y = [], [], [], []
        counter, n = 0, 2500
        for batch in self.dataset.get_iterator('valid', self.hps['batch_size'], shuffle=None, prefetch=None, repeat=False, image_size=self.hps['image_size']):
            sketches, p_crops, n_crops, p_class, n_class = batch
            obj_enc, obj_enc_n, skt_enc, skt_rep, obj_rep, obj_rep_n = self.forward(p_crops, n_crops, sketches)
            obj_pred_ys.append(self.obj_encoder.embedding_classifier(obj_enc))
            skt_pred_ys.append(self.skt_encoder.embedding_classifier(skt_enc))
            skt_y.append(p_class)
            all_y.append(p_class)
            counter += 1
            if counter % 5 == 0:
                print("Processed {} batches, {} samples".format(counter, self.hps['batch_size'] * counter))
            if self.hps['batch_size'] * counter >= n:
                break
        obj_pred_ys = np.concatenate(obj_pred_ys, axis=0)
        skt_pred_ys = np.concatenate(skt_pred_ys, axis=0)
        skt_y = np.concatenate(skt_y, axis=0)
        all_y = np.concatenate(all_y, axis=0)

        return obj_pred_ys, skt_pred_ys, all_y, skt_y

    def compute_common_SBIR_on_validation_set(self):
        obj_feats, skt_feats, all_sketches, all_crops = [], [], [], []

        qdset = dataloaders.get_dataloader_by_name('indexed-quickdraw')()
        ccset = dataloaders.get_dataloader_by_name('cococrops-tfrecord')(self.hps)
        qditerators = qdset.get_all_class_iterators('valid', batch_size=50, shuffle=None, repeat=True)
        cciterators = ccset.get_all_class_iterators('valid', batch_size=1, shuffle=None, repeat=True)
        queries = 64
        for class_name, cciterator in cciterators.items():
            try:
                obj_crop, label = next(cciterator)
            except StopIteration:
                print("No crops for {}".format(class_name))
                continue
            obj_rep = self.common_project_and_norm(self.obj_encoder.encoder(obj_crop))

            sketches, s_labels = next(qditerators[class_name])
            skt_rep = self.common_project_and_norm(self.skt_encoder.encoder(sketches))

            obj_feats.append(obj_rep)
            skt_feats.append(skt_rep)
            all_sketches.append(sketches)
            all_crops.append(obj_crop)

        obj_feats = np.concatenate(obj_feats, axis=0)
        skt_feats = np.concatenate(skt_feats, axis=0)
        all_sketches = np.concatenate(all_sketches, axis=0)
        all_sketches = np.concatenate([all_sketches, all_sketches, all_sketches], axis=-1)
        all_crops = ccset.deprocess_image(np.concatenate(all_crops, axis=0))

        sel_sketches = []
        for i, obj_feat in enumerate(obj_feats):
            sketch_features_for_this_class = skt_feats[i*50:i*50+50]
            sketches_of_this_class = all_sketches[i*50:i*50+50]
            _, _, _, _, rank_mat = utils.sbir.simple_sbir(np.array([obj_feat]), sketch_features_for_this_class, return_mat=True)
            sel_sketches.append(sketches_of_this_class[rank_mat[:, :5]])
        sel_sketches = np.array(sel_sketches)

        # mAP, top1, top5, top10, rank_mat = utils.sbir.simple_sbir(obj_feats, skt_feats, return_mat=True)
        mAP, top1, top5, top10 = 0, 0, 0, 0
        return all_crops[:queries], sel_sketches, mAP, top1, top5, top10

    def compute_FGSBIR_on_test_set(self):
        obj_feats, skt_feats, all_sketches, all_crops = [], [], [], []
        sketchyset = dataloaders.get_dataloader_by_name('sketchy-extended-tf')()
        counter, n = 0, 2500
        for batch in sketchyset.get_iterator('valid', self.hps['batch_size'], shuffle=None, prefetch=5, repeat=False):
            sketches, p_crops, n_crops, p_class, n_class = batch
            obj_enc, obj_enc_n, skt_enc, skt_rep, obj_rep, obj_rep_n = self.forward(p_crops, n_crops, sketches)
            obj_feats.append(obj_rep)
            skt_feats.append(skt_rep)
            all_sketches.append(sketches)
            all_crops.append(p_crops)
            counter += 1
            if counter % 5 == 0:
                print("Processed {} batches, {} samples".format(counter, self.hps['batch_size'] * counter))
            if self.hps['batch_size'] * counter >= n:
                break
        obj_feats = np.concatenate(obj_feats, axis=0)
        skt_feats = np.concatenate(skt_feats, axis=0)
        all_sketches = np.concatenate(all_sketches, axis=0)
        all_sketches = np.concatenate([all_sketches, all_sketches, all_sketches], axis=-1)
        all_crops = sketchyset.deprocess_image(np.concatenate(all_crops, axis=0))

        mAP, top1, top5, top10, rank_mat = utils.sbir.simple_sbir(obj_feats, skt_feats, return_mat=True)
        return all_crops[:n], all_sketches[rank_mat[:n, 0:5]], mAP, top1, top5, top10
