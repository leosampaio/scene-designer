import os

import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import skimage.transform as sk_transform
from sklearn.cluster import KMeans
from PIL import Image

from core.metrics import ProjectionMetric, ImageMetric


class TSNEProjection(ProjectionMetric):
    name = 'tsne'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        np.random.seed()

        # [194, 103, 317, 100, 112, 221, 223, 293, 239, 8],
        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        y, pred_z = y[idx], pred_z[idx]
        for label, feature in zip(y, pred_z):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                feats.append(feature)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        tsne_results = tsne.fit_transform(feats)
        return np.concatenate((tsne_results, labels),
                              axis=1)


class TSNEImagesProjection(ImageMetric):
    name = 'tsne-images'
    input_type = 'features_and_images'

    def compute(self, input_data):
        feats, images = input_data

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        tsne_results = tsne.fit_transform(feats)
        tx, ty = tsne_results[:, 0], tsne_results[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
        tsne_results[:, 0], tsne_results[:, 1] = tx, ty

        width = 4000
        height = 4000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(images, tx, ty):
            tile = img
            rs = max(1, tile.shape[0] / max_dim, tile.shape[1] / max_dim)
            tile = sk_transform.resize(tile, (int(tile.shape[0] / rs), int(tile.shape[1] / rs)), anti_aliasing=True)
            tile = Image.fromarray(np.uint8(tile * 255))
            full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

        image_plot_file = '/tmp/tsne-image-plot.png'
        full_image.save(image_plot_file)

        return image_plot_file


class EmbeddingTSNEProjection(ProjectionMetric):
    name = 'embedding-tsne'
    input_type = 'embedding_from_appearence_net_on_validation'

    def compute(self, input_data):
        emb, y = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        np.random.seed()

        sel_x, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        y = y[idx]
        emb = emb[idx]
        for feat, label in zip(emb, y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                sel_x.append(feat)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=5000,
                    random_state=14)
        tsne_results = tsne.fit_transform(sel_x)
        return np.concatenate((tsne_results, labels),
                              axis=1)

class EmbeddingPCAProjection(ProjectionMetric):
    name = 'embedding-pca'
    input_type = 'embedding_from_appearence_net_on_validation'

    def compute(self, input_data):
        emb, y = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        np.random.seed()

        sel_x, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        y = y[idx]
        emb = emb[idx]
        for feat, label in zip(emb, y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                sel_x.append(feat)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        pca = PCA(n_components=2)
        pca.fit(sel_x)
        pca_result = pca.transform(sel_x)
        return np.concatenate((pca_result, labels),
                              axis=1)


class EmbeddingsFromAppearenceNetTSNEProjection(ProjectionMetric):
    name = 'common-embedding-tsne'
    input_type = 'common_embedding_from_appearence_net_on_validation'
    plot_type = 'scatter-with-shapes'

    def compute(self, input_data):
        obj_emb, sketch_emb, y, skt_y = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        skt_idx = np.random.permutation(len(skt_y))
        np.random.seed()

        sel_objs, sel_skts, labels, skt_labels, sel_labels, counter, label_counter = [], [], [], [], [], 0, 0
        y = y[idx]
        obj_emb = obj_emb[idx]
        skt_y = skt_y[skt_idx]
        sketch_emb = sketch_emb[skt_idx]
        for skt, label in zip(sketch_emb, skt_y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                sel_skts.append(skt)
                skt_labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        counter = 0
        for obj, label in zip(obj_emb, y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                sel_objs.append(obj)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        combined_embs = np.concatenate((sel_objs, sel_skts), axis=0)
        print(np.array(obj_emb).shape, np.array(sketch_emb).shape, np.array(sel_objs).shape, np.array(sel_skts).shape, np.array(combined_embs).shape,  np.array(labels).shape)
        tsne_results = tsne.fit_transform(combined_embs)
        objs_tsne, skt_tsne = tsne_results[:len(sel_objs)], tsne_results[len(sel_objs):]
        return np.array([np.concatenate((objs_tsne, np.expand_dims(labels, -1)),
                                        axis=1),
                         np.concatenate((skt_tsne, np.expand_dims(skt_labels, -1)),
                                        axis=1)])


class EmbeddingsFromTripletTSNEProjection(ProjectionMetric):
    name = 'common-obj-rep-tsne'
    input_type = 'rep_SBIR_on_validation_set'
    plot_type = 'scatter-with-shapes'

    def compute(self, input_data):
        _, _, _, _, _, R_ave, mAP, obj_emb, sketch_emb, y = input_data
        skt_y = y

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        skt_idx = np.random.permutation(len(skt_y))
        np.random.seed()

        sel_objs, sel_skts, labels, skt_labels, sel_labels, counter, label_counter = [], [], [], [], [], 0, 0
        y = y[idx]
        obj_emb = obj_emb[idx]
        skt_y = skt_y[skt_idx]
        sketch_emb = sketch_emb[skt_idx]
        for skt, label in zip(sketch_emb, skt_y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                sel_skts.append(skt)
                skt_labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        counter = 0
        for obj, label in zip(obj_emb, y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                sel_objs.append(obj)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        combined_embs = np.concatenate((sel_objs, sel_skts), axis=0)
        print(np.array(obj_emb).shape, np.array(sketch_emb).shape, np.array(sel_objs).shape, np.array(sel_skts).shape, np.array(combined_embs).shape,  np.array(labels).shape)
        tsne_results = tsne.fit_transform(combined_embs)
        objs_tsne, skt_tsne = tsne_results[:len(sel_objs)], tsne_results[len(sel_objs):]
        return np.array([np.concatenate((objs_tsne, labels),
                                        axis=1),
                         np.concatenate((skt_tsne, skt_labels),
                                        axis=1)])


class ClusterTSNEProjection(ProjectionMetric):
    name = 'tsne-cluster'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(x))
        np.random.seed()

        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        x, y, pred_z = x[idx], y[idx], pred_z[idx]
        for sketch, label, feature in zip(x, y, pred_z):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                feats.append(feature)
                labels.append(label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        tsne_results = tsne.fit_transform(feats)

        kmeans = KMeans(n_clusters=30, random_state=14).fit(feats)
        cluster_labels = kmeans.labels_
        sel_feats, feats_labels = None, None
        for i in range(30):
            clustered_feats = np.array(tsne_results)[np.where(cluster_labels == i)[0]]
            sel_feats = np.concatenate((sel_feats, clustered_feats)) if sel_feats is not None else clustered_feats
            feat_label = np.ones(len(clustered_feats,)) * i
            feats_labels = np.concatenate((feats_labels, feat_label)) if feats_labels is not None else feat_label

        np.random.seed()
        return np.concatenate((sel_feats, np.expand_dims(feats_labels, axis=1)),
                              axis=1)


class PredictedLabelsTSNEProjection(ProjectionMetric):
    name = 'tsne-predicted'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        x, y, pred_x, pred_y, pred_z, tokenizer, plot_filepath, tmp_filepath = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(y))
        np.random.seed()

        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        y, pred_z, pred_y = y[idx], pred_z[idx], pred_y[idx]
        for label, feature, pred_label in zip(y, pred_z, pred_y):
            if label not in sel_labels and label_counter < 10:
                sel_labels.append(label)
                label_counter += 1
            if label in sel_labels:
                feats.append(feature)
                labels.append(pred_label)
                counter += 1
            if counter >= 1000:
                break

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)
        tsne_results = tsne.fit_transform(feats)
        return np.concatenate((tsne_results, np.expand_dims(labels, axis=1)),
                              axis=1)


class PCAProjection(ProjectionMetric):
    name = 'pca'
    input_type = 'predictions_on_validation_set'

    def compute(self, input_data):
        entries, _, plot_filepath, tmp_filepath = input_data

        np.random.seed(14)
        idx = np.random.permutation(len(entries))
        np.random.seed()

        feats, labels, sel_labels, counter, label_counter = [], [], [], 0, 0
        for i in idx:
            skt = entries[i]
            if skt['label'] not in sel_labels and label_counter < 10:
                sel_labels.append(skt['label'])
                label_counter += 1
            if skt['label'] in sel_labels:
                feats.append(skt['features'])
                labels.append(skt['label'])
                counter += 1
            if counter >= 1000:
                break

        pca = PCA(n_components=2)
        pca.fit(feats)
        pca_result = pca.transform(feats)
        return np.concatenate((pca_result, np.expand_dims(labels, axis=1)),
                              axis=1)
