import numpy as np
import os
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM

from core.metrics import ImageMetric, HistoryMetric
import utils

from sklearn.cluster import KMeans
import tensorflow as tf


class ReconstructedImageFromSceneGraph(ImageMetric):
    name = 'image-from-scene-graph'
    input_type = 'scene_graph_forward_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        origs, reconstructions, _, _, _ = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:32]
        np.random.seed()

        imgs = np.zeros((len(idx) * 2,) + origs.shape[1:])
        imgs[0::2] = origs[idx]
        imgs[1::2] = reconstructions[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class SimpleReconstruction(ImageMetric):
    name = 'reconstruction'
    input_type = 'reconstruction_samples'
    plot_type = 'image-grid'

    def compute(self, input_data):
        origs, reconstructions = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:32]
        np.random.seed()

        imgs = np.zeros((len(idx) * 2,) + origs.shape[1:])
        imgs[0::2] = origs[idx]
        imgs[1::2] = reconstructions[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class BigReconstruction(ImageMetric):
    name = 'big-reconstruction'
    input_type = 'reconstruction_samples'
    plot_type = 'image-grid'
    dpi = 400

    def compute(self, input_data):
        origs, reconstructions = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:128]
        np.random.seed()

        imgs = np.zeros((len(idx) * 2,) + origs.shape[1:])
        imgs[0::2] = origs[idx]
        imgs[1::2] = reconstructions[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs

class ReconstructedLayoutFromSceneGraph(ImageMetric):
    name = 'layout-from-scene-graph'
    input_type = 'scene_graph_layouts_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        gt, pred, skt_pred = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(gt))[:27]
        np.random.seed()

        imgs = np.zeros((len(idx) * 3,) + gt.shape[1:])
        imgs[0::3] = gt[idx]
        imgs[1::3] = skt_pred[idx]
        imgs[2::3] = pred[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedLayoutFromSceneGraphBigValid(ImageMetric):
    name = 'layout-from-scene-graph-big-valid'
    input_type = 'scene_graph_layouts_on_big_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        gt, pred, skt_pred = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(gt))[:147]
        np.random.seed()

        imgs = np.zeros((len(idx) * 3,) + gt.shape[1:])
        imgs[0::3] = gt[idx]
        imgs[1::3] = skt_pred[idx]
        imgs[2::3] = pred[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedImageFromPredLayout(ImageMetric):
    name = 'image-from-pred-layout'
    input_type = 'scene_graph_forward_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        origs, _, reconstructions, _, _ = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:32]
        np.random.seed()

        imgs = np.zeros((len(idx) * 2,) + origs.shape[1:])
        imgs[0::2] = origs[idx]
        imgs[1::2] = reconstructions[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedImageFromSketchLayout(ImageMetric):
    name = 'image-from-sketch-layout'
    input_type = 'scene_graph_forward_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        origs, _, _, sketches, reconstructions = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:27]
        np.random.seed()

        imgs = np.zeros((len(idx) * 3,) + origs.shape[1:])
        imgs[0::3] = origs[idx]
        imgs[1::3] = sketches[idx]
        imgs[2::3] = reconstructions[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedObjectsFromSceneGraphGenerator(ImageMetric):
    name = 'objs-from-layout'
    input_type = 'object_generation_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        origs, recons, recons_pred_layout, sketches, recons_sketch_layout = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:20]
        np.random.seed()

        imgs = np.zeros((100,) + origs.shape[1:])
        imgs[0::5] = origs[idx]
        imgs[1::5] = recons[idx]
        imgs[2::5] = recons_pred_layout[idx]
        imgs[3::5] = sketches[idx]
        imgs[4::5] = recons_sketch_layout[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class EdgemapsFromSceneGraphGenerator(ImageMetric):
    name = 'obj-edgemaps'
    input_type = 'object_edgemap_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        origs, edges, sketches = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:27]
        np.random.seed()

        imgs = np.zeros((81,) + origs.shape[1:])
        imgs[0::3] = origs[idx]
        imgs[1::3] = edges[idx]
        imgs[2::3] = sketches[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedMasksFromSceneGraphGenerator(ImageMetric):
    name = 'generated-masks'
    input_type = 'mask_generation_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        masks, masks_pred, skts, masks_sketch = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(masks))[:16]
        np.random.seed()

        imgs = np.zeros((64,) + masks.shape[1:])
        imgs[0::4] = masks[idx]
        imgs[1::4] = masks_pred[idx]
        imgs[2::4] = skts[idx]
        imgs[3::4] = masks_sketch[idx]
        imgs = np.clip(imgs, 0., 1.)
        if len(imgs.shape) == 4 and imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedMasksFromGraph(ImageMetric):
    name = 'masks-from-graph'
    input_type = 'mask_generation_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        masks, masks_pred, _, _ = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(masks))[:32]
        np.random.seed()

        imgs = np.zeros((len(idx) * 2,) + masks.shape[1:])
        imgs[0::2] = masks[idx]
        imgs[1::2] = masks_pred[idx]
        imgs = np.clip(imgs, 0., 1.)
        if len(imgs.shape) == 4 and imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedMasksFromSceneGraphGeneratorOnBigValidationSet(ImageMetric):
    name = 'generated-masks-big-valid'
    input_type = 'mask_generation_on_big_validation_set'
    plot_type = 'image-grid'
    dpi = 300

    def compute(self, input_data):
        masks, masks_pred, skts, masks_sketch = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(masks))[:144]
        np.random.seed()

        imgs = np.zeros((4 * len(idx),) + masks.shape[1:])
        imgs[0::4] = masks[idx]
        imgs[1::4] = masks_pred[idx]
        imgs[2::4] = skts[idx]
        imgs[3::4] = masks_sketch[idx]
        imgs = np.clip(imgs, 0., 1.)
        if len(imgs.shape) == 4 and imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedAllFromSceneGraphOnBigValidationSet(ImageMetric):
    name = 'images-from-scene-graph-big-valid'
    input_type = 'scene_graph_forward_on_big_validation_set'
    plot_type = 'image-grid'
    dpi = 600

    def compute(self, input_data):
        origs, recons, pred_layout, sketches, sketch_layout = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:125]
        np.random.seed()

        imgs = np.zeros((len(idx) * 5,) + origs.shape[1:])
        imgs[0::5] = origs[idx]
        imgs[1::5] = recons[idx]
        imgs[2::5] = pred_layout[idx]
        imgs[3::5] = sketches[idx]
        imgs[4::5] = sketch_layout[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class ReconstructedImagesFromSceneGraphOnBigValidationSet(ImageMetric):
    name = 'reconstructed-images-from-big-valid'
    input_type = 'scene_graph_forward_on_big_validation_set'
    plot_type = 'image-grid'
    dpi = 500

    def compute(self, input_data):
        origs, recons, _, _, _, _ = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(origs))[:128]
        np.random.seed()

        imgs = np.zeros((len(idx) * 2,) + origs.shape[1:])
        imgs[0::2] = origs[idx]
        imgs[1::2] = recons[idx]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class SBIROnValidationSet(ImageMetric):
    name = 'sbir-on-valid'
    input_type = 'SBIR_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(all_skts))[:28]
        np.random.seed()

        imgs = np.zeros((len(idx) * 7,) + all_skts.shape[1:])
        imgs[0::7] = right_imgs[idx]
        imgs[1::7] = all_skts[idx]
        imgs[2::7] = all_imgs[idx][:, 0]
        imgs[3::7] = all_imgs[idx][:, 1]
        imgs[4::7] = all_imgs[idx][:, 2]
        imgs[5::7] = all_imgs[idx][:, 3]
        imgs[6::7] = all_imgs[idx][:, 4]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class CommonEmbeddingSBIROnValidationSet(ImageMetric):
    name = 'common-sbir-on-valid'
    input_type = 'common_SBIR_on_validation_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        all_skts, all_imgs, mAP, top1, top5, top10 = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(all_skts))[:54]
        np.random.seed()

        imgs = np.zeros((len(idx) * 6,) + all_skts.shape[1:])
        imgs[0::6] = all_skts[idx]
        imgs[1::6] = all_imgs[idx][:, 0]
        imgs[2::6] = all_imgs[idx][:, 1]
        imgs[3::6] = all_imgs[idx][:, 2]
        imgs[4::6] = all_imgs[idx][:, 3]
        imgs[5::6] = all_imgs[idx][:, 4]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class CommonEmbeddingFGSBIROnTestSet(ImageMetric):
    name = 'fgsbir-on-test'
    input_type = 'FGSBIR_on_test_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        all_skts, all_imgs, mAP, top1, top5, top10 = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(all_skts))[:54]
        np.random.seed()

        imgs = np.zeros((len(idx) * 6,) + all_skts.shape[1:])
        imgs[0::6] = all_skts[idx]
        imgs[1::6] = all_imgs[idx][:, 0]
        imgs[2::6] = all_imgs[idx][:, 1]
        imgs[3::6] = all_imgs[idx][:, 2]
        imgs[4::6] = all_imgs[idx][:, 3]
        imgs[5::6] = all_imgs[idx][:, 4]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class SBIROnTestSet(ImageMetric):
    name = 'sbir-on-test'
    input_type = 'SBIR_on_test_set'
    plot_type = 'image-grid'

    def compute(self, input_data):
        all_skts, all_imgs, right_imgs, mAP, top1, top5, top10 = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(all_skts))[:28]
        np.random.seed()

        imgs = np.zeros((len(idx) * 7,) + all_skts.shape[1:])
        imgs[0::7] = right_imgs[idx]
        imgs[1::7] = all_skts[idx]
        imgs[2::7] = all_imgs[idx][:, 0]
        imgs[3::7] = all_imgs[idx][:, 1]
        imgs[4::7] = all_imgs[idx][:, 2]
        imgs[5::7] = all_imgs[idx][:, 3]
        imgs[6::7] = all_imgs[idx][:, 4]
        imgs = np.clip(imgs, 0., 1.)
        if imgs.shape[3] == 1:
            imgs = np.squeeze(imgs, axis=(3,))
        return imgs


class DataSanityCheck(ImageMetric):
    name = 'data-sanity-check'
    input_type = 'data_sanity_check'
    plot_type = 'image-grid'
    dpi = 300

    def compute(self, input_data):
        imgs = input_data

        np.random.seed(19)
        idx = np.random.permutation(len(imgs))[:9]
        np.random.seed()

        imgs_grid = np.zeros((len(idx),) + imgs.shape[1:])
        imgs_grid[...] = imgs[idx]
        imgs_grid = np.clip(imgs_grid, 0., 1.)
        if len(imgs_grid.shape) == 4 and imgs_grid.shape[3] == 1:
            imgs_grid = np.squeeze(imgs_grid, axis=(3,))
        return imgs_grid
