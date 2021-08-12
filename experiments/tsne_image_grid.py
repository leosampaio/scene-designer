import os
import glob

import numpy as np
import tensorflow as tf
import skimage.io as skio
import matplotlib.pyplot
import pickle
import skimage.transform as sk_transform
# from lapjv import lapjv
from scipy.spatial.distance import cdist
from PIL import Image
from sklearn.manifold import TSNE

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


class TSNEImageGrid(Experiment):
    name = "tsne-image-grid"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            results_dir='/vol/vssp/cvpnobackup/scratch_4weeks/m13395/results',
            featsfile='coco_scene_triplet_valid_feats.npz',
        )
        return hps

    def compute(self, model=None):
        filename = os.path.join(self.hps['featsfile'], self.hps['featsfile'])
        data = np.load(filename, allow_pickle=True)

        to_plot = 1500

        tsne = TSNE(n_components=2,
                    verbose=0, perplexity=30,
                    n_iter=1000,
                    random_state=14)

        combined_embs = np.concatenate((data['img_reps'][:to_plot], data['skt_reps'][:to_plot]), axis=0)
        tsne_r = tsne.fit_transform(combined_embs)

        tx, ty = tsne_r[:, 0], tsne_r[:, 1]
        tx = (tx - np.min(tx)) / (np.max(tx) - np.min(tx))
        ty = (ty - np.min(ty)) / (np.max(ty) - np.min(ty))
        tsne_r[:, 0], tsne_r[:, 1] = tx, ty

        tsne_results, tsne_results_skt = tsne_r[:to_plot], tsne_r[to_plot:]
        tx, ty, tx_s, ty_s = tx[:to_plot], ty[:to_plot], tx[to_plot:], ty[to_plot:]

        # plot based on the objects present
        labels = []
        label_names = ['animal', 'food', 'vehicle', 'furniture', 'sport stuff']
        n2id = model.dataset.obj_name_to_idx
        id2idx = model.dataset.obj_id_to_idx
        for objs in data['objs'][:to_plot]:
            if id2idx[n2id['dog']] in objs or id2idx[n2id['cat']] in objs or id2idx[n2id['cow']] in objs or id2idx[n2id['sheep']] in objs or id2idx[n2id['horse']] in objs or id2idx[n2id['zebra']] in objs or id2idx[n2id['elephant']] in objs or id2idx[n2id['giraffe']] in objs:
                labels.append(0)
            elif id2idx[n2id['vegetable']] in objs or id2idx[n2id['banana']] in objs or id2idx[n2id['wine glass']] in objs or id2idx[n2id['donut']] in objs or id2idx[n2id['apple']] in objs or id2idx[n2id['pizza']] in objs in objs or id2idx[n2id['cake']] in objs or id2idx[n2id['broccoli']] in objs or id2idx[n2id['hot dog']] in objs or id2idx[n2id['fruit']] in objs or id2idx[n2id['carrot']] in objs or id2idx[n2id['food-other']] in objs or id2idx[n2id['sandwich']] in objs:
                labels.append(1)
            elif id2idx[n2id['airplane']] in objs or id2idx[n2id['car']] in objs or id2idx[n2id['truck']] in objs or id2idx[n2id['train']] in objs:
                labels.append(2)
            elif id2idx[n2id['bed']] in objs or id2idx[n2id['refrigerator']] in objs or id2idx[n2id['microwave']] in objs or id2idx[n2id['toilet']] in objs or id2idx[n2id['chair']] in objs or id2idx[n2id['table']] in objs:
                labels.append(3)
            # elif id2idx[n2id['baseball bat']] in objs or id2idx[n2id['tennis racket']] in objs or id2idx[n2id['sports ball']] in objs:
            #     labels.append(4)
            else:
                labels.append(4)  # other
        labels = np.array(labels)

        cmap = cm.tab10
        category_labels = labels
        unique_labels = [0, 1, 2, 3]
        norm = colors.Normalize(vmin=0, vmax=len(unique_labels))
        for i, label in enumerate(unique_labels):
            category_labels = [i if x == label else x for x in category_labels]
            unique_labels = [i if x == label else x for x in unique_labels]

        plt.figure(figsize=(16, 16))

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        mapped_colors = cmapper.to_rgba(unique_labels)

        for i, (name, colour) in enumerate(zip(label_names, mapped_colors)):
            plt.scatter(tx[labels == i], ty[labels == i], color=colour, label=name, alpha=0.7, edgecolors='none')
        for i, (name, colour) in enumerate(zip(label_names, mapped_colors)):
            plt.scatter(tx_s[labels == i], ty_s[labels == i], color=colour, alpha=0.7, edgecolors='none', marker='v')

        plt.legend(loc='best')

        label_plot_file = os.path.join(model.tmp_out_dir, 'tsne-label-plot.png')
        plt.savefig(label_plot_file, dpi=300, bbox_inches='tight')
        model.notifyier.notify_with_image(label_plot_file,
                                          model.identifier)
        plt.clf()

        # plot based on number of objects
        n_obj_labels = np.array([len(objs) for objs in data['objs'][:to_plot]])
        print(n_obj_labels)

        cmap = cm.tab10
        category_labels = n_obj_labels
        unique_labels = [1, 2, 3, 4, 5, 6, 7, 8]
        norm = colors.Normalize(vmin=0, vmax=unique_labels[-1])

        plt.figure(figsize=(16, 16))

        cmapper = cm.ScalarMappable(norm=norm, cmap=cmap)
        mapped_colors = cmapper.to_rgba(unique_labels)

        for i, colour in zip(unique_labels, mapped_colors):
            plt.scatter(tx[n_obj_labels == i], ty[n_obj_labels == i], color=colour, label=i, alpha=0.7, edgecolors='none')
        for i, colour in zip(unique_labels, mapped_colors):
            plt.scatter(tx_s[n_obj_labels == i], ty_s[n_obj_labels == i], color=colour, alpha=0.7, edgecolors='none', marker='v')

        plt.legend(loc='best')

        nobjs_plot_file = os.path.join(model.tmp_out_dir, 'tsne-label-plot.png')
        plt.savefig(nobjs_plot_file, dpi=300, bbox_inches='tight')
        model.notifyier.notify_with_image(nobjs_plot_file,
                                          model.identifier)
        plt.clf()

        width = 4000
        height = 4000
        max_dim = 100

        full_image = Image.new('RGBA', (width, height))
        for img, x, y in zip(data['images'][:to_plot], tx, ty):
            tile = img
            rs = max(1, tile.shape[0] / max_dim, tile.shape[1] / max_dim)
            tile = sk_transform.resize(tile, (int(tile.shape[0] / rs), int(tile.shape[1] / rs)), anti_aliasing=True)
            tile = Image.fromarray(np.uint8(tile * 255))
            full_image.paste(tile, (int((width - max_dim) * x), int((height - max_dim) * y)), mask=tile.convert('RGBA'))

        image_plot_file = os.path.join(model.tmp_out_dir, 'tsne-image-plot.png')
        full_image.save(image_plot_file)
        model.notifyier.notify_with_image(image_plot_file,
                                          model.identifier)
        return

        out_dim = 25
        out_res = 64
        sel_tsne = tsne_results[:to_plot]
        grid = np.dstack(np.meshgrid(np.linspace(0, 1, out_dim), np.linspace(0, 1, out_dim))).reshape(-1, 2)
        cost_matrix = cdist(grid, sel_tsne, "sqeuclidean").astype(np.float32)
        cost_matrix = cost_matrix * (100000 / cost_matrix.max())
        row_asses, col_asses, _ = lapjv(cost_matrix)
        grid_jv = grid[col_asses]
        out = np.ones((out_dim * out_res, out_dim * out_res, 3))

        for pos, img in zip(grid_jv, data['images'][0:to_plot]):
            h_range = int(np.floor(pos[0] * (out_dim - 1) * out_res))
            w_range = int(np.floor(pos[1] * (out_dim - 1) * out_res))
            out[h_range:h_range + out_res, w_range:w_range + out_res] = img

        im = Image.fromarray(np.uint8(out * 255))
        image_grid_plot_file = os.path.join(model.tmp_out_dir, 'tsne-image-grid-plot.png')
        im.save(image_grid_plot_file)

        model.notifyier.notify_with_image(image_grid_plot_file,
                                          model.identifier)
