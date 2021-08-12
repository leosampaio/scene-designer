import os
import glob

import numpy as np
import tensorflow as tf

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


class ExtractAttention(Experiment):
    name = "extract-attention"
    requires_model = True

    @classmethod
    def specific_default_hparams(cls):
        hps = utils.hparams.HParams(
            reuse=False,
            n_samples=10,
        )
        return hps

    def gather_sketches_for(self, set_type):
        filename = os.path.join(self.model.wgt_out_dir, 'attention_weights_images_graphs.npz')
        if self.hps['reuse']:
            data = np.load(filename, allow_pickle=True)
            print("Loaded preprocessed data from {}".format(filename))
        else:

            counter = 0
            all_skts, all_imgs, all_reps, all_skt_reps, all_attw, all_attw_skt, all_objs = [], [], [], [], [], [], []
            batch_iterator = self.model.dataset.get_n_samples_batch_iterator_from(set_type, self.hps['n_samples'], batch_size=1)
            for batch in batch_iterator:

                (imgs, objs, boxes, masks, triples, attr, objs_to_img,
                 sketches, n_crops, n_labels, sims) = batch
                sketches = np.array(sketches)

                img_s = self.model.dataset.hps['image_size']

                def fun(skts, boxes, obj_to_img):
                    return utils.layout.get_sketch_composition(skts, boxes, obj_to_img, img_s, img_s, 1)
                skt_comp = self.model.reduce_lambda_concat(fun, sketches, boxes, objs_to_img)

                n_imgs = tf.shape(imgs)[0]

                # extract embeddings and attention weights

                # object level encoding
                g_crops = utils.bbox.crop_bbox_batch(imgs, boxes, objs_to_img, self.model.dataset.hps['crop_size'])
                g_obj_encoded = self.model.multidomain_encoder.obj_encoder.forward(g_crops, objs, masks)
                g_obj_rep = self.model.enc_net(g_obj_encoded)

                # correlated object level encoding
                pred_vecs, edges = self.model.embedding_model(triples)
                obj_vecs, _ = self.model.compute_gcn(g_obj_rep, pred_vecs, edges, n_imgs)

                # image level encoding
                img_reps, e_masks, obj_co_vecs, eatt_weights = self.model.graph_encoder(obj_vecs, objs_to_img, n_imgs, attr, training=False)

                # repeat, but using sketch objects
                skt_obj_encoded = self.model.multidomain_encoder.skt_encoder.forward(sketches)
                skt_obj_rep = self.model.enc_net(skt_obj_encoded)
                sketch_obj_vecs, _ = self.model.compute_gcn(skt_obj_rep, pred_vecs, edges, n_imgs)
                img_skt_reps, e_skt_masks, skt_co_vecs, eatt_weights_skt = self.model.graph_encoder(sketch_obj_vecs, objs_to_img, n_imgs, attr, training=False)

                if self.model.hps['bottleneck_vectors']:
                    obj_co_vecs, datt_weights = self.model.graph_decoder(img_reps, e_masks, objs_to_img, n_imgs, training=False)
                    skt_co_vecs, datt_weights_skt = self.model.graph_decoder(img_skt_reps, e_skt_masks, objs_to_img, n_imgs, training=False)

                all_skts.append(np.concatenate((skt_comp, skt_comp, skt_comp), axis=-1))
                all_reps.append(img_reps)
                all_skt_reps.append(img_skt_reps)
                all_imgs.append(imgs)
                all_attw.append(eatt_weights)
                all_attw_skt.append(eatt_weights_skt)
                all_objs.append(objs)

                counter += 1
                print("[Processing] {}/{} images done!".format(counter, self.hps['n_samples']))

            all_skts = np.concatenate(all_skts, axis=0)
            # all_nover_skts = np.concatenate(all_nover_skts, axis=0)
            all_reps = np.concatenate(all_reps, axis=0)
            all_skt_reps = np.concatenate(all_skt_reps, axis=0)
            all_imgs = self.model.dataset.deprocess_image(np.concatenate(all_imgs, axis=0))

            data = {'skts': all_skts,
                    'reps': all_reps,
                    'skt_reps': all_skt_reps,
                    'imgs': all_imgs,
                    'att_weights': all_attw,
                    'att_weights_skt': all_attw_skt, 
                    'objs': all_objs}
            np.savez(filename, **data)

    def compute(self, model=None):
        self.model = model

        self.gather_sketches_for('valid')
        # self.gather_sketches_for('test')
