import time

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import json

import builders
import utils
from utils.hparams import HParams
from core.models import BaseModel
from builders.layers.scenegraph import (GraphTripleConvNet,
                                        GraphTripleConvLayer,
                                        build_mlp)
from builders.layers.graphatt import GraphEncoder, GraphDecoder
from builders.layers.scenegenerators import build_mask_net
from builders.activations import multiply_gradients
from models.mixins import (GraphAttentionRepresentationMetrics,
                           CommonEmbeddingLayersMixin,
                           GraphAttentionRepresentationInferenceMixin)
from models.multidomain_classifier import MultidomainMapper
from utils.representation_randomizer import TFVectorRepQueue


class GraphAttentionRepresentation(BaseModel,
                                   GraphAttentionRepresentationMetrics,
                                   CommonEmbeddingLayersMixin,
                                   GraphAttentionRepresentationInferenceMixin):
    name = 'graph-attention'
    quick_metrics = ['obj_triplet_loss', 'enc_sketch_clas_loss',
                     'enc_obj_clas_loss', 'global_triplet_loss',
                     'bbox_recon_loss', 'end_clas_loss',
                     'g_mask_loss', 'g_mask_sketch_loss',
                     'mask_recon_loss', 'g_mask_feat_loss',
                     'time']
    slow_metrics = []

    @classmethod
    def specific_default_hparams(cls):
        """Return default HParams"""
        hparams = HParams(

            # optimiser params
            optimizer='adam',
            learning_rate=1e-4,

            # initial embeddings
            obj_pred_embedding_dim=128,

            # graph CNN parameters
            gconv_dim=256,
            gconv_hidden_dim=512,
            gconv_pooling='avg',
            gconv_num_layers=4,
            mlp_normalization='none',

            sketch_appearance_arch='big',  # deprecated, use use_imagenet_pretrained=True
            image_appearance_arch='big',  # deprecated, use use_imagenet_pretrained=True

            edge_repr_net_hidden_size=256,
            common_repr_size=128,
            encoding_size=256,
            use_enc_not_rep=True,

            # submodels
            d_obj_arch='small',
            pt_mdomain_id='',  # pretrained triplet model
            pt_obj_id='',  # deprecated, pretrain multidomain classifier above
            pt_skt_id='',  # deprecated, pretrain multidomain classifier above
            pretrained_e_model='mobilenet2',  # pretrained net arch
            use_euclidean_distance_map=False,
            use_imagenet_pretrained=True,
            triplet_loss_weight=1.,

            mul_grad_to_repr=False,
            g_layout_feat_weight=10.,
            g_mask_norm='batch',
            d_layout_norm='instance',
            accum_grads_for=1,  # deprecated
            do_channel_dis=False,
            use_ttur=True,  # deprecated
            use_lab_smooth=True,
            use_noise_dis=False,  # deprecated
            use_sn_mask_dis=True,

            # transformer parameters
            queue_size=512,  # deprecated
            g_enc_n_layers=3,
            g_enc_n_heads=16,
            g_enc_encoding_dim=512,
            use_pos_encoding=True,
            use_moco_loss=False,
            obj_emb_weight=2.,
            no_grad_to_common=False,

            layout_d_arch='big',
            use_pos_on_graph=False,
            use_giou_loss=True,
            recon_on_sketch=True,

            # abblations
            abbl_skip_transformer=False,
            abbl_skip_graph=False,
            abbl_clas_loss=1.,
            abbl_triplet_loss=1.,
            abbl_mask_recon_loss=10.,
            abbl_no_mask_gan=False,
            abbl_include_feat_loss=False,
            abbl_no_shuffled_negative=False,

            # deprecated, from similarity matrix experiments
            margin_mul=1.,
            clas_on_final=True,
            abbl_no_skt_mask_loss=False,
            freeze_all_but_mask=False,
            no_global_contrastive=False,
            bottleneck_vectors=False,
            gcn_activation='leakyrelu',
            zero_att_img_emb=True,

            attributes_dim=10,
            include_scoco=False,

            # extra params from multidomain classifier (no)
            finetune=False,
            finetune_dataset='sketchy-extended-tf',
            finetune_ratio=0.5,
            finetune_extra_classifier=False,

            three_part_pos_enc=False,
            randomize_pos_enc=False,
            superbox_reconstruction=False,

            image_size=96,
            use_focal_loss=False,
        )
        return hparams

    def __init__(self, hps, dataset, out_dir, experiment_id):
        self.losses_manager = builders.losses.LossManager()
        super().__init__(hps, dataset, out_dir, experiment_id)

    def build_model(self):

        # prepare some hyperparams
        self.image_size = self.dataset.image_size
        self.num_objs = self.dataset.n_classes
        self.num_preds = len(self.dataset.pred_idx_to_name)

        # load our submodel
        self.multidomain_encoder = MultidomainMapper(
            self.hps, self.dataset, self.out_dir_root, self.hps['pt_mdomain_id'])
        self.multidomain_encoder.restore_checkpoint_if_exists('latest')
        self.enc_net = tf.keras.Sequential(self.multidomain_encoder.repr_net.layers[0:2])

        # build initial embeddings and graph cnn
        self.embedding_model = self.build_embedding_model()
        if not self.hps['abbl_skip_graph']:
            self.gconv = self.build_graph_cnn()
        else:
            self.gconv = self.build_abblation_graph_cnn()

        # the multihead attention and self attention encoder
        self.graph_encoder = GraphEncoder(
            num_layers=self.hps['g_enc_n_layers'], d_model=self.hps['gconv_dim'], num_heads=self.hps['g_enc_n_heads'], dff=512,
            seq_len=9, maximum_position_encoding=256,
            use_positional_encoding=self.hps['use_pos_encoding'],
            abblation=self.hps['abbl_skip_transformer'],
            zero_att_img_emb=self.hps['zero_att_img_emb'],
            three_part_pos_enc=self.hps['three_part_pos_enc'],
            randomize_pos_enc=self.hps['randomize_pos_enc'])
        self.global_triplet_encoder = tf.keras.layers.Dense(self.hps['gconv_dim'])

        if self.hps['bottleneck_vectors']:
            self.graph_decoder = GraphDecoder(num_layers=self.hps['g_enc_n_layers'], d_model=self.hps['gconv_dim'],
                                              num_heads=self.hps['g_enc_n_heads'], dff=512, seq_len=9)

        # the bounding box generator is just an MLP
        box_net_layers = [self.hps['gconv_dim'], self.hps['gconv_hidden_dim'], 4]
        self.box_net = build_mlp(box_net_layers,
                                 batch_norm=self.hps['mlp_normalization'])

        # the mask generator is an upsampling CNN
        self.mask_net = build_mask_net(self.hps['gconv_dim'], self.dataset.mask_size, norm=self.hps['g_mask_norm'])
        self.generate_mask = self.build_generate_masks_fun()

        self.mask_dis = self.build_mask_discriminator()

        # finally, build the model trainers
        self.optimizer = tf.keras.optimizers.Adam(self.hps['learning_rate'])
        self.triplet_optimizer = tf.keras.optimizers.Adam(self.hps['learning_rate'], beta_1=0.5, beta_2=0.999, epsilon=1e-9)
        self.d_optimizer = tf.keras.optimizers.Adam(self.hps['learning_rate'] * 4, beta_1=0.5, beta_2=0.999, epsilon=1e-9)

        self.prepare_all_losses()
        if not self.hps['freeze_all_but_mask']:
            self.networks = [self.embedding_model, self.gconv,
                             self.graph_encoder, self.global_triplet_encoder,
                             self.mask_net, self.box_net]
        else:
            self.networks = [self.mask_net, self.box_net]

        if self.hps['clas_on_final']:
            self.end_classifier = tf.keras.layers.Dense(self.num_objs)
            self.networks += [self.end_classifier]

        if self.hps['bottleneck_vectors']:
            self.networks += [self.graph_decoder]

        self.forward = self.build_forward_fun()

        self.model_trainer = self.build_model_trainer()

        self.all_vars = []
        # for g in self.networks:
        #     self.all_vars += g.trainable_variables

        self.is_first_run = True

    def build_embedding_model(self):
        self.pred_embeddings = tf.keras.layers.Embedding(
            self.num_preds, self.hps['obj_pred_embedding_dim'])

        def embedding_model(triples):
            # break down the triples and get embeddings
            s, p, o = triples[:, 0], triples[:, 1], triples[:, 2]
            edges = tf.stack([s, o], axis=1)
            pred_vecs = self.pred_embeddings(p)
            return edges, pred_vecs
        return embedding_model

    def build_graph_cnn(self):
        if self.hps['use_pos_on_graph']:
            obj_vec_shape = self.hps['edge_repr_net_hidden_size'] + self.hps['attributes_dim']
        else:
            obj_vec_shape = self.hps['edge_repr_net_hidden_size']
        in_objects = tf.keras.Input(shape=(obj_vec_shape,))
        in_predicates = tf.keras.Input(shape=(self.hps['obj_pred_embedding_dim'],))
        in_edges = tf.keras.Input(shape=(2,), dtype=tf.int32)

        # first graph CNN layer changes dimensions
        gconv = GraphTripleConvLayer(
            input_dim=self.hps['edge_repr_net_hidden_size'],
            attributes_dim=self.hps['attributes_dim'],
            output_dim=self.hps['gconv_dim'],
            hidden_dim=self.hps['gconv_hidden_dim'],
            pooling=self.hps['gconv_pooling'],
            mlp_normalization=self.hps['mlp_normalization'],
            activation=self.hps['gcn_activation'])

        # and the next ones just stacks layers
        gconv_net = None
        if self.hps['gconv_num_layers'] > 1:
            gconv_net = GraphTripleConvNet(
                input_dim=self.hps['gconv_dim'],
                hidden_dim=self.hps['gconv_hidden_dim'],
                pooling=self.hps['gconv_pooling'],
                num_layers=self.hps['gconv_num_layers'] - 1,
                mlp_normalization=self.hps['mlp_normalization'],
                activation=self.hps['gcn_activation'])

        self.internal_gconv_net = gconv_net.gconvs[-1].net2

        # pass the resulting vectors through the graphCNN
        obj_vecs, pred_vecs = gconv(in_objects, in_predicates, in_edges)
        obj_vecs, pred_vecs = gconv_net(obj_vecs, pred_vecs, in_edges)

        return tf.keras.Model(name="graph_cnn", inputs=[in_objects, in_predicates, in_edges], outputs=[obj_vecs, pred_vecs])

    def compute_gcn(self, g_obj_rep, pred_vecs, edges, n_imgs):
        obj_vecs, pred_vecs = self.gconv([g_obj_rep, pred_vecs, edges])
        return obj_vecs, pred_vecs

    def build_abblation_graph_cnn(self):
        self.graph_dense_abblation = tf.keras.layers.Dense(self.hps['gconv_dim'])
        if self.hps['use_pos_on_graph']:
            obj_vec_shape = self.hps['common_repr_size'] + self.hps['attributes_dim']
        else:
            obj_vec_shape = self.hps['common_repr_size']
        in_objects = tf.keras.Input(shape=(obj_vec_shape,))
        in_predicates = tf.keras.Input(shape=(self.hps['obj_pred_embedding_dim'],))
        in_edges = tf.keras.Input(shape=(2,), dtype=tf.int32)

        obj_vecs = self.graph_dense_abblation(in_objects)

        return tf.keras.Model(
            name="graph_cnn",
            inputs=[in_objects, in_predicates, in_edges],
            outputs=[obj_vecs, in_predicates])

    def build_generate_masks_fun(self):
        def generate_masks(mask_vecs):
            mask_squares = tf.expand_dims(tf.expand_dims(mask_vecs, axis=1), axis=1)
            mask_scores = self.mask_net(mask_squares)
            masks_pred = tf.sigmoid(tf.squeeze(mask_scores, axis=-1))
            return masks_pred
        return generate_masks

    def prepare_all_losses(self):
        self.losses_manager.add_categorical_crossentropy('enc_obj_clas', weight=self.hps['abbl_clas_loss'] * .5 * self.hps['obj_emb_weight'], from_logits=True)
        self.losses_manager.add_categorical_crossentropy('enc_sketch_clas', weight=self.hps['abbl_clas_loss'] * self.hps['obj_emb_weight'], from_logits=True)
        self.losses_manager.add_triplet_loss('triplet_loss', weight=self.hps['abbl_triplet_loss'] * self.hps['obj_emb_weight'], margin=.5)
        self.losses_manager.add_contrastive_plus_similarity_matrix_loss(
            'global_triplet_loss', weight=1., mul=self.hps['margin_mul'], abbl_no_shuffled_negative=self.hps['abbl_no_shuffled_negative'])
        self.losses_manager.add_mse_loss('rep_recons', weight=.1)

        if self.hps['use_giou_loss']:
            self.losses_manager.add_giou_loss('bbox_recon', weight=10.)
        else:
            self.losses_manager.add_mse_loss('bbox_recon', weight=10.)

        if self.hps['abbl_no_skt_mask_loss']:
            g_mask_w, g_mask_sketch_w, d_mask_fake_w = 1., 0., .5
        else:
            g_mask_w, g_mask_sketch_w, d_mask_fake_w = .5, .5, .25

        self.losses_manager.add_binary_crossentropy('mask_recon', weight=self.hps['abbl_mask_recon_loss'])
        self.losses_manager.add_multiscale_lsgan_fake_loss('d_mask_fake', weight=d_mask_fake_w)
        self.losses_manager.add_multiscale_lsgan_real_loss('d_mask_real', weight=.5, smooth=self.hps['use_lab_smooth'])
        self.losses_manager.add_multiscale_lsgan_real_loss('g_mask', weight=g_mask_w)
        self.losses_manager.add_multiscale_lsgan_real_loss('g_mask_sketch', weight=g_mask_sketch_w)
        self.losses_manager.add_feature_matching_loss('g_mask_feat', weight=self.hps['g_layout_feat_weight'])

        self.losses_manager.add_triplet_loss('scoco_triplet', margin=0.5)
        self.losses_manager.add_categorical_crossentropy('final_clas', weight=1., from_logits=True)

        self.compute_losses = self.build_compute_losses_fun()

    def build_compute_losses_fun(self):

        def compute_losses(imgs, objs, boxes, masks, obj_to_img,
                           img_reps, img_tri_reps, img_skt_reps, img_skt_tri_reps,
                           obj_vecs, skt_obj_vecs,
                           n_img_tri_reps, masks_pred, boxes_pred,
                           skt_masks_pred, skt_boxes_pred,
                           t_objs_n, t_obj_enc, t_obj_enc_n, t_skt_enc,
                           t_skt_rep, t_obj_rep, t_obj_rep_n):

            t_one_hot_obj = tf.squeeze(tf.one_hot(objs, self.num_objs), axis=1)
            t_one_hot_obj_n = tf.squeeze(tf.one_hot(t_objs_n, self.num_objs), axis=1)

            # classification
            skt_enc_clas = self.multidomain_encoder.skt_encoder.embedding_classifier(t_skt_enc)
            obj_enc_clas = self.multidomain_encoder.obj_encoder.embedding_classifier(t_obj_enc)
            obj_enc_clas_n = self.multidomain_encoder.obj_encoder.embedding_classifier(t_obj_enc_n)
            enc_obj_clas_loss_p = self.losses_manager.compute_loss('enc_obj_clas', t_one_hot_obj, obj_enc_clas)
            enc_obj_clas_loss_n = self.losses_manager.compute_loss('enc_obj_clas', t_one_hot_obj_n, obj_enc_clas_n)
            enc_sketch_clas_loss = self.losses_manager.compute_loss('enc_sketch_clas', t_one_hot_obj, skt_enc_clas)
            enc_obj_clas_loss = tf.reduce_sum([enc_obj_clas_loss_p, enc_obj_clas_loss_n])

            # object triplet
            obj_triplet_loss = self.losses_manager.compute_loss('triplet_loss', t_skt_rep, t_obj_rep, t_obj_rep_n)

            # global triplet
            if self.hps['no_global_contrastive']:
                global_triplet_loss = tf.constant(0.)
            else:
                train_neg_samples = n_img_tri_reps
                pos_samples = img_tri_reps
                anc_samples = img_skt_tri_reps
                sim_mat = tf.eye(tf.shape(pos_samples)[0])
                global_triplet_loss = self.losses_manager.compute_loss(
                    'global_triplet_loss', anc_samples, pos_samples, train_neg_samples, sim_mat)

            # mask and box generation
            bbox_recon_loss = self.losses_manager.compute_loss('bbox_recon', boxes, boxes_pred)
            mask_recon_loss = self.losses_manager.compute_loss('mask_recon', masks, masks_pred)
            if self.hps['recon_on_sketch'] or self.hps['include_scoco']:
                if self.hps['include_scoco']:
                    scoco_obj_index = tf.where(obj_to_img >= self.hps['batch_size'])[0, 0]
                    r_boxes, r_masks = boxes[scoco_obj_index:], masks[scoco_obj_index:]
                    r_skt_boxes, r_skt_masks = skt_boxes_pred[scoco_obj_index:], skt_masks_pred[scoco_obj_index:]
                else:
                    r_boxes, r_masks, r_skt_boxes, r_skt_masks = boxes, masks, skt_boxes_pred, skt_masks_pred
                s_bbox_recon_loss = self.losses_manager.compute_loss('bbox_recon', r_boxes, r_skt_boxes)
                s_mask_recon_loss = self.losses_manager.compute_loss('mask_recon', r_masks, r_skt_masks)
                bbox_recon_loss = tf.reduce_mean(bbox_recon_loss) + tf.reduce_mean(s_bbox_recon_loss)
                mask_recon_loss = tf.reduce_mean(mask_recon_loss) + tf.reduce_mean(s_mask_recon_loss)
            if self.hps['abbl_no_mask_gan']:
                d_mask_loss, g_mask_feat_loss, g_mask_loss, g_mask_sketch_loss = tf.constant(0.), tf.constant(0.), tf.constant(0.), tf.constant(0.)
            else:
                masks, masks_pred, skt_masks_pred = tf.expand_dims(masks, -1), tf.expand_dims(masks_pred, -1), tf.expand_dims(skt_masks_pred, -1)
                real_mask_score = self.mask_dis([masks, t_one_hot_obj])
                pred_mask_score = self.mask_dis([masks_pred, t_one_hot_obj])
                pred_skt_mask_score = self.mask_dis([skt_masks_pred, t_one_hot_obj])
                d_mask_real_loss = self.losses_manager.compute_loss('d_mask_real', real_mask_score)
                d_mask_fake_loss = self.losses_manager.compute_loss('d_mask_fake', pred_mask_score)
                d_mask_fake_sketch_loss = self.losses_manager.compute_loss('d_mask_fake', pred_skt_mask_score)
                g_mask_loss = self.losses_manager.compute_loss('g_mask', pred_mask_score)
                g_mask_sketch_loss = self.losses_manager.compute_loss('g_mask_sketch', pred_skt_mask_score)
                g_mask_feat_loss = self.losses_manager.compute_loss('g_mask_feat', real_mask_score, pred_mask_score)
                g_mask_feat_sketch_loss = self.losses_manager.compute_loss('g_mask_feat', real_mask_score, pred_skt_mask_score)
                g_mask_feat_loss = tf.reduce_sum([g_mask_feat_loss, g_mask_feat_sketch_loss])
                d_mask_loss = tf.reduce_sum([d_mask_real_loss, d_mask_fake_loss, d_mask_fake_sketch_loss])

            if self.hps['clas_on_final']:
                obj_end_clas = self.end_classifier(obj_vecs)
                skt_end_clas = self.end_classifier(skt_obj_vecs)
                obj_end_clas_loss = self.losses_manager.compute_loss('final_clas', t_one_hot_obj, obj_end_clas)
                skt_end_clas_loss = self.losses_manager.compute_loss('final_clas', t_one_hot_obj, skt_end_clas)
                end_clas_loss = obj_end_clas_loss + skt_end_clas_loss
            else:
                end_clas_loss = tf.constant(0.)

            losses = [global_triplet_loss, bbox_recon_loss,
                      g_mask_loss, g_mask_sketch_loss, mask_recon_loss]
            if self.hps['abbl_include_feat_loss']:
                losses += [g_mask_feat_loss]
            obj_losses = [obj_triplet_loss, enc_obj_clas_loss,
                          enc_sketch_clas_loss, end_clas_loss]
            losses = tf.reduce_sum([tf.reduce_mean(l) for l in losses])
            obj_losses = tf.reduce_sum([tf.reduce_mean(l) for l in obj_losses])
            quick_metrics = {
                'obj_triplet_loss': obj_triplet_loss,
                'enc_obj_clas_loss': tf.reduce_mean(enc_obj_clas_loss),
                'enc_sketch_clas_loss': tf.reduce_mean(enc_sketch_clas_loss),
                'global_triplet_loss': tf.reduce_mean(global_triplet_loss),
                'bbox_recon_loss': tf.reduce_mean(bbox_recon_loss),
                'g_mask_loss': g_mask_loss,
                'g_mask_sketch_loss': g_mask_sketch_loss,
                'mask_recon_loss': mask_recon_loss,
                'end_clas_loss': tf.reduce_mean(end_clas_loss),
                'g_mask_feat_loss': g_mask_feat_loss
            }
            # losses, obj_losses, d_mask_loss = self.reduce_divide_by_replicas(losses, obj_losses, d_mask_loss)

            return quick_metrics, losses, obj_losses, d_mask_loss
        return compute_losses

    def compute_common_encoding(self, obj_encoded, pred_vecs, edges, attributes, n_imgs, obj_to_img, training):
        if self.hps['mul_grad_to_repr']:
            grad_multiplier = multiply_gradients(0.1)
            obj_encoded = grad_multiplier(obj_encoded)
        obj_rep = self.enc_net(obj_encoded)
        if self.hps['no_grad_to_common']:
            obj_rep = tf.stop_gradient(obj_rep)

        # correlated object level encoding
        if self.hps['use_pos_on_graph']:
            obj_rep = tf.concat([obj_rep, attributes], axis=-1)
        obj_vecs, comp_pred_vecs = self.compute_gcn(obj_rep, pred_vecs, edges, n_imgs)

        # image level encoding
        img_reps, e_masks, obj_co_vecs, _ = self.graph_encoder(obj_vecs, obj_to_img, n_imgs, attributes, training)
        img_tri_reps = self.global_triplet_encoder(img_reps)
        img_tri_reps = tf.math.l2_normalize(img_tri_reps, axis=1)

        if self.hps['bottleneck_vectors']:
            obj_co_vecs, _ = self.graph_decoder(img_reps, e_masks, obj_to_img, n_imgs, training)

        return img_reps, obj_co_vecs, img_tri_reps

    def build_forward_fun(self):
        def forward(imgs, objs, triples, masks,
                    boxes, attributes, obj_to_img, sketches, n_crops,
                    training=False):
            n_imgs = tf.shape(imgs)[0]

            # object level encoding
            g_crops = utils.bbox.crop_bbox_batch(
                imgs, boxes, obj_to_img, self.dataset.ccset.crop_size)
            g_obj_encoded = self.multidomain_encoder.obj_encoder.encoder(g_crops)

            edges, pred_vecs = self.embedding_model(triples)
            img_reps, obj_co_vecs, img_tri_reps = self.compute_common_encoding(
                g_obj_encoded, pred_vecs, edges, attributes, n_imgs, obj_to_img, training)

            # negative objects
            n_obj_encoded = self.multidomain_encoder.obj_encoder.encoder(n_crops)
            n_img_reps, _, n_img_tri_reps = self.compute_common_encoding(
                n_obj_encoded, pred_vecs, edges, attributes, n_imgs, obj_to_img, training)

            # repeat, but using sketch objects
            skt_obj_encoded = self.multidomain_encoder.skt_encoder.encoder(sketches)
            img_skt_reps, skt_co_vecs, img_skt_tri_reps = self.compute_common_encoding(
                skt_obj_encoded, pred_vecs, edges, attributes, n_imgs, obj_to_img, training)

            # generation
            skt_masks_pred = self.generate_mask(skt_co_vecs)
            skt_boxes_pred = self.box_net(skt_co_vecs)
            masks_pred = self.generate_mask(obj_co_vecs)
            boxes_pred = self.box_net(obj_co_vecs)

            return (img_reps, img_tri_reps, img_skt_reps, img_skt_tri_reps,
                    obj_co_vecs, skt_co_vecs, n_img_tri_reps,
                    masks_pred, boxes_pred, skt_masks_pred, skt_boxes_pred)
        return forward

    def forward_and_compute_losses(self, batch):
        (imgs, objs, boxes, masks, triples, attr, objs_to_img,
            sketches, n_crops, n_labels, _) = batch

        # graph model
        (img_reps, img_tri_reps, img_skt_reps,
            img_skt_tri_reps,
            obj_vecs, skt_obj_vecs,
            n_img_tri_reps,
            masks_pred, boxes_pred,
            skt_masks_pred, skt_boxes_pred) = self.forward(imgs, objs, triples, masks, boxes, attr, objs_to_img, sketches, n_crops, training=True)

        # triplet model
        g_crops = utils.bbox.crop_bbox_batch(
            imgs, boxes, objs_to_img, self.dataset.ccset.crop_size)
        t_obj_enc, t_obj_enc_n, t_skt_enc, t_skt_rep, t_obj_rep, t_obj_rep_n = self.multidomain_encoder.forward(
            g_crops, n_crops, sketches)

        # self.negative_skt_reps_queue.update_queue(img_skt_tri_reps)

        return self.compute_losses(imgs, objs, boxes, masks, objs_to_img,
                                   img_reps, img_tri_reps, img_skt_reps, img_skt_tri_reps,
                                   obj_vecs, skt_obj_vecs,
                                   n_img_tri_reps, masks_pred, boxes_pred,
                                   skt_masks_pred, skt_boxes_pred,
                                   n_labels, t_obj_enc, t_obj_enc_n, t_skt_enc,
                                   t_skt_rep, t_obj_rep, t_obj_rep_n)

    def build_model_trainer(self):

        def train(batch):
            # do a forward pass and get predicted image and layouts
            print("[BUILD] building the main training fun")
            with tf.GradientTape() as d_tape, tf.GradientTape() as tape:
                all_losses, losses, obj_losses, d_mask_loss = self.forward_and_compute_losses(next(batch))
                g_loss = losses + obj_losses

            self.mask_dis.trainable = False
            self.optimizer.apply_gradients(zip(tape.gradient(g_loss, self.trainable_variables), self.trainable_variables))
            if not self.hps['abbl_no_mask_gan']:
                self.mask_dis.trainable = True
                self.d_optimizer.apply_gradients(zip(d_tape.gradient(d_mask_loss, self.mask_dis.trainable_variables), self.mask_dis.trainable_variables))

            return all_losses

        return tf.function(train, experimental_relax_shapes=True)

    def train_on_batch(self, iterator):
        return self.model_trainer(iterator)

    def prepare_for_end_of_epoch(self):
        pass

    def train_iterator(self):
        return iter(self.dataset.get_iterator('train', self.hps['batch_size'],
                                              shuffle=50, prefetch=10, repeat=True))
