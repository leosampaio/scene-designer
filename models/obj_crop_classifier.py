import time

import tensorflow as tf
import numpy as np

import builders
import utils
from utils.hparams import HParams
from core.models import BaseModel
from builders.layers.edge_detectors import CannyEdgeDetector
from models.mixins import (ObjCropClassifierMetricMixin, CommonEmbeddingLayersMixin)


class ObjCropClassifier(BaseModel, ObjCropClassifierMetricMixin, CommonEmbeddingLayersMixin):
    name = 'obj-crop-classifier'
    quick_metrics = ['enc_obj_clas_loss', 'time']
    slow_metrics = ['embedding-tsne', 'crops-obj-accuracy', 'embedding-obj-accuracy']

    @classmethod
    def specific_default_hparams(cls):
        """Return default HParams"""
        hparams = HParams(
            learning_rate=1e-4,
            encoding_size=128,
            pretrained_e_model='mobilenet2'
        )
        return hparams

    def __init__(self, hps, dataset, out_dir, experiment_id):
        self.losses_manager = builders.losses.LossManager()
        super().__init__(hps, dataset, out_dir, experiment_id)

    def build_model(self):

        self.num_objs = self.dataset.n_classes

        self.encoder = self.build_pretrained_image_encoder()
        self.embedding_classifier = tf.keras.layers.Dense(self.num_objs)

        # prepare optimizers
        self.optimizer = tf.keras.optimizers.Adam(
            self.hps['learning_rate'], beta_1=0.5, beta_2=0.999, epsilon=1e-9)

        # finally, build the model trainers
        self.prepare_all_losses()
        self.model_trainer = self.build_model_trainer()

    def prepare_all_losses(self):
        self.losses_manager.add_categorical_crossentropy('enc_obj_clas', weight=1., from_logits=True)

    def compute_losses(self, obj_imgs, objs, obj_enc):
        one_hot_obj = tf.squeeze(tf.one_hot(objs, self.num_objs), axis=1)
        obj_enc_clas = self.embedding_classifier(obj_enc)
        enc_obj_clas_loss = self.losses_manager.compute_loss('enc_obj_clas', one_hot_obj, obj_enc_clas)

        losses = tf.reduce_mean(enc_obj_clas_loss)
        return {'enc_obj_clas_loss': enc_obj_clas_loss}, losses

    def forward_and_compute_losses(self, obj_imgs, objs):
        obj_enc = self.encoder(obj_imgs)
        return self.compute_losses(obj_imgs, objs, obj_enc)

    def build_model_trainer(self):
        crop_size = self.dataset.crop_size

        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, crop_size, crop_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
        ])
        def train(obj_imgs, objs):
            print("[BUILD] building the main training fun")
            with tf.GradientTape(persistent=True) as tape:
                (all_losses, losses) = self.forward_and_compute_losses(
                    obj_imgs, objs)
            self.optimizer.apply_gradients(zip(tape.gradient(losses, self.trainable_variables), self.trainable_variables))

            return all_losses
        return train

    def train_on_batch(self, batch):
        imgs, objs, boxes, masks = batch
        return self.model_trainer(imgs, objs, masks)
