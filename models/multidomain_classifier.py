import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

import dataloaders
import utils
import builders
from core.models import BaseModel
from models.mixins import (MultidomainRepresentationMetricMixin,
                           CommonEmbeddingLayersMixin)
from models.obj_crop_classifier import ObjCropClassifier
from models.sketch_classifier import SketchClassifier


class MultidomainMapper(BaseModel, MultidomainRepresentationMetricMixin, CommonEmbeddingLayersMixin):
    name = 'multidomain-representation'
    quick_metrics = ['enc_obj_clas_loss',
                     'enc_sketch_clas_loss', 'triplet_loss', 'time',
                     'data_time']
    slow_metrics = ['common-embedding-tsne', 'embedding-obj-accuracy', 'embedding-skt-accuracy']

    @classmethod
    def specific_default_hparams(cls):
        """Return default HParams"""
        hparams = utils.hparams.HParams(

            learning_rate=2e-4,
            edge_repr_net_hidden_size=256,
            common_repr_size=128,
            triplet_loss_weight=1.,
            encoding_size=256,
            image_size=96,

            finetune=False,
            finetune_dataset='sketchy-extended-tf',
            finetune_hps='',

            use_focal_loss=False,
            use_euclidean_distance_map=False,
            pretrained_e_model='mobilenet2'
        )
        return hparams

    def __init__(self, hps, dataset, out_dir, experiment_id):
        self.losses_manager = builders.losses.LossManager()
        super().__init__(hps, dataset, out_dir, experiment_id)

    def build_model(self):

        self.num_objs = self.dataset.n_classes
        if self.hps['finetune']:
            FinetuneSet = dataloaders.get_dataloader_by_name(self.hps['finetune_dataset'])
            ft_hps = FinetuneSet.parse_hparams(self.hps['finetune_hps'])
            self.finetune_set = FinetuneSet(ft_hps)
            self.num_objs_finetune = self.finetune_set.n_classes

        self.obj_encoder = ObjCropClassifier(
            self.hps, self.dataset, self.out_dir_root, 'empty')
        self.skt_encoder = SketchClassifier(
            self.hps, self.dataset, self.out_dir_root, 'empty')

        self.repr_net = self.build_common_representation_net()
        self.img_classifier = self.build_image_classifier()
        self.skt_classifier = self.build_sketch_classifier()
        self.full_img_encoder = self.build_image_encoder()
        self.full_skt_encoder = self.build_sketch_encoder()

        # prepare optimizers
        self.optimizer = tf.keras.optimizers.Adam(
            lambda: self.hps['learning_rate'] * tf.maximum(0.2, 1. - (tf.cast(self.current_step, tf.float32) / self.hps['iterations'] * 0.9)),
            beta_1=0.5, beta_2=0.999)

        # finally, build the model trainers
        self.prepare_all_losses()
        self.forward = self.build_forward_fun()

        self.model_trainer = self.build_model_trainer()

    def prepare_all_losses(self):
        if self.hps['use_focal_loss']:
            self.losses_manager.add_focal_class_loss('enc_obj_clas', weight=.5)
            self.losses_manager.add_focal_class_loss('enc_sketch_clas', weight=1.)
        else:
            self.losses_manager.add_categorical_crossentropy('enc_obj_clas', weight=.5, from_logits=True)
            self.losses_manager.add_categorical_crossentropy('enc_sketch_clas', weight=1., from_logits=True)
        self.losses_manager.add_triplet_loss('triplet_loss', weight=self.hps['triplet_loss_weight'], margin=.5)
        self.losses_manager.add_triplet_loss('triplet_loss_harder', weight=self.hps['triplet_loss_weight'], margin=1.)

    def compute_losses(self, objs, objs_n, obj_enc, obj_enc_n, skt_enc, skt_rep, obj_rep, obj_rep_n, finetune=False):

        one_hot_obj = tf.one_hot(objs, self.num_objs)
        one_hot_obj_n = tf.one_hot(objs_n, self.num_objs)
        obj_enc_clas = self.obj_encoder.embedding_classifier(obj_enc)
        obj_enc_clas_n = self.obj_encoder.embedding_classifier(obj_enc_n)
        skt_enc_clas = self.skt_encoder.embedding_classifier(skt_enc)
        triplet_loss = self.losses_manager.compute_loss('triplet_loss', skt_rep, obj_rep, obj_rep_n)

        enc_obj_clas_loss_p = self.losses_manager.compute_loss('enc_obj_clas', one_hot_obj, obj_enc_clas)
        enc_obj_clas_loss_n = self.losses_manager.compute_loss('enc_obj_clas', one_hot_obj_n, obj_enc_clas_n)
        enc_obj_clas_loss = tf.reduce_sum([enc_obj_clas_loss_p, enc_obj_clas_loss_n])
        enc_sketch_clas_loss = self.losses_manager.compute_loss('enc_sketch_clas', one_hot_obj, skt_enc_clas)

        losses = tf.reduce_sum([tf.reduce_mean(l) for l in [enc_obj_clas_loss, enc_sketch_clas_loss, triplet_loss]])

        quick_metrics = {
            'enc_obj_clas_loss': enc_obj_clas_loss,
            'enc_sketch_clas_loss': enc_sketch_clas_loss,
            'triplet_loss': triplet_loss
        }
        return quick_metrics, losses

    def common_project_and_norm(self, enc):
        enc = self.repr_net(enc)
        enc = tf.math.l2_normalize(enc, axis=1)
        return enc

    def encode_sketch(self, sketches):
        if self.hps['use_euclidean_distance_map']:
            sketches = tfa.image.euclidean_dist_transform(tf.cast(sketches > 0.5, tf.uint8))
            sketches = tf.stop_gradient(sketches)
        return self.common_project_and_norm(self.skt_encoder.encoder(sketches))

    def encode_image(self, images):
        return self.common_project_and_norm(self.obj_encoder.encoder(images))

    def build_image_classifier(self):
        input_x = tf.keras.Input((self.hps['image_size'], self.hps['image_size'], 3))
        y_pred = self.obj_encoder.embedding_classifier(self.obj_encoder.encoder(input_x))
        return tf.keras.Model(name='img_classifier', inputs=input_x, outputs=y_pred)

    def build_sketch_classifier(self):
        input_x = tf.keras.Input((self.hps['image_size'], self.hps['image_size'], 1))
        y_pred = self.skt_encoder.embedding_classifier(self.skt_encoder.encoder(input_x))
        return tf.keras.Model(name='skt_classifier', inputs=input_x, outputs=y_pred)

    def build_image_encoder(self):
        input_x = tf.keras.Input((self.hps['image_size'], self.hps['image_size'], 3))
        x = self.common_project_and_norm(self.obj_encoder.encoder(input_x))
        return tf.keras.Model(name='img_encoder', inputs=input_x, outputs=x)

    def build_sketch_encoder(self):
        input_x = tf.keras.Input((self.hps['image_size'], self.hps['image_size'], 1))
        x = self.common_project_and_norm(self.skt_encoder.encoder(input_x))
        return tf.keras.Model(name='skt_encoder', inputs=input_x, outputs=x)

    def build_forward_fun(self):

        def forward(obj_imgs, obj_imgs_n, sketches):
            if self.hps['use_euclidean_distance_map']:
                sketches = tfa.image.euclidean_dist_transform(tf.cast(sketches > 0.5, tf.uint8))
                sketches = tf.stop_gradient(sketches)

            obj_encoded = self.obj_encoder.encoder(obj_imgs)
            obj_encoded_n = self.obj_encoder.encoder(obj_imgs_n)
            skt_encoded = self.skt_encoder.encoder(sketches)

            obj_rep = self.common_project_and_norm(obj_encoded)
            obj_rep_n = self.common_project_and_norm(obj_encoded_n)
            skt_rep = self.common_project_and_norm(skt_encoded)

            return obj_encoded, obj_encoded_n, skt_encoded, skt_rep, obj_rep, obj_rep_n
        return forward

    def forward_and_compute_losses(self, sketches, p_crops, n_crops, p_class, n_class, finetune=False):
        out = self.forward(p_crops, n_crops, sketches)
        return self.compute_losses(p_class, n_class, *out, finetune)

    def build_model_trainer(self):
        signature = [
            tf.TensorSpec(shape=(None, self.hps['image_size'], self.hps['image_size'], 1), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.hps['image_size'], self.hps['image_size'], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, self.hps['image_size'], self.hps['image_size'], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int64),
            tf.TensorSpec(shape=(None,), dtype=tf.int64)]

        @tf.function(input_signature=signature)
        def train(sketch, p_crop, n_crop, p_class, n_class):
            print("[BUILD] building the main training fun")
            with tf.GradientTape(persistent=True) as tape:
                (all_losses, losses) = self.forward_and_compute_losses(
                    sketch, p_crop, n_crop, p_class, n_class)
            self.optimizer.apply_gradients(zip(tape.gradient(losses, self.trainable_variables), self.trainable_variables))
            return all_losses

        return train

    def train_on_batch(self, iterator):
        sketch, p_crop, n_crop, p_class, n_class = next(iterator)
        return self.model_trainer(sketch, p_crop, n_crop, p_class, n_class)

    def train_iterator(self):
        if self.hps['finetune']:
            main_iterator = iter(self.dataset.get_iterator(
                'train', self.hps['batch_size'],
                shuffle=50, prefetch=1, repeat=True, image_size=self.hps['image_size']))
            finetune_iterator = iter(self.finetune_set.get_iterator(
                'train', self.hps['batch_size'],
                shuffle=50, prefetch=1, repeat=True, image_size=self.hps['image_size']))

            def iterator():
                while True:
                    n_fine = int(self.p_sk() * self.hps['batch_size'])
                    n_main = self.hps['batch_size'] - n_fine
                    sketch_m, p_crop_m, n_crop_m, p_class_m, n_class_m = next(main_iterator)
                    sketch_f, p_crop_f, n_crop_f, p_class_f, n_class_f = next(finetune_iterator)
                    sketch = np.concatenate([sketch_m.numpy()[:n_main], sketch_f.numpy()[:n_fine]], 0)
                    p_crop = np.concatenate([p_crop_m.numpy()[:n_main], p_crop_f.numpy()[:n_fine]], 0)
                    n_crop = np.concatenate([n_crop_m.numpy()[:n_main], n_crop_f.numpy()[:n_fine]], 0)
                    p_class = np.concatenate([p_class_m.numpy()[:n_main], p_class_f.numpy()[:n_fine]], 0)
                    n_class = np.concatenate([n_class_m.numpy()[:n_main], n_class_f.numpy()[:n_fine]], 0)
                    yield sketch, p_crop, n_crop, p_class, n_class
            return iterator()
        else:
            return iter(self.dataset.get_iterator('train', self.hps['batch_size'],
                                                  shuffle=100, prefetch=1, repeat=True))

    def p_sk(self, lambda_var=1.):
        return 0.1 + tf.minimum(0.5, tf.pow(self.current_step / self.hps['iterations'],
                                            lambda_var))
