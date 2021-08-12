import tensorflow as tf

import builders
import utils
from core.models import BaseModel
from models.mixins import (SketchClassifierMetricMixin,
                           CommonEmbeddingLayersMixin)


class SketchClassifier(BaseModel, SketchClassifierMetricMixin, CommonEmbeddingLayersMixin):
    name = 'sketch-classifier'
    quick_metrics = ['enc_sketch_clas_loss', 'time']
    slow_metrics = ['embedding-tsne', 'embedding-skt-accuracy']

    @classmethod
    def specific_default_hparams(cls):
        hparams = utils.hparams.HParams(
            learning_rate=1e-4,
            encoding_size=128,
            pretrained_e_model='mobilenet2'
        )
        return hparams

    def __init__(self, hps, dataset, out_dir, experiment_id):
        self.losses_manager = builders.losses.LossManager()
        super().__init__(hps, dataset, out_dir, experiment_id)

    def build_model(self):

        # prepare some hyperparams
        self.num_objs = self.dataset.n_classes

        self.encoder = self.build_pretrained_sketch_encoder()
        self.embedding_classifier = tf.keras.layers.Dense(self.num_objs)

        # prepare optimizers
        self.optimizer = tf.keras.optimizers.Adam(
            self.hps['learning_rate'], beta_1=0.5, beta_2=0.999, epsilon=1e-9)

        # finally, build the model trainers
        self.prepare_all_losses()
        self.model_trainer = self.build_model_trainer()
        self.build_vars()

    def build_vars(self):
        self.encoder.build((None, self.dataset.sketch_size, self.dataset.sketch_size, 1))
        self.embedding_classifier.build((None, self.hps['encoding_size']))

    def prepare_all_losses(self):
        self.losses_manager.add_categorical_crossentropy('enc_sketch_clas', weight=1., from_logits=True)
        self.compute_losses = self.build_compute_losses_fun()

    def build_compute_losses_fun(self):
        def compute_losses(objs, skt_enc):

            one_hot_obj = tf.squeeze(tf.one_hot(objs, self.num_objs), axis=1)
            skt_enc_clas = self.embedding_classifier(skt_enc)
            enc_sketch_clas_loss = self.losses_manager.compute_loss('enc_sketch_clas', one_hot_obj, skt_enc_clas)

            all_losses = {'enc_sketch_clas_loss': enc_sketch_clas_loss}
            return all_losses, enc_sketch_clas_loss
        return compute_losses

    def forward_and_compute_losses(self, objs, skts):
        skt_enc = self.encoder(skts)
        return self.compute_losses(objs, skt_enc)

    def build_model_trainer(self):
        @tf.function(input_signature=[
            tf.TensorSpec(shape=(None, 1), dtype=tf.int32),
            tf.TensorSpec(shape=(None, self.dataset.sketch_size, self.dataset.sketch_size, 1), dtype=tf.float32)
        ])
        def train(objs, skts):
            print("[BUILD] building the main training fun")
            with tf.GradientTape(persistent=True) as tape:
                (all_losses, losses) = self.forward_and_compute_losses(objs, skts)

            self.optimizer.apply_gradients(zip(tape.gradient(losses, self.trainable_variables), self.trainable_variables))
            return all_losses
        return train

    def train_on_batch(self, batch):
        sketches, labels = batch
        return self.model_trainer(labels, sketches)
