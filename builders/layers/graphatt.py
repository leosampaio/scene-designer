import tensorflow as tf
from ..utils import positional_encoding
from builders.layers.transformer import EncoderLayer, SelfAttnV2, DenseExpander


class GraphEncoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, seq_len,
                 maximum_position_encoding, rate=0.1,
                 use_positional_encoding=True, abblation=False,
                 zero_att_img_emb=False, three_part_pos_enc=False, randomize_pos_enc=False):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.use_pos = use_positional_encoding
        self.is_fake_encoder = abblation

        if not zero_att_img_emb:
            self.embedding_token = tf.Variable(tf.zeros((1, d_model)), aggregation=tf.VariableAggregation.SUM)
        else:
            self.embedding_token = tf.zeros((1, d_model))

        if self.is_fake_encoder:
            return

        self.randomize_pos_enc = randomize_pos_enc

        if self.use_pos:
            self.three_part_pos_enc = three_part_pos_enc
            if three_part_pos_enc:
                self.pos_encoding = [positional_encoding(maximum_position_encoding, self.d_model // 3 + self.d_model % 3),
                                     positional_encoding(maximum_position_encoding, self.d_model // 3),
                                     positional_encoding(maximum_position_encoding, self.d_model // 3)]
            else:
                self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                        self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]
        self.embedding_token = tf.Variable(tf.zeros((1, d_model)), aggregation=tf.VariableAggregation.SUM)

        self.dropout = tf.keras.layers.Dropout(rate)

    def gather_pad_input_and_create_masks(self, obj_vecs, obj_to_img, n_imgs):
        padded = tf.TensorArray(tf.float32, size=n_imgs, element_shape=(self.seq_len, self.d_model))
        masks = tf.TensorArray(tf.float32, size=n_imgs, element_shape=(self.seq_len,))
        for img_id in tf.range(n_imgs):
            cur_obj_vecs = obj_vecs[obj_to_img == img_id]
            if self.use_pos:
                if self.three_part_pos_enc:
                    penc = tf.concat([self.pos_encoding[0][:, -1, :], self.pos_encoding[1][:, -1, :], self.pos_encoding[2][:, -1, :]], -1)
                    embedding_plus_positional = self.embedding_token + penc
                else:
                    embedding_plus_positional = self.embedding_token + self.pos_encoding[..., -1]
            else:
                embedding_plus_positional = self.embedding_token
            cur_obj_vecs = tf.concat((embedding_plus_positional, cur_obj_vecs), axis=0)
            mask = tf.zeros((tf.shape(cur_obj_vecs)[0],))
            padded = padded.write(img_id, tf.pad(cur_obj_vecs, [[0, self.seq_len - tf.shape(cur_obj_vecs)[0]], [0, 0]]))
            masks = masks.write(img_id, tf.pad(mask, [[0, self.seq_len - tf.shape(cur_obj_vecs)[0]]], constant_values=1))

        padded = padded.stack()
        masks = masks.stack()
        masks = masks[:, tf.newaxis, tf.newaxis, :]
        return padded, masks

    def add_positional_encoding(self, obj_vecs, attr):
        location_attr = attr[:, 10:]
        if self.three_part_pos_enc:
            size_attr = attr[:, :10]
            x, y = tf.argmax(location_attr, 1) % 5,  tf.argmax(location_attr, 1) // 5
            x_attr, y_attr = tf.one_hot(x, 5), tf.one_hot(y, 5)
            x_attr, y_attr, size_attr = tf.cast(x_attr, tf.bool), tf.cast(y_attr, tf.bool), tf.cast(size_attr, tf.bool)
            p_enc_x = tf.boolean_mask(tf.tile(self.pos_encoding[0][:, :x_attr.shape[1]], [tf.shape(x_attr)[0], 1, 1]), x_attr)
            p_enc_y = tf.boolean_mask(tf.tile(self.pos_encoding[1][:, :y_attr.shape[1]], [tf.shape(y_attr)[0], 1, 1]), y_attr)
            p_enc_size = tf.boolean_mask(tf.tile(self.pos_encoding[2][:, :size_attr.shape[1]], [tf.shape(size_attr)[0], 1, 1]), size_attr)
            if self.randomize_pos_enc:
                p_enc_x = tf.expand_dims(tf.cast(tf.random.uniform([tf.shape(obj_vecs)[0]]) > 0.5, tf.float32), -1)*p_enc_x
                p_enc_y = tf.expand_dims(tf.cast(tf.random.uniform([tf.shape(obj_vecs)[0]]) > 0.5, tf.float32), -1)*p_enc_y
                p_enc_size = tf.expand_dims(tf.cast(tf.random.uniform([tf.shape(obj_vecs)[0]]) > 0.5, tf.float32), -1)*p_enc_size
            positional_encoding = tf.concat([p_enc_x, p_enc_y, p_enc_size], -1)
        else:
            location_mask = tf.cast(location_attr, tf.bool)
            positional_encoding = tf.boolean_mask(tf.tile(self.pos_encoding[:, :location_attr.shape[1]], [tf.shape(location_mask)[0], 1, 1]), location_mask)
            if self.randomize_pos_enc:
                positional_encoding = tf.expand_dims(tf.cast(tf.random.uniform([tf.shape(obj_vecs)[0]]) > 0.5, tf.float32), -1)*positional_encoding
        return obj_vecs + positional_encoding

    def reflat_obj_vecs(self, x, obj_to_img, n_imgs):
        if tf.executing_eagerly():
            flat_objs = []
            for img_id in tf.range(n_imgs):
                cur_n_objs = tf.reduce_sum(tf.cast(obj_to_img == img_id, tf.int32))
                cur_obj_vecs = x[img_id, 1:cur_n_objs + 1]
                flat_objs.append(cur_obj_vecs)
            return tf.concat(flat_objs, axis=0)
        else:
            flat_objs = tf.TensorArray(tf.float32, size=n_imgs, element_shape=(None, self.d_model,))
            for img_id in tf.range(n_imgs):
                cur_n_objs = tf.reduce_sum(tf.cast(obj_to_img == img_id, tf.int32))
                cur_obj_vecs = x[img_id, 1:cur_n_objs + 1]
                flat_objs = flat_objs.write(img_id, cur_obj_vecs)
            return flat_objs.concat()

    def call(self, obj_vecs, obj_to_img, n_imgs, attr, training):

        if self.is_fake_encoder:
            padded_vecs, masks = self.gather_pad_input_and_create_masks(obj_vecs, obj_to_img, n_imgs)
            obj_vecs = self.reflat_obj_vecs(padded_vecs, obj_to_img, n_imgs)
            x = tf.reduce_sum(padded_vecs, axis=1)
            return x, masks, obj_vecs, masks

        x = obj_vecs * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        if self.use_pos:
            x = self.add_positional_encoding(x, attr)
        padded_vecs, masks = self.gather_pad_input_and_create_masks(x, obj_to_img, n_imgs)

        x = self.dropout(padded_vecs, training=training)

        all_att_weights = []
        for i in range(self.num_layers):
            x, att_weights = self.enc_layers[i](x, training, masks)
            all_att_weights.append(att_weights)

        obj_vecs = self.reflat_obj_vecs(x, obj_to_img, n_imgs)
        x = x[:, 0, :]  # take the sequence that corresponds to the embedding token

        return x, masks, obj_vecs, all_att_weights  # (batch_size, d_model)


class GraphDecoder(tf.keras.layers.Layer):

    def __init__(self, num_layers, d_model, num_heads, dff, seq_len, rate=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

        self.dense_expander = DenseExpander(seq_len)

        self.dropout = tf.keras.layers.Dropout(rate)

    def reflat_obj_vecs(self, x, obj_to_img, n_imgs):
        if tf.executing_eagerly():
            flat_objs = []
            for img_id in tf.range(n_imgs):
                cur_n_objs = tf.reduce_sum(tf.cast(obj_to_img == img_id, tf.int32))
                cur_obj_vecs = x[img_id, 1:cur_n_objs + 1]
                flat_objs.append(cur_obj_vecs)
            return tf.concat(flat_objs, axis=0)
        else:
            flat_objs = tf.TensorArray(tf.float32, size=n_imgs, element_shape=(None, self.d_model,))
            for img_id in tf.range(n_imgs):
                cur_n_objs = tf.reduce_sum(tf.cast(obj_to_img == img_id, tf.int32))
                cur_obj_vecs = x[img_id, 1:cur_n_objs + 1]
                flat_objs = flat_objs.write(img_id, cur_obj_vecs)
            return flat_objs.concat()

    def call(self, x, masks, obj_to_img, n_imgs, training):
        x = self.dense_expander(x)
        x = self.dropout(x, training=training)
        masks = tf.zeros_like(masks)

        all_att_weights = []
        for i in range(self.num_layers):
            x, att_weights = self.enc_layers[i](x, training, masks)
            all_att_weights.append(att_weights)
        x = self.reflat_obj_vecs(x, obj_to_img, n_imgs)

        return x, all_att_weights  # (batch_size, seq_len, d_model)
