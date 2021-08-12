import tensorflow as tf
import numpy as np
import json

import utils


class GraphAttentionRepresentationInferenceMixin(object):

    def inference_image_representation(self, imgs, triples, boxes, attr, obj_to_img):
        crops = utils.bbox.crop_bbox_batch(
            imgs, boxes, obj_to_img, self.dataset.crop_size)
        return self.inference_image_representation_from_crop(tf.shape(imgs)[0], crops, triples, boxes, attr, obj_to_img)

    def inference_image_representation_from_crop(self, n_imgs, crops, triples, boxes, attr, obj_to_img):
        g_obj_encoded = self.multidomain_encoder.obj_encoder.encoder(crops, None, None)
        edges, pred_vecs = self.embedding_model(triples)
        img_reps, obj_co_vecs, img_tri_reps = self.compute_common_encoding(
            g_obj_encoded, pred_vecs, edges, attr, n_imgs, obj_to_img, training=False)
        return img_reps, obj_co_vecs

    def inference_sketch_representation(self, n_imgs, triples, boxes, attr, obj_to_img, sketches):
        skt_obj_encoded = self.multidomain_encoder.skt_encoder.encoder(sketches)
        edges, pred_vecs = self.embedding_model(triples)
        img_skt_reps, skt_co_vecs, img_skt_tri_reps = self.compute_common_encoding(
            skt_obj_encoded, pred_vecs, edges, attr, n_imgs, obj_to_img, training=False)
        return img_skt_reps, skt_co_vecs

    def inference_mixed_representation(self, sketches, crops, boxes):
        if crops is not None:
            img_obj_encoded = self.multidomain_encoder.obj_encoder.encoder(crops, None, None)
        if sketches is not None:
            skt_obj_encoded = self.multidomain_encoder.skt_encoder.encoder(sketches)

        if crops is not None and sketches is not None:
            mixed_encoded = tf.concat([skt_obj_encoded, img_obj_encoded], axis=0)
        elif crops is not None:
            mixed_encoded = img_obj_encoded
        else:
            mixed_encoded = skt_obj_encoded

        triples, attr = utils.graph.compute_scene_graph_for_image(boxes, self.dataset.pred_name_to_idx)
        triples = tf.convert_to_tensor(triples, dtype=tf.int32)
        attr = tf.convert_to_tensor(attr, dtype=tf.int32)
        edges, pred_vecs = self.embedding_model(triples)

        obj_to_img = np.zeros((len(mixed_encoded),), dtype=np.int32)
        img_reps, obj_co_vecs, _ = self.compute_common_encoding(
            mixed_encoded, pred_vecs, edges, attr, 1, obj_to_img, training=False)
        return img_reps, obj_co_vecs

    def inference_representation(self, imgs, objs, triples, masks, boxes, attr, obj_to_img, sketches):
        img_reps, obj_co_vecs = self.inference_image_representation(
            imgs, triples, boxes, attr, obj_to_img)
        img_skt_reps, skt_co_vecs = self.inference_sketch_representation(
            imgs.shape[0], triples, boxes, attr, obj_to_img, sketches)
        return img_reps, img_skt_reps, obj_co_vecs, skt_co_vecs

    def inference_generation(self, obj_co_vecs):
        masks_pred = self.generate_mask(obj_co_vecs)
        boxes_pred = self.box_net(obj_co_vecs)
        return masks_pred, boxes_pred

    def inference_layout_generation(self, obj_co_vecs, objs,
                                    boxes_bg=None, masks_bg=None, objs_bg=None,
                                    rgb=False, gaugan=False):
        masks_pred, boxes_pred = self.inference_generation(obj_co_vecs)
        return self.make_layout(boxes_pred, masks_pred, objs, boxes_bg,
                                masks_bg, objs_bg,
                                rgb, gaugan)

    def make_layout(self, boxes_pred, masks_pred, objs,
                    boxes_bg=None, masks_bg=None, objs_bg=None,
                    rgb=False, gaugan=False):
        if len(objs.shape) == 2:
            objs = tf.squeeze(objs, axis=1)
        one_hot = tf.one_hot(objs, self.num_objs)
        obj_to_img = np.zeros((len(objs),), dtype=np.int32)
        layout = utils.layout.masks_to_layout(
            one_hot, boxes_pred, masks_pred, obj_to_img,
            self.image_size[0], self.image_size[0], 1, test_mode=True).numpy()
        if boxes_bg is not None:
            bg_layout = self.make_bg_layout(boxes_bg, masks_bg, objs_bg)
            layout = utils.layout.combine_layouts(bg_layout, layout)
        if rgb:
            return utils.layout.one_hot_to_rgb(layout)[0]
        elif gaugan:
            layout = utils.layout.convert_layout_to_gaugan(layout, self.dataset.obj_idx_list)
            inst_layout = utils.layout.masks_to_gaugan_instance_layout(
                objs, boxes_pred, masks_pred, self.image_size[0])
            return layout, inst_layout
        else:
            return layout

    def make_bg_layout(self, boxes_bg, masks_bg, objs_bg, rgb=False):
        one_hot_bg = tf.squeeze(tf.one_hot(objs_bg, self.num_objs), axis=1)
        obj_to_img_bg = np.zeros((len(boxes_bg),), dtype=np.int32)
        bg_layout = utils.layout.masks_to_layout(
            one_hot_bg, boxes_bg, masks_bg, obj_to_img_bg,
            self.image_size[0], self.image_size[0], 1, test_mode=True).numpy()
        with open('prep_data/coco/excluded_scoco_with_bg.json') as emf:
            materials_metadata = json.load(emf)
            background_list = materials_metadata["materials"]
        bg_layout = utils.layout.fix_holes_in_layouts(
            bg_layout, self.dataset.obj_idx_to_name, self.dataset.obj_idx_list,
            background_list, dummy_bg=172, offset=0)
        if rgb:
            return utils.layout.one_hot_to_rgb(bg_layout)[0]
        else:
            return bg_layout
