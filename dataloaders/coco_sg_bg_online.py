import os
import time
import json
import random
from collections import namedtuple

import tensorflow as tf
import numpy as np

import utils
from core.data import BaseDataLoaderV2
from dataloaders.quickdraw_tfrecord_indexed import QuickDrawTFRecordIndexed
from dataloaders.coco_crops_tfrecord import COCOCCropsTFRecord


class COCOBGOnline(BaseDataLoaderV2):
    name = 'coco-bg-tf'

    @classmethod
    def default_hparams(cls):
        hps = utils.hparams.HParams(
            coco_dir='/store/shared/datasets/coco-stuff/',
            qdraw_dir='/store/lribeiro/datasets/tfrecord_quickdraw_indexed',
            val_size=2048,

            scene_size=256,
            min_object_size=0.05,
            min_objects_per_image=1,
            max_objects_per_image=8,
            mask_size=64,
            crop_size=64,
            include_image_obj=False,
            excluded_meta_file='prep_data/coco/mats_objs_sketchables_v2.json'
        )
        return hps

    def __init__(self, hps=None):

        super().__init__(hps)

        # read all the metadata
        train_image_dir = os.path.join(self.hps['coco_dir'], 'images/train2017')
        self.val_image_dir = os.path.join(self.hps['coco_dir'], 'images/val2017')
        train_instances_json = os.path.join(self.hps['coco_dir'], 'annotations/instances_train2017.json')
        train_stuff_json = os.path.join(self.hps['coco_dir'], 'annotations/stuff_train2017.json')
        val_instances_json = os.path.join(self.hps['coco_dir'], 'annotations/instances_val2017.json')
        val_stuff_json = os.path.join(self.hps['coco_dir'], 'annotations/stuff_val2017.json')
        self.image_size = (self.hps['scene_size'], self.hps['scene_size'])

        # load the secondary datasets
        self.qdset = QuickDrawTFRecordIndexed(hps)
        self.crop_size = self.hps['crop_size']
        self.sketch_size = self.qdset.sketch_size
        self.attributes_dim = 35

        print("Loading train metadata...")
        # (object_idx_to_name, object_name_to_idx, objects_list, total_objs,
        #     _, _, _, _) = utils.coco.prepare_and_load_metadata(train_instances_json, train_stuff_json)

        print("Loading validation metadata...")
        (object_idx_to_name, object_name_to_idx, objects_list, total_objs,
         self.valid_image_ids,
         self.valid_image_id_to_filename,
         self.valid_image_id_to_size,
         self.valid_image_id_to_objects) = utils.coco.prepare_and_load_metadata(val_instances_json, val_stuff_json)

        self.test_image_ids = self.valid_image_ids[self.hps['val_size']:]
        self.valid_image_ids = self.valid_image_ids[:self.hps['val_size']]

        with open(self.hps['excluded_meta_file']) as emf:
            materials_metadata = json.load(emf)
            concrete_objs = materials_metadata["objects"]
            allowed_materials = materials_metadata["materials"]
            fully_excluded_objs = materials_metadata["fully_excluded"]
            excluded_materials = materials_metadata["excluded_materials"]

        object_id_to_idx = {ident: i for i, ident in enumerate(objects_list)}

        # include info about the extra __image__ object
        object_id_to_idx[0] = len(objects_list)
        objects_list = np.append(objects_list, 0)

        print("Collecting metadata...")
        PREDICATES_VALUES = ['left of', 'right of', 'above', 'below', 'inside', 'surrounding']
        pred_idx_to_name = ['__in_image__'] + PREDICATES_VALUES
        pred_name_to_idx = {name: idx for idx, name in enumerate(pred_idx_to_name)}
        self.meta = {
            'obj_name_to_ID': object_name_to_idx,
            'obj_ID_to_name': object_idx_to_name,
            'obj_idx_to_ID': objects_list.tolist(),
            'obj_ID_to_idx': object_id_to_idx,
            'obj_idx_to_name': [object_idx_to_name[objects_list[i]] for i in range(len(objects_list))],
            'train_total_objs': total_objs,
            'concrete_objs': concrete_objs,
            'allowed_materials': allowed_materials,
            'fully_excluded_objs': fully_excluded_objs,
            'pred_idx_to_name': pred_idx_to_name,
            'pred_name_to_idx': pred_name_to_idx,
            'excluded_materials': excluded_materials
        }
        self.n_classes = len(objects_list)
        self.pred_idx_to_name = pred_idx_to_name
        self.pred_name_to_idx = pred_name_to_idx
        self.mask_size = self.hps['mask_size']
        self.obj_idx_to_name = self.meta['obj_ID_to_name']
        self.obj_idx_list = objects_list

    def get_paths(self, img_id):
        return img_id, os.path.join(self.val_image_dir, self.valid_image_id_to_filename[img_id.numpy()])

    def process_path(self, img_id, file_path):
        # load the raw data from the file as a string
        try:
            img = tf.io.read_file(file_path)
            img = self.decode_img(img)
        except:
            img = tf.zeros([self.hps['scene_size'], self.hps['scene_size'], 3])
        return img_id, img

    def decode_img(self, img):
        # convert the compressed string to a 3D uint8 tensor
        img = tf.image.decode_jpeg(img, channels=3)
        return img

    def dummy(self):
        return 0., 0, 0., 0., 0, 0, 0, 0, 0., 0.

    def filter_dummy(self, img, objs, boxes, masks, triples, attr, img_id, bg_objs, bg_boxes, bg_masks):
        return img_id != 0

    def build_compute_graph(self):
        def compute_graph(img_id, img):
            size = self.valid_image_id_to_size[img_id.numpy()]
            objs = self.valid_image_id_to_objects[img_id.numpy()]
            img, c_objs, c_boxes, c_masks, c_triples, c_attributes, bg_objs, bg_boxes, bg_masks = utils.coco.preprocess(
                img, size, objs, self.hps, self.meta, return_materials=True)
            if img is None:
                return self.dummy()
            else:
                return (img, c_objs, c_boxes, np.array(c_masks), np.array(c_triples),
                        c_attributes.astype(np.int32), img_id, bg_objs, bg_boxes, bg_masks)
        out_types = [tf.float32, tf.int64, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.int64, tf.float32, tf.float32]
        return compute_graph, out_types

    def get_iterator(self, split_name, shuffle=None, prefetch=5, repeat=False):
        image_ids = self.test_image_ids if split_name == 'test' else self.valid_image_ids
        dataset = tf.data.Dataset.from_tensor_slices(image_ids)
        dataset = dataset.map(lambda *x: tf.py_function(func=self.get_paths, inp=[*x], Tout=[tf.int32, tf.string]))
        dataset = dataset.map(self.process_path, num_parallel_calls=8)

        if repeat:
            dataset = dataset.repeat()

        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        compute_graph, out_types = self.build_compute_graph()
        dataset = dataset.map(lambda *x: tf.py_function(func=compute_graph, inp=[*x], Tout=out_types))
        dataset = dataset.filter(self.filter_dummy)

        qditerators = self.qdset.get_all_class_iterators(split_name, batch_size=None, shuffle=shuffle, repeat=True)
        mix_batch, out_types = self.build_mix_batch(qditerators)
        dataset = dataset.map(lambda *x: tf.py_function(func=mix_batch, inp=[*x], Tout=out_types))
        dataset = dataset.map(self.include_shape_info, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        return dataset.prefetch(prefetch)

    def get_simple_iterator(self, split_name, shuffle=None, prefetch=5, repeat=False):
        dataset = self.splits[split_name].dataset
        if repeat:
            dataset = dataset.repeat()

        if shuffle is not None:
            dataset = dataset.shuffle(shuffle)

        dataset = dataset.map(
            utils.tfrecord.parse_coco_scene_graph_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset.prefetch(prefetch)

    def preprocess_image(self, images):
        return (images - 0.5) * 2

    def deprocess_image(self, images):
        return (images / 2) + 0.5

    def build_mix_batch(self, qditerators):
        def mix_batch(images, objs, boxes, masks, triples, attributes, ids, bg_objs, bg_boxes, bg_masks):

            objs_to_img = tf.zeros((tf.shape(objs)[0],), dtype=tf.int32)
            f_objs = tf.reshape(objs, [-1, 1])
            bg_objs = tf.reshape(bg_objs, [-1, 1])

            # include sketches
            sketches = [None for _ in f_objs]
            for i, obj_y in enumerate(f_objs):
                sketches[i], _ = next(qditerators[self.meta['obj_idx_to_name'][obj_y[0]]])
            sketches = tf.stack(sketches)

            return images, f_objs, boxes, masks, triples, attributes, objs_to_img, sketches, ids, bg_objs, bg_boxes, bg_masks
        out_types = [tf.float32, tf.int64, tf.float32, tf.float32, tf.int32, tf.int32, tf.int32, tf.float32, tf.int32, tf.int64, tf.float32, tf.float32]
        return mix_batch, out_types

    def include_shape_info(self, images, f_objs, f_boxes, f_masks, f_triples, f_attr, objs_to_img, sketches, ids, bg_objs, bg_boxes, bg_masks):
        images, ids = images[None, ...], ids[None, ...]
        images.set_shape([None, self.hps['scene_size'], self.hps['scene_size'], 3])
        f_objs.set_shape([None, 1])
        f_boxes.set_shape([None, 4])
        f_masks.set_shape([None, self.hps['mask_size'], self.hps['mask_size']])
        f_triples.set_shape([None, 3])
        f_attr.set_shape([None, self.attributes_dim])
        objs_to_img.set_shape([None, ])
        sketches.set_shape([None, self.sketch_size, self.sketch_size, 1])
        ids.set_shape([None, ])
        bg_objs.set_shape([None, 1])
        bg_boxes.set_shape([None, 4])
        bg_masks.set_shape([None, self.hps['mask_size'], self.hps['mask_size']])
        return (images - 0.5) * 2, f_objs, f_boxes, f_masks, f_triples, f_attr, objs_to_img, sketches, ids, bg_objs, bg_boxes, bg_masks


def main():
    qdset = COCOBGOnline()

    start_time = time.time()
    iterator = qdset.get_iterator('valid', shuffle=50, repeat=True)
    counter = 0
    for sample in iterator:
        counter += 1
        time_until_now = time.time() - start_time
        print("Processed {} batches in {}s ({}s/b)".format(
            counter, time_until_now, time_until_now / counter))
        # time.sleep(0.1)


if __name__ == '__main__':
    main()
