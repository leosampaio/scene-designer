import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _image_float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value.flatten()))


def _int64_list_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def raster_sketch_example(raster_sketch, label):

    feature = {
        'label': _int64_feature(label),
        'size': _int64_feature(raster_sketch.shape[0]),
        'sketch': _image_float_feature(raster_sketch),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_raster_sketch_record(example_proto):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'size': tf.io.FixedLenFeature([], tf.int64),
        'sketch': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    sketch = tf.reshape(parsed['sketch'], [parsed['size'], parsed['size'], 1])
    return sketch, parsed['label']


def coco_scene_graph_example(image, objs, boxes, masks, triples, attributes, identifier):

    feature = {
        'image': _image_float_feature(image),
        'size': _int64_feature(image.shape[0]),
        'n_objs':  _int64_feature(len(objs)),
        'objs': _int64_list_feature(objs),
        'boxes': _image_float_feature(boxes),
        'mask_size': _int64_feature(masks.shape[1]),
        'masks': _image_float_feature(masks),
        'n_triples': _int64_feature(len(triples)),
        'triples': _int64_list_feature(triples.flatten()),
        'attr_size': _int64_feature(attributes.shape[1]),
        'attributes': _int64_list_feature(attributes.flatten()),
        'id': _int64_feature(identifier),
    }

    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_coco_scene_graph_record(example_proto):
    image_feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'n_objs': tf.io.FixedLenFeature([], tf.int64),
        'n_triples': tf.io.FixedLenFeature([], tf.int64),
        'size': tf.io.FixedLenFeature([], tf.int64),
        'mask_size': tf.io.FixedLenFeature([], tf.int64),
        'attr_size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'objs': tf.io.VarLenFeature(tf.int64),
        'boxes': tf.io.VarLenFeature(tf.float32),
        'masks': tf.io.VarLenFeature(tf.float32),
        'triples': tf.io.VarLenFeature(tf.int64),
        'attributes': tf.io.VarLenFeature(tf.int64),

    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    boxes = tf.reshape(
        parsed['boxes'].values, [parsed['n_objs'], 4])
    masks = tf.reshape(
        parsed['masks'].values, [parsed['n_objs'], parsed['mask_size'], parsed['mask_size']])
    triples = tf.reshape(
        parsed['triples'].values, [parsed['n_triples'], 3])
    attributes = tf.reshape(
        parsed['attributes'].values, [parsed['n_objs'], parsed['attr_size']])

    return parsed['n_objs'], parsed['n_triples'], image, parsed['objs'].values, boxes, masks, triples, attributes, parsed['id']


def coco_crop_example(crop, label):
    feature = {
        'image': _image_float_feature(crop),
        'size': _int64_feature(crop.shape[0]),
        'n_objs':  _int64_feature(label)  # typo: should be 'label'
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_coco_crop_record(example_proto):
    image_feature_description = {
        'size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'n_objs': tf.io.FixedLenFeature([], tf.int64),  # typo: sould be 'label'
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    label = parsed['n_objs']

    return image, label


def sketchycoco_crop_example(crop, label, sketch):
    feature = {
        'image': _image_float_feature(crop),
        'size': _int64_feature(crop.shape[0]),
        'label':  _int64_feature(label),
        'sketch': _image_float_feature(sketch),
        'sketch_size': _int64_feature(sketch.shape[0]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_sketchycoco_crop_record(example_proto):
    image_feature_description = {
        'size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sketch': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sketch_size': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    sketch = tf.reshape(parsed['sketch'], [parsed['sketch_size'], parsed['sketch_size'], 1])

    return image, parsed['label'], sketch


def sketchy_example(image, sketches, label):
    feature = {
        'image': _image_float_feature(image),
        'size': _int64_feature(image.shape[0]),
        'label':  _int64_feature(label),
        'sketches': _image_float_feature(sketches),
        'sketch_size': _int64_feature(sketches.shape[1]),
        'n_sketches': _int64_feature(sketches.shape[0]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_sketchy_record(example_proto):
    image_feature_description = {
        'size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sketches': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sketch_size': tf.io.FixedLenFeature([], tf.int64),
        'n_sketches': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    sketches = tf.reshape(parsed['sketches'], [parsed['n_sketches'], parsed['sketch_size'], parsed['sketch_size'], 1])

    return image, parsed['label'], sketches


def sketchy_plus_saliency_example(image, sketches, saliency, label):
    feature = {
        'image': _image_float_feature(image),
        'size': _int64_feature(image.shape[0]),
        'label':  _int64_feature(label),
        'sketches': _image_float_feature(sketches),
        'saliency': _image_float_feature(saliency),
        'sketch_size': _int64_feature(sketches.shape[1]),
        'n_sketches': _int64_feature(sketches.shape[0]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_sketchy_plus_saliency_record(example_proto):
    image_feature_description = {
        'size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sketches': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'saliency': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sketch_size': tf.io.FixedLenFeature([], tf.int64),
        'n_sketches': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    sketches = tf.reshape(parsed['sketches'], [parsed['n_sketches'], parsed['sketch_size'], parsed['sketch_size'], 1])
    saliency = tf.reshape(parsed['saliency'], [parsed['sketch_size'], parsed['sketch_size'], 1])

    return image, parsed['label'], sketches, saliency


def flickr_saliency_example(image, sketch, saliency, label):
    feature = {
        'image': _image_float_feature(image),
        'size': _int64_feature(image.shape[0]),
        'label':  _int64_feature(label),
        'sketch': _image_float_feature(sketch),
        'saliency': _image_float_feature(saliency),
        'sketch_size': _int64_feature(sketch.shape[1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_flickr_saliency_record(example_proto):
    image_feature_description = {
        'size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'sketch': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'saliency': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sketch_size': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    sketch = 1 - tf.reshape(parsed['sketch'], [parsed['sketch_size'], parsed['sketch_size'], 1])
    saliency = tf.reshape(parsed['saliency'], [parsed['sketch_size'], parsed['sketch_size'], 1])

    return image, parsed['label'], sketch, saliency


def qdcoco_fg_example(image, objs, boxes, masks, triples, attributes, identifier, sketches):
    feature = {
        'image': _image_float_feature(image),
        'size': _int64_feature(image.shape[0]),
        'n_objs':  _int64_feature(len(objs)),
        'objs': _int64_list_feature(objs),
        'boxes': _image_float_feature(boxes),
        'mask_size': _int64_feature(masks.shape[1]),
        'masks': _image_float_feature(masks),
        'n_triples': _int64_feature(len(triples)),
        'triples': _int64_list_feature(triples.flatten()),
        'attr_size': _int64_feature(attributes.shape[1]),
        'attributes': _int64_list_feature(attributes.flatten()),
        'id': _int64_feature(identifier),
        'sketches': _image_float_feature(sketches),
        'sketch_size': _int64_feature(sketches.shape[2]),
        'n_sketches': _int64_feature(sketches.shape[1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_qdcoco_fg_record(example_proto):
    image_feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'n_objs': tf.io.FixedLenFeature([], tf.int64),
        'n_triples': tf.io.FixedLenFeature([], tf.int64),
        'size': tf.io.FixedLenFeature([], tf.int64),
        'mask_size': tf.io.FixedLenFeature([], tf.int64),
        'attr_size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'objs': tf.io.VarLenFeature(tf.int64),
        'boxes': tf.io.VarLenFeature(tf.float32),
        'masks': tf.io.VarLenFeature(tf.float32),
        'triples': tf.io.VarLenFeature(tf.int64),
        'attributes': tf.io.VarLenFeature(tf.int64),
        'sketches': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sketch_size': tf.io.FixedLenFeature([], tf.int64),
        'n_sketches': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    boxes = tf.reshape(
        parsed['boxes'].values, [parsed['n_objs'], 4])
    masks = tf.reshape(
        parsed['masks'].values, [parsed['n_objs'], parsed['mask_size'], parsed['mask_size']])
    triples = tf.reshape(
        parsed['triples'].values, [parsed['n_triples'], 3])
    attributes = tf.reshape(
        parsed['attributes'].values, [parsed['n_objs'], parsed['attr_size']])
    sketches = tf.reshape(parsed['sketches'], [parsed['n_objs'], parsed['n_sketches'], parsed['sketch_size'], parsed['sketch_size'], 1])

    return parsed['n_objs'], parsed['n_triples'], image, parsed['objs'].values, boxes, masks, triples, attributes, parsed['id'], sketches


def sketchycoco_scene_graph_example(image, objs, boxes, masks, triples, attributes, identifier, sketches):
    feature = {
        'image': _image_float_feature(image),
        'size': _int64_feature(image.shape[0]),
        'n_objs':  _int64_feature(len(objs)),
        'objs': _int64_list_feature(objs),
        'boxes': _image_float_feature(boxes),
        'mask_size': _int64_feature(masks.shape[1]),
        'masks': _image_float_feature(masks),
        'n_triples': _int64_feature(len(triples)),
        'triples': _int64_list_feature(triples.flatten()),
        'attr_size': _int64_feature(attributes.shape[1]),
        'attributes': _int64_list_feature(attributes.flatten()),
        'id': _int64_feature(identifier),
        'sketches': _image_float_feature(sketches),
        'sketch_size': _int64_feature(sketches.shape[1]),
        'n_sketches': _int64_feature(sketches.shape[0]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_sketchycoco_scene_graph_record(example_proto):
    image_feature_description = {
        'id': tf.io.FixedLenFeature([], tf.int64),
        'n_objs': tf.io.FixedLenFeature([], tf.int64),
        'n_triples': tf.io.FixedLenFeature([], tf.int64),
        'size': tf.io.FixedLenFeature([], tf.int64),
        'mask_size': tf.io.FixedLenFeature([], tf.int64),
        'attr_size': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'objs': tf.io.VarLenFeature(tf.int64),
        'boxes': tf.io.VarLenFeature(tf.float32),
        'masks': tf.io.VarLenFeature(tf.float32),
        'triples': tf.io.VarLenFeature(tf.int64),
        'attributes': tf.io.VarLenFeature(tf.int64),
        'sketches': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'sketch_size': tf.io.FixedLenFeature([], tf.int64),
        'n_sketches': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    image = tf.reshape(parsed['image'], [parsed['size'], parsed['size'], 3])
    boxes = tf.reshape(
        parsed['boxes'].values, [parsed['n_objs'], 4])
    masks = tf.reshape(
        parsed['masks'].values, [parsed['n_objs'], parsed['mask_size'], parsed['mask_size']])
    triples = tf.reshape(
        parsed['triples'].values, [parsed['n_triples'], 3])
    attributes = tf.reshape(
        parsed['attributes'].values, [parsed['n_objs'], parsed['attr_size']])
    sketches = tf.reshape(parsed['sketches'], [parsed['n_objs'], parsed['sketch_size'], parsed['sketch_size'], 1])

    return parsed['n_objs'], parsed['n_triples'], image, parsed['objs'].values, boxes, masks, triples, attributes, parsed['id'], sketches


def token_sketch_example(sketch, label, patch_labels):
    feature = {
        'label': _int64_feature(label),
        'size': _int64_feature(sketch.shape[0]),
        'sketch': _image_float_feature(sketch),
        'patch_labels': _int64_list_feature(patch_labels),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_token_sketch_record(example_proto):
    image_feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'size': tf.io.FixedLenFeature([], tf.int64),
        'sketch': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'patch_labels': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    sketch = tf.reshape(parsed['sketch'], [parsed['size'], parsed['size'], 1])
    return sketch, parsed['label'], parsed['patch_labels']


def gram_matrices_example(g0, g1, g2, g3, g4):
    feature = {
        'g0': _image_float_feature(g0),
        'g0_size': _int64_feature(g0.shape[-1]),
        'g1': _image_float_feature(g1),
        'g1_size': _int64_feature(g1.shape[-1]),
        'g2': _image_float_feature(g2),
        'g2_size': _int64_feature(g2.shape[-1]),
        'g3': _image_float_feature(g3),
        'g3_size': _int64_feature(g3.shape[-1]),
        'g4': _image_float_feature(g4),
        'g4_size': _int64_feature(g4.shape[-1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature))


def parse_gram_matrices_record(example_proto):
    image_feature_description = {
        'g0': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'g0_size': tf.io.FixedLenFeature([], tf.int64),
        'g1': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'g1_size': tf.io.FixedLenFeature([], tf.int64),
        'g2': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'g2_size': tf.io.FixedLenFeature([], tf.int64),
        'g3': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'g3_size': tf.io.FixedLenFeature([], tf.int64),
        'g4': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
        'g4_size': tf.io.FixedLenFeature([], tf.int64),
    }
    parsed = tf.io.parse_single_example(example_proto, image_feature_description)
    return parsed['g0'], parsed['g1'], parsed['g2'], parsed['g3'], parsed['g4']
