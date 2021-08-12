import glob
import argparse
import os
import numpy as np

import sys
from os.path import abspath, dirname, join
sys.path.insert(0, abspath(join(dirname(__file__), '..')))
import utils


def preprocess(data, raster_size):
    preprocessed = []

    for sketch in data:
        sketch = np.array(sketch, dtype=np.float32)

        # removes large gaps from the data
        sketch = np.minimum(sketch, 1000)
        sketch = np.maximum(sketch, -1000)

        # get bounds of sketch and use them to normalise
        min_x, max_x, min_y, max_y = utils.sketch.get_bounds(sketch)
        max_dim = max([max_x - min_x, max_y - min_y, 1])
        sketch[:, :2] /= max_dim

        sketch = convert_sketch_from_stroke3_to_image(sketch, raster_size)

        preprocessed.append(sketch)
    return np.array(preprocessed)


def convert_sketch_from_stroke3_to_image(sketch, raster_size):
    image_size = (raster_size, raster_size)
    lines = utils.tu_sketch_tools.strokes_to_lines(sketch, scale=raster_size / 2, start_from_origin=True)
    lines = utils.tu_sketch_tools.centralise_lines(lines, image_size)
    img = utils.tu_sketch_tools.draw_lines(
        lines, image_shape=image_size, background_pixel=1.0,
        colour=False, line_width=2, typing='float')
    img = np.expand_dims(img, axis=-1)
    return img


def main():
    # Parsing arguments
    parser = argparse.ArgumentParser(
        description='Prepare large dataset for chunked loading')
    parser.add_argument('--dataset-dir')
    parser.add_argument('--target-dir')
    parser.add_argument('--size', type=int, default=96)

    args = parser.parse_args()

    if not os.path.isdir(args.target_dir):
        os.mkdir(args.target_dir)

    meta_file = [f for f in glob.glob("{}/*".format(args.dataset_dir))
                 if os.path.basename(f).startswith('meta')][0]
    meta_dict = np.load(meta_file, allow_pickle=True)
    np.savez(os.path.join(args.target_dir, os.path.basename(meta_file)), **meta_dict)
    # return

    train_f = [f for f in glob.glob("{}/*".format(args.dataset_dir))
               if os.path.basename(f).startswith('train')]
    test_f = [f for f in glob.glob("{}/*".format(args.dataset_dir))
              if os.path.basename(f).startswith('test')]
    valid_f = [f for f in glob.glob("{}/*".format(args.dataset_dir))
               if os.path.basename(f).startswith('valid')]
    all_files = [os.path.basename(f) for f in valid_f + test_f + train_f]

    for i, file in enumerate(all_files):
        print("Loading and processing {} ({}/{})...".format(file, i, len(all_files)))
        loaded_dict = np.load(os.path.join(args.dataset_dir, file), allow_pickle=True)
        data = {'vector_x': loaded_dict['x'],
                'y': loaded_dict['y'],
                'label_names': loaded_dict['label_names']}
        if 'std' in loaded_dict.keys():
            data['std'] = loaded_dict['std']
            data['mean'] = loaded_dict['mean']
        data['x'] = preprocess(data['vector_x'], args.size)
        np.savez(os.path.join(args.target_dir, file), **data)


if __name__ == '__main__':
    main()
