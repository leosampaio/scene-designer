import flask
from flask import request
import numpy as np
from flask import jsonify

import os
from collections import OrderedDict

import data
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from util.visualizer import Visualizer
from util import html
import torch

from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import util.util as util
import os

app = flask.Flask(__name__)
app.config["DEBUG"] = True


def preprocess(label, instance_map, image, opt):
    label = Image.fromarray(label)
    params = get_params(opt, label.size)
    transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
    label_tensor = transform_label(label) * 255.0
    label_tensor[label_tensor == 255] = opt.label_nc  # 'unknown' is opt.label_nc

    image = Image.fromarray(image)
    transform_image = get_transform(opt, params)
    image_tensor = transform_image(image)

    # if using instance maps
    if opt.no_instance:
        instance_tensor = 0
    else:
        instance = Image.fromarray(instance_map)
        if instance.mode == 'L':
            instance_tensor = transform_label(instance) * 255
            instance_tensor = instance_tensor.long()
        else:
            instance_tensor = transform_label(instance)

    input_dict = {'label': torch.unsqueeze(label_tensor, 0),
                  'instance': torch.unsqueeze(instance_tensor, 0),
                  'image': torch.unsqueeze(image_tensor, 0),
                  'path': 'null',
                  }
    return input_dict


@app.route('/', methods=['POST'])
def test():
    request_data = request.json
    instances = np.array(request_data['instances'],  dtype=np.uint8)
    layout = np.array(request_data['layout'], dtype=np.uint8)
    real_image = np.array(request_data['real_image'],  dtype=np.uint8)
    model = app.config.get('model')
    print(layout.shape, real_image.shape,  instances.shape)
    model_input = preprocess(
        layout, instances, real_image, app.config.get('opt'))
    generated = model(model_input, mode='inference')
    return jsonify(generated.tolist())

if __name__ == '__main__':
    opt = TestOptions().parse()
    model = Pix2PixModel(opt)
    model.eval()
    app.config['model'] = model
    app.config['opt'] = opt
    app.run()
