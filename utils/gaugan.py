import requests
import numpy as np


def generate(layout, instance_layout, server="http://127.0.0.1:5000/"):
    jsondata = {'layout': layout.tolist(),
                'instances': instance_layout.tolist(),
                'real_image': np.zeros((256, 256, 3), dtype=np.uint8).tolist()}

    response = requests.post(server, json=jsondata)
    gen_image = np.transpose(np.array(response.json())[0], (1, 2, 0))
    return (gen_image + 1) / 2
