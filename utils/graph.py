import numpy as np


def compute_scene_graph_for_image(boxes, pred_name_to_idx):

    # create the location and size attributes
    n_objs = len(boxes)
    size_attribute = np.zeros(
        (n_objs, 10), dtype=np.float32)
    location_attribute = np.zeros(
        (n_objs, 25), dtype=np.float32)

    # compute centers of all objects
    obj_centers = []
    l_root = 25 ** (.5)
    size_attribute_len = 10
    for i, box in enumerate(boxes):
        x0, y0, x1, y1 = box

        mean_x = 0.5 * (x0 + x1)
        mean_y = 0.5 * (y0 + y1)
        obj_centers.append([mean_x, mean_y])

        location_index = round(mean_x * (l_root - 1)) + l_root * round(mean_y * (l_root - 1))
        location_attribute[i, int(location_index)] = 1.0

        w, h = x1 - x0, y1 - y0
        size_index = int(round((size_attribute_len - 1) * (w * h)))
        size_attribute[i, size_index] = 1.0

    obj_centers = np.array(obj_centers)

    if n_objs == 1:  # adds a self-referencing triple to keep single-objects working
        triples = [[0, 0, 0]]
    else:
        # add triples (defining the graph)
        triples = []
        real_objs = np.arange(n_objs)
        for i, cur in enumerate(real_objs):
            choices = real_objs[(i + 1):]
            for other in choices:
                if np.random.random() > 0.5:
                    s, o = cur, other
                else:
                    s, o = other, cur

                # check for inside / surrounding
                sx0, sy0, sx1, sy1 = boxes[s]
                ox0, oy0, ox1, oy1 = boxes[o]
                d = obj_centers[s] - obj_centers[o]
                theta = np.arctan2(d[1], d[0])

                if sx0 < ox0 and sx1 > ox1 and sy0 < oy0 and sy1 > oy1:
                    p = 'surrounding'
                elif sx0 > ox0 and sx1 < ox1 and sy0 > oy0 and sy1 < oy1:
                    p = 'inside'
                elif theta >= 3 * np.pi / 4 or theta <= -3 * np.pi / 4:
                    p = 'left of'
                elif -3 * np.pi / 4 <= theta < -np.pi / 4:
                    p = 'above'
                elif -np.pi / 4 <= theta < np.pi / 4:
                    p = 'right of'
                elif np.pi / 4 <= theta < 3 * np.pi / 4:
                    p = 'below'

                p = pred_name_to_idx[p]
                triples.append([s, p, o])

    attributes = np.concatenate([size_attribute, location_attribute], axis=1)
    return triples, attributes
