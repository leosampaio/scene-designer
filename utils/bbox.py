import tensorflow as tf


def crop_bbox_batch(feats, bbox, bbox_to_feats, HH, WW=None):
    if WW is None:
        WW = HH

    expanded_feats = tf.gather(feats, tf.cast(bbox_to_feats, tf.int32))
    crops = crop_bbox(expanded_feats, bbox, HH, WW)

    return crops


def crop_bbox(feats, bbox, HH, WW=None):
    """
    Take differentiable crops of feats specified by bbox.

    Inputs:
    - feats: Tensor of shape (N, C, H, W)
    - bbox: Bounding box coordinates of shape (N, 4) in the format
      [x0, y0, x1, y1] in the [0, 1] coordinate space.
    - HH, WW: Size of the output crops.

    Returns:
    - crops: Tensor of shape (N, C, HH, WW) where crops[i] is the portion of
      feats[i] specified by bbox[i], reshaped to (HH, WW) using bilinear sampling.
    """
    N = tf.shape(feats)[0]
    if WW is None:
        WW = HH

    # change box from [0, 1] to [-1, 1] coordinate system
    bbox = 2 * bbox - 1

    x0, y0 = bbox[:, 0], bbox[:, 1]
    x1, y1 = bbox[:, 2], bbox[:, 3]
    X = tensor_linspace(x0, x1, steps=WW)
    X = tf.broadcast_to(tf.reshape(X, (N, 1, WW)), (N, HH, WW))
    Y = tensor_linspace(y0, y1, steps=HH)
    Y = tf.broadcast_to(tf.reshape(Y, (N, HH, 1)), (N, HH, WW))

    return bilinear_sampler(feats, X, Y)


def _invperm(p):
    N = p.shape[0]
    eye = tf.range(0, N)
    pp = tf.where(tf.equal(eye[:, None], p))[:, 1]
    return pp


def tensor_linspace(start, end, steps=10):
    """
    Vectorized version of torch.linspace.

    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer

    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    view_size = tf.concat([tf.shape(start), [1]], axis=0)
    w_size = tf.concat([tf.ones((len(start.shape)), dtype=tf.int32), [steps]], axis=0)
    out_size = tf.concat([tf.shape(start), [steps]], axis=0)

    start_w = tf.linspace(1., 0., num=steps)
    start_w = tf.broadcast_to(tf.reshape(start_w, (w_size)), out_size)
    end_w = tf.linspace(0., 1., num=steps)
    end_w = tf.broadcast_to(tf.reshape(end_w, (w_size)), out_size)

    start = tf.broadcast_to(
        tf.reshape(tf.cast(start, tf.float32), (view_size)),
        out_size)
    end = tf.broadcast_to(
        tf.reshape(tf.cast(end, tf.float32), (view_size)),
        out_size)

    out = start_w * start + end_w * end
    return out


def bilinear_sampler(img, x, y):
    """
    https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, tf.int32)
    max_x = tf.cast(W - 1, tf.int32)

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, tf.float32)
    y = tf.cast(y, tf.float32)
    x = 0.5 * ((x + 1.0) * tf.cast(max_x - 1, tf.float32))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y - 1, tf.float32))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), tf.int32)
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), tf.int32)
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, 0, max_x)
    x1 = tf.clip_by_value(x1, 0, max_x)
    y0 = tf.clip_by_value(y0, 0, max_y)
    y1 = tf.clip_by_value(y1, 0, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, tf.float32)
    x1 = tf.cast(x1, tf.float32)
    y0 = tf.cast(y0, tf.float32)
    y1 = tf.cast(y1, tf.float32)

    # calculate deltas
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

    return out


def get_pixel_value(img, x, y):
    """
    https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.cast(tf.gather_nd(img, indices), tf.float32)
