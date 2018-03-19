from __future__ import division
import tensorflow as tf
from tensorflow.python.util import nest


def compute_eer(far, frr):
    cords = zip(far, frr)
    min_dist = 999999
    for item in cords:
        item_far, item_frr = item
        dist = abs(item_far - item_frr)
        if dist < min_dist:
            min_dist = dist
            eer = (item_far + item_frr) / 2
    return eer


def _with_flat_batch(flat_batch_fn):
    def fn(x, *args, **kwargs):
        shape = tf.shape(x)
        flat_batch_x = tf.reshape(x, tf.concat([[-1], shape[-3:]], axis=0))
        flat_batch_r = flat_batch_fn(flat_batch_x, *args, **kwargs)
        r = nest.map_structure(lambda x: tf.reshape(x, tf.concat([shape[:-3], x.shape[1:]], axis=0)),
                               flat_batch_r)
        return r
    return fn


def structural_dissimilarity(X, Y, data_range=None, K1=0.01, K2=0.03, win_size=11,
                             use_sample_covariance=True):
    """
    Compute structural dissimilarity between images.
    :param X: A 4-D tensor of shape `[batch, height, width, channels]`.
    :param Y: A 4-D tensor of shape `[batch, height, width, channels]`.
    :param data_range: float. The data range of the input image (distance between minimum and maximum possible values).
    :param K1: float, algorithm parameter. See [1].
    :param K2: float, algorithm parameter. See [1].
    :param win_size: The side-length of the sliding window used in comparison.  Must be an odd value.
    :param use_sample_covariance: bool.
    :return: The DSSIM between two image batches of shape `[batch, channels]`
    References
    ----------
    https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_structural_similarity.py
    .. [1] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
       (2004). Image quality assessment: From error visibility to
       structural similarity. IEEE Transactions on Image Processing,
       13, 600-612.
       https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
       DOI:10.1109/TIP.2003.819861
    """
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)

    ndim = 2  # number of spatial dimensions
    nch = tf.shape(X)[-1]

    filter_func = _with_flat_batch(tf.nn.depthwise_conv2d)
    kernel = tf.cast(tf.fill([win_size, win_size, nch, 1], 1 / win_size ** 2), X.dtype)
    filter_args = {'filter': kernel, 'strides': [1] * 4, 'padding': 'VALID'}

    if X.shape.ndims != 4 or Y.shape.ndims != 4:
        raise ValueError("The images must be a 4-D tensor.")

    if data_range is None:
        data_range = 2.0

    NP = win_size ** ndim

    # filter has already normalized by NP
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match Wang et. al. 2004

    # compute means
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)

    # compute variances and covariances
    uxx = filter_func(X * X, **filter_args)
    uyy = filter_func(Y * Y, **filter_args)
    uxy = filter_func(X * Y, **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2

    A1, A2, B1, B2 = ((2 * ux * uy + C1,
                       2 * vxy + C2,
                       ux ** 2 + uy ** 2 + C1,
                       vx + vy + C2))
    D = B1 * B2
    S = (A1 * A2) / D

    dssim = (1 - S) / 2
    dssim = tf.reduce_sum(dssim, axis=(-2, -3))
    return dssim
