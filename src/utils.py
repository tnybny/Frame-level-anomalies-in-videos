import tensorflow as tf


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


def structural_dissimilarity(images0, images1, data_range=None, k1=0.01, k2=0.03, kernel_size=11,
                             use_sample_covariance=True):
    """
    Compute structural dissimilarity between images.
    :param images0: A 4-D tensor of shape `[batch, height, width, channels]`.
    :param images1: A 4-D tensor of shape `[batch, height, width, channels]`.
    :param data_range: float. The data range of the input image (distance between minimum and maximum possible values).
    :param k1: float, algorithm parameter. See [1].
    :param k2: float, algorithm parameter. See [1].
    :param kernel_size: The side-length of the sliding window used in comparison.  Must be an odd value.
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
    images0 = tf.convert_to_tensor(images0)
    images1 = tf.convert_to_tensor(images1)

    if images0.shape.ndims != 4 or images1.shape.ndims != 4:
        raise ValueError("The images must be a 4-D tensor.")

    if data_range is None:
        data_range = tf.reduce_max([tf.reduce_max(images0) - tf.reduce_min(images0),
                                    tf.reduce_max(images1) - tf.reduce_min(images1)])

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    # compute patches independently per channel as in the reference implementation
    patches0 = []
    patches1 = []
    # use no padding (i.e. valid padding) so we don't compute values near the borders
    kwargs = dict(ksizes=[1] + [kernel_size] * 2 + [1],
                  strides=[1] * 4, rates=[1] * 4, padding="VALID")
    for image0_single_channel, image1_single_channel in zip(tf.unstack(images0, axis=-1),
                                                            tf.unstack(images1, axis=-1)):
        patches0_single_channel = \
            tf.extract_image_patches(tf.expand_dims(image0_single_channel, -1), **kwargs)
        patches1_single_channel = \
            tf.extract_image_patches(tf.expand_dims(image1_single_channel, -1), **kwargs)
        patches0.append(patches0_single_channel)
        patches1.append(patches1_single_channel)
    patches0 = tf.stack(patches0, axis=-2)
    patches1 = tf.stack(patches1, axis=-2)

    mean0, var0 = tf.nn.moments(patches0, axes=[-1])
    mean1, var1 = tf.nn.moments(patches1, axes=[-1])
    cov01 = tf.reduce_mean(patches0 * patches1, axis=-1) - mean0 * mean1

    if use_sample_covariance:
        NP = kernel_size ** 2  # 2 spatial dimensions
        cov_norm = NP / (NP - 1)  # sample covariance
    else:
        cov_norm = 1.0  # population covariance to match [1]

    var0 *= cov_norm
    var1 *= cov_norm
    cov01 *= cov_norm

    ssim = (2 * mean0 * mean1 + c1) * (2 * cov01 + c2)
    denom = (tf.square(mean0) + tf.square(mean1) + c1) * (var0 + var1 + c2)
    ssim /= denom
    ssim = tf.reduce_mean(ssim, axis=(1, 2))
    dssim = (1 - ssim) / 2
    return dssim
