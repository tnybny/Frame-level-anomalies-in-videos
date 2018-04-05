import tensorflow as tf


def max_pool_with_argmax(x, stride=2):
    """
    Tensorflow default implementation does not provide gradient operation on max_pool_with_argmax
    Therefore, we use max_pool_with_argmax to extract mask and plain max_pool for max_pooling.
    We assume pooling filter size = stride.
    """
    _, mask = tf.nn.max_pool_with_argmax(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1],
                                         padding='SAME')
    mask = tf.stop_gradient(mask)
    pool = tf.nn.max_pool(x, ksize=[1, stride, stride, 1], strides=[1, stride, stride, 1], padding='SAME')
    return pool, mask


def unpool(input, mask, stride=2):
    assert mask is not None
    ksize = [1, stride, stride, 1]
    input_shape = input.get_shape().as_list()
    #  calculation new shape
    output_shape = (input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3])
    # calculation indices for batch, height, width and feature maps
    one_like_mask = tf.ones_like(mask)
    batch_range = tf.reshape(tf.range(output_shape[0], dtype=tf.int64), shape=[input_shape[0], 1, 1, 1])
    b = one_like_mask * batch_range
    y = mask // (output_shape[2] * output_shape[3])
    x = mask % (output_shape[2] * output_shape[3]) // output_shape[3]
    feature_range = tf.range(output_shape[3], dtype=tf.int64)
    f = one_like_mask * feature_range
    # transpose indices & reshape update values to one dimension
    updates_size = tf.size(input)
    indices = tf.transpose(tf.reshape(tf.stack([b, y, x, f]), [4, updates_size]))
    values = tf.reshape(input, [updates_size])
    ret = tf.scatter_nd(indices, values, output_shape)
    return ret
