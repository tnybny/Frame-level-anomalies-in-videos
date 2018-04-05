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

