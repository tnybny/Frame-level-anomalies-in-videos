from __future__ import division
import tensorflow as tf
from tensorflow.python.util import nest
from glob import glob
import numpy as np
from PIL import Image
import os


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


def get_mean_frame(dirs, ext):
    num_frames_in_dir = [0] * len(dirs)
    for i in range(len(dirs)):
        num_frames_in_dir[i] = len(glob(os.path.join(dirs[i], '*.' + ext)))
    tot_frames = 5000
    which_dirs = np.random.choice(np.arange(len(dirs)), size=tot_frames, replace=True)
    for i in range(which_dirs.shape[0]):
        f_idx = np.random.randint(0, num_frames_in_dir[which_dirs[i]])
        fnames = sorted(glob(os.path.join(dirs[which_dirs[i]], '*.' + ext)))
        im = np.array(Image.open(fnames[f_idx]), dtype='float64') / 255.
        if im.ndim == 2:
            im = np.expand_dims(im, axis=2)
        if i == 0:
            frames = np.zeros((tot_frames,) + im.shape)
        frames[i] = im
    return np.mean(frames, axis=0)
