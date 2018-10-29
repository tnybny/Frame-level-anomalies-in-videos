from __future__ import print_function
from PIL import Image
import os
from glob import glob
import numpy as np
import re
import ConfigParser
import tensorflow as tf


Config = ConfigParser.ConfigParser()
config_path = os.path.join("config", "config.ini")
Config.read(config_path)
data_dir = Config.get("Default", "DATA_DIR")
ext = Config.get("Default", "EXT")

train_dir = os.path.join(data_dir, "Train")
test_dir = os.path.join(data_dir, "Test")
TVOL = int(Config.get("Default", "TVOL"))

train_dirs = sorted([os.path.join(train_dir, d) for d in os.listdir(train_dir) if re.match(r'Train[0-9][0-9][0-9]$',
                                                                                           d)])
assert len(train_dirs) >= 1
test_dirs = sorted([os.path.join(test_dir, d) for d in os.listdir(test_dir) if re.match(r'Test[0-9][0-9][0-9]$', d)])
assert len(test_dirs) >= 1
train_filename = os.path.join('data.nosync', 'train.tfrecords')
test_filename = os.path.join('data.nosync', 'test.tfrecords')

for split in ['Train', 'Test']:
    dirs = train_dirs if split is 'Train' else test_dirs
    writer = tf.python_io.TFRecordWriter(train_filename) if split is 'Train' \
        else tf.python_io.TFRecordWriter(test_filename)
    j = 0
    for seq_idx in range(len(dirs)):
        print("Reading images from directory:", dirs[seq_idx])
        fnames = sorted(glob(os.path.join(dirs[seq_idx], '*.' + ext)))
        i = 0
        feature = {}
        while i <= len(fnames) - TVOL:
            if ext == 'tif':  # write dense uint8 array
                ims = np.stack([np.array(Image.open(x)) for x in fnames[i:i + TVOL]], axis=0).astype('uint8')
                feature['vid_clip'] = tf.train.Feature(bytes_list=
                                                       tf.train.BytesList(value=[tf.compat.as_bytes(ims.tostring())]))
            else:  # png, jpg, bmp, gif -- write bytes in native encoding
                ims = [tf.gfile.FastGFile(x).read() for x in fnames[i:i + TVOL]]
                for j in range(TVOL):
                    feature['vid_clip_' + str(j).zfill(2)] = \
                        tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(ims[j])]))
            i += 1

            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            j += 1
        print("Total of {0:d} records written from {1} sequence".format(i, seq_idx))
    print("Total of {0:d} records written from {1} split".format(j, split))
    writer.close()
