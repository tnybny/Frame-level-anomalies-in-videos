import PIL
from PIL import Image
import os
from glob import glob
import numpy as np


train_dir = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
test_dir = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"

# load images and resize to 227 x 227
train = [np.array(Image.open(y).resize((227, 227), PIL.Image.ANTIALIAS)) for x in sorted(list(os.walk(train_dir)))
         for y in sorted(glob(os.path.join(x[0], '*.tif')))]
test = [np.array(Image.open(y).resize((227, 227), PIL.Image.ANTIALIAS)) for x in sorted(list(os.walk(test_dir)))
        for y in sorted(glob(os.path.join(x[0], '*.tif')))]

# rescale to [0, 1]
train, test = [x / 255. for x in train], [x / 255. for x in test]
train, test = np.asarray(train).astype('float16'), np.asarray(test).astype('float16')
train, test = np.expand_dims(train, axis=train.ndim),  np.expand_dims(test, axis=test.ndim)

# centering
tr_mu = np.mean(train, axis=0)
train, test = train - tr_mu, test - tr_mu

np.save('../data/train.npy', train), np.save('../data/test.npy', test)

clip_params = np.asarray([np.min(train), np.max(train)])
np.save('../data/clip_params.npy', clip_params)
