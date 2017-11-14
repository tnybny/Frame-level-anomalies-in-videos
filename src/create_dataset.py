from PIL import Image
import os
from glob import glob
import numpy as np
from cv2 import resize


train_dir = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
test_dir = "../data/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"

# load images and resize to 227 x 227
train = [resize(np.array(Image.open(y)), (227, 227)) for x in sorted(list(os.walk(train_dir))) for y in sorted(
    glob(os.path.join(x[0], '*.tif')))]
test = [resize(np.array(Image.open(y)), (227, 227)) for x in sorted(list(os.walk(test_dir))) for y in sorted(
    glob(os.path.join(x[0], '*.tif')))]

# rescale to [0, 1]
train, test = [x / 255. for x in train], [x / 255. for x in test]
train, test = np.asarray(train), np.asarray(test)
train, test = train.reshape((train.shape + (1, ))),  test.reshape((test.shape + (1, )))

# centering
tr_mu, tr_sigma = np.mean(train, axis=0), np.std(train, axis=0)
train, test = (train - tr_mu) / tr_sigma, (test - tr_mu) / tr_sigma

np.save('../data/train.npy', train), np.save('../data/test.npy', test)
