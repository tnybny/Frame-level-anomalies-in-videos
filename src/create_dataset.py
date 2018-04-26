import PIL
from PIL import Image
import os
from glob import glob
import numpy as np
import re
from collections import defaultdict
import pickle
import create_labels

train_dir = "../data.nosync/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train"
test_dir = "../data.nosync/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test"

# load images and resize to 227 x 227
train = [np.array(Image.open(y).resize((227, 227), PIL.Image.ANTIALIAS)) for x in sorted(list(os.walk(train_dir)))
         for y in sorted(glob(os.path.join(x[0], '*.tif')))]
test = [np.array(Image.open(y).resize((227, 227), PIL.Image.ANTIALIAS)) for x in sorted(list(os.walk(test_dir)))
        for y in sorted(glob(os.path.join(x[0], '*.tif')))]

# rescale to [0, 1]
train, test = [x / 255. for x in train], [x / 255. for x in test]
train, test = np.asarray(train).astype('float32'), np.asarray(test).astype('float32')
train, test = np.expand_dims(train, axis=train.ndim),  np.expand_dims(test, axis=test.ndim)

# centering
tr_mu = np.mean(train, axis=0)
train, test = train - tr_mu, test - tr_mu

# load pixel level ground truth masks
pix_test = defaultdict(list)
bmp_paths = [y for x in sorted(list(os.walk(test_dir))) for y in sorted(glob(os.path.join(x[0], '*.bmp')))]
reg = re.compile('[0-9][0-9][0-9]')
for p in bmp_paths:
    vidId = int(reg.findall(p.split('/')[-2].split('_')[-2])[0])
    pix_test[vidId].append(np.array(Image.open(p).resize((227, 227))))

np.save('../data.nosync/train.npy', train), np.save('../data.nosync/test.npy', test)
if pix_test:
    with open('../data.nosync/pix_test.pkl', 'wb') as f:
        pickle.dump(pix_test, f, pickle.HIGHEST_PROTOCOL)
