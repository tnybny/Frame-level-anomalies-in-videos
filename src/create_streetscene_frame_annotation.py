from __future__ import print_function
import numpy as np
import os
import re
from glob import glob

# ground truth annotation of which frames are anomalous, only test videos have anomalous frames
test_dir = '/home/bramach2/VAD_Datasets/StreetScene/Test'
test_dirs = [os.path.join(test_dir, d) for d in sorted(os.listdir(test_dir)) if re.match(r'Test[0-9][0-9][0-9]$', d)]
print(test_dirs)
labels = list()
for i in range(len(test_dirs)):
    print(test_dirs[i])
    framefiles = glob(os.path.join(test_dirs[i], '*.jpg'))
    gt_trackfile = glob(os.path.join(test_dirs[i], '*_gt.txt'))[0]
    print(gt_trackfile)
    anoms = []
    with open(gt_trackfile) as f:
        for line in f:
            f_idx = int(line.split(' ')[0].split('/')[-1].split('.')[0]) - 1
            track_idx = int(line.split(' ')[1])
            if len(anoms) != track_idx + 1:
                anoms.append([f_idx])
            elif len(anoms) == track_idx + 1:
                anoms[track_idx].append(f_idx)
    flat_list = [item for sublist in anoms for item in sublist]
    flat_list = sorted(list(set(flat_list)))
    print(flat_list)
    labels.append(flat_list)

for i in range(len(labels)):
    labels[i] = np.array(labels[i])
labels = np.array(labels)

np.save('../data.nosync/anomalous_frames_streetscene.npy', labels)
