import numpy as np

# ground truth of which frames are anomalous by manual image inspection, only test videos have anomalous frames
labels = list()
labels.append(range(60, 180))
labels.append(range(94, 180))
labels.append(range(0, 146))
labels.append(range(30, 180))
labels.append(range(0, 129))
labels.append(range(0, 159))
labels.append(range(45, 180))
labels.append(range(0, 180))
labels.append(range(0, 120))
labels.append(range(0, 150))
labels.append(range(0, 180))
labels.append(range(87, 180))

for i in range(len(labels)):
    labels[i] = np.array(labels[i])
labels = np.array(labels)

np.save('../data.nosync/anomalous_frames_ucsdped2.npy', labels)
