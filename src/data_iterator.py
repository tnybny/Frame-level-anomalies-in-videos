import numpy as np

FRAMES_PER_VIDEO = 200
TVOL = 10


class DataIterator(object):
    def __init__(self, p_train, p_test, p_labels, batch_size, stride=1):
        self.train, self.test, self.labels = np.load(p_train), np.load(p_test), np.load(p_labels)
        self._index = 0
        if stride > TVOL:
            raise ValueError('The stride can not be greater than temporal volume!')
        self._stride = stride
        self.batch_size = batch_size

    def get_train_batch(self):
        """
        create volumes of videos with temporal augmentation at random
        :return: self.batch_size volumes of videos
        """
        batch = np.zeros(shape=(self.batch_size, TVOL) + self.train[0].shape)
        for i in xrange(self.batch_size):
            vid_idx = np.random.randint(0, self.train.shape[0] / FRAMES_PER_VIDEO)
            aug_idx = np.random.randint(1, 4)
            frame_idx = np.random.randint(0, FRAMES_PER_VIDEO - TVOL * aug_idx)
            batch[i] = self.train[(FRAMES_PER_VIDEO * vid_idx + frame_idx):
                                  (FRAMES_PER_VIDEO * vid_idx + frame_idx + TVOL * aug_idx):aug_idx]
        return batch

    def get_test_batch(self):
        """
        create sequential volumes of videos of batch_size and for each volume, skipping volume creation by self._stride
        until test set is exhausted
        :return: self.batch_size volumes of videos and index for every frame in these volumes
        """
        batch = np.zeros(shape=(self.batch_size, TVOL) + self.test[0].shape)
        frame_indices = np.full(shape=(self.batch_size, TVOL), fill_value=-1, dtype=np.int)
        for i in xrange(self.batch_size):
            if not self.check_data_exhausted():
                if self._index % FRAMES_PER_VIDEO + TVOL > FRAMES_PER_VIDEO:
                    self._index = (self._index / FRAMES_PER_VIDEO + 1) * FRAMES_PER_VIDEO
                batch[i] = self.test[self._index:(self._index + TVOL)]
                frame_indices[i] = np.arange(self._index, self._index + TVOL)
                self._index += self._stride
            else:
                break
        return batch, frame_indices

    def get_test_labels(self):
        return self.labels

    def get_train_size(self):
        return self.train.shape[0]

    def get_test_size(self):
        return self.test.shape[0]

    def check_data_exhausted(self):
        return self._index + TVOL > self.test.shape[0]

    def reset_index(self):
        self._index = 0
