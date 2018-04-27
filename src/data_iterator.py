import numpy as np
import pickle
import abc

FRAMES_PER_VIDEO = 200


class DataIterator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, p_train, p_test, p_labels, p_pix_mask, batch_size, stride=1, tvol=10, taug=False):
        self.train, self.test, self.labels = np.load(p_train), np.load(p_test), np.load(p_labels)
        with open(p_pix_mask, 'rb') as f:
            self.pix_mask = pickle.load(f)
        self._index = 0
        self._tvol = tvol
        self._taug = taug
        if stride > tvol:
            raise ValueError('The stride can not be greater than temporal volume!')
        self._stride = stride
        self.batch_size = batch_size

    @abc.abstractmethod
    def get_train_batch(self):
        """
        create volumes of videos with temporal augmentation at random
        :return: self.batch_size volumes of videos
        """
        pass

    @abc.abstractmethod
    def get_test_batch(self):
        """
        create sequential volumes of videos of batch_size and for each volume, skipping volume creation by self._stride
        until test set is exhausted
        :return: self.batch_size volumes of videos and index for every frame in these volumes
        """
        pass

    def get_test_labels(self):
        return self.labels

    def get_pix_mask(self):
        return self.pix_mask

    def get_train_size(self):
        return self.train.shape[0]

    def get_test_size(self):
        return self.test.shape[0]

    def check_data_exhausted(self):
        return self._index + self._tvol > self.test.shape[0]

    def reset_index(self):
        self._index = 0


class DataIteratorNormal(DataIterator):
    def __init__(self, p_train, p_test, p_labels, p_pix_mask, batch_size, stride=1, tvol=10, taug=False):
        super(DataIteratorNormal, self).__init__(p_train, p_test, p_labels, p_pix_mask, batch_size, stride, tvol, taug)

    def get_train_batch(self):
        aug_idx = 1
        batch = np.zeros(shape=(self.batch_size, self.train[0].shape[0], self.train[0].shape[1],
                                self.train[0].shape[2] * self._tvol))
        for i in range(self.batch_size):
            vid_idx = np.random.randint(0, self.train.shape[0] / FRAMES_PER_VIDEO)
            if self._taug:
                aug_idx = np.random.randint(1, 4)
            frame_idx = np.random.randint(0, FRAMES_PER_VIDEO - self._tvol * aug_idx)
            batch_i = self.train[(FRAMES_PER_VIDEO * vid_idx + frame_idx):
            (FRAMES_PER_VIDEO * vid_idx + frame_idx + self._tvol * aug_idx):aug_idx]
            batch[i] = np.reshape(np.transpose(batch_i, [1, 2, 0, 3]),
                                  (self.train[0].shape[0], self.train[0].shape[1], self.train[0].shape[2] * self._tvol))
        return batch

    def get_test_batch(self):
        batch = np.zeros(shape=(self.batch_size, self.train[0].shape[0], self.train[0].shape[1],
                                self.train[0].shape[2] * self._tvol))
        frame_indices = np.full(shape=(self.batch_size, self._tvol), fill_value=-1, dtype=np.int)
        for i in range(self.batch_size):
            if not self.check_data_exhausted():
                if self._index % FRAMES_PER_VIDEO + self._tvol > FRAMES_PER_VIDEO:
                    self._index = (self._index / FRAMES_PER_VIDEO + 1) * FRAMES_PER_VIDEO
                batch_i = self.test[self._index:(self._index + self._tvol)]
                batch[i] = np.reshape(np.transpose(batch_i, [1, 2, 0, 3]),
                                      (self.train[0].shape[0], self.train[0].shape[1],
                                       self.train[0].shape[2] * self._tvol))
                frame_indices[i] = np.arange(self._index, self._index + self._tvol)
                self._index += self._stride
            else:
                break
        return batch, frame_indices


class DataIteratorStae(DataIterator):
    def __init__(self, p_train, p_test, p_labels, p_pix_mask, batch_size, stride=1, tvol=10, taug=False):
        super(DataIteratorStae, self).__init__(p_train, p_test, p_labels, p_pix_mask, batch_size, stride, tvol, taug)

    def get_train_batch(self):
        aug_idx = 1
        batch = np.zeros(shape=(self.batch_size, self._tvol) + self.train[0].shape)
        for i in range(self.batch_size):
            vid_idx = np.random.randint(0, self.train.shape[0] / FRAMES_PER_VIDEO)
            if self._taug:
                aug_idx = np.random.randint(1, 4)
            frame_idx = np.random.randint(0, FRAMES_PER_VIDEO - self._tvol * aug_idx)
            batch[i] = self.train[(FRAMES_PER_VIDEO * vid_idx + frame_idx):
                                  (FRAMES_PER_VIDEO * vid_idx + frame_idx + self._tvol * aug_idx):aug_idx]
        return batch

    def get_test_batch(self):
        batch = np.zeros(shape=(self.batch_size, self._tvol) + self.test[0].shape)
        frame_indices = np.full(shape=(self.batch_size, self._tvol), fill_value=-1, dtype=np.int)
        for i in range(self.batch_size):
            if not self.check_data_exhausted():
                if self._index % FRAMES_PER_VIDEO + self._tvol > FRAMES_PER_VIDEO:
                    self._index = (self._index / FRAMES_PER_VIDEO + 1) * FRAMES_PER_VIDEO
                batch[i] = self.test[self._index:(self._index + self._tvol)]
                frame_indices[i] = np.arange(self._index, self._index + self._tvol)
                self._index += self._stride
            else:
                break
        return batch, frame_indices
