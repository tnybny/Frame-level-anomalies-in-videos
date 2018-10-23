import os
import abc
import tensorflow as tf
import re
from utils import get_mean_frame


class DataIterator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, ext, batch_size, tvol=10):
        self._tvol = tvol
        self.batch_size = batch_size
        train_dir = os.path.join(data_dir, 'Train')
        train_dirs = sorted(
            [os.path.join(train_dir, d) for d in os.listdir(train_dir) if re.match(r'Train[0-9][0-9][0-9]$', d)])
        assert len(train_dirs) >= 1

        self.mean_frame = get_mean_frame(train_dirs, ext)

        self.train_filename = os.path.join('data.nosync', 'train.tfrecords')
        self.test_filename = os.path.join('data.nosync', 'test.tfrecords')

    @abc.abstractmethod
    def get_train_batch(self):
        """
        create volumes of videos at random
        :return: self.batch_size training video clips of length self._tvol
        """
        pass

    @abc.abstractmethod
    def get_test_batch(self):
        """
        create sequential volumes of videos of batch_size and for each volume, until test set is exhausted
        :return: self.batch_size test video clips of length self._tvol
        """
        pass

    @abc.abstractmethod
    def _extract_fn(self, tfrecord):
        """
        parse the tfrecord to obtain information stored as a tf Feature
        :param tfrecord: elements of a TFRecord Dataset
        :return: preprocessed (RGB or grayscale) video clips
        """
        pass


class DataIteratorNormal(DataIterator):
    def __init__(self, data_dir, ext, batch_size, tvol=10):
        super(DataIteratorNormal, self).__init__(data_dir, ext, batch_size, tvol=tvol)
        self.tr_buffer = tf.data.TFRecordDataset([self.train_filename]).map(self._extract_fn).shuffle(300).repeat() \
            .batch(batch_size).map(self._resize_clips).prefetch(1)
        self.te_buffer = tf.data.TFRecordDataset([self.test_filename]).map(self._extract_fn).batch(batch_size)\
            .map(self._resize_clips).prefetch(1)
        self.tr_iter = self.tr_buffer.make_initializable_iterator()
        self.te_iter = self.te_buffer.make_initializable_iterator()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.tr_iter.initializer)
        self.sess.run(self.te_iter.initializer)
        self.next_tr_batch = self.tr_iter.get_next()
        self.next_te_batch = self.te_iter.get_next()

    def get_train_batch(self):
        batch = self.sess.run(self.next_tr_batch)
        return batch

    def get_test_batch(self):
        try:
            batch = self.sess.run(self.next_te_batch)
            return batch
        except tf.errors.OutOfRangeError:
            self.sess.run(self.te_iter.initializer)
            return None

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        feature = {'vid_clip': tf.FixedLenFeature([], tf.string),
                   'nchannels': tf.FixedLenFeature([], tf.int64),
                   'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64)}
        # Extract the data record
        sample = tf.parse_single_example(tfrecord, feature)
        clip = tf.decode_raw(sample['vid_clip'], tf.uint8)
        self.nchannels = sample['nchannels']
        self.height = sample['height']
        self.width = sample['width']
        clip = tf.cast(tf.reshape(clip, tf.cast([self._tvol, self.height, self.width, self.nchannels], tf.int64)),
                       tf.float64)
        clip = tf.reshape(tf.transpose(clip / tf.constant(255., tf.float64) - self.mean_frame, [1, 2, 0, 3]),
                          [tf.shape(clip)[1], tf.shape(clip)[2], tf.shape(clip)[3] * tf.shape(clip)[0]])
        return clip

    @staticmethod
    def _resize_clips(clips):
        clips = tf.image.resize_bilinear(clips, [227, 227], align_corners=True)
        return clips


class DataIteratorStae(DataIterator):
    def __init__(self, data_dir, ext, batch_size, tvol=10):
        super(DataIteratorStae, self).__init__(data_dir, ext, batch_size, tvol=tvol)
        self.tr_buffer = tf.data.TFRecordDataset([self.train_filename]).map(self._extract_fn).shuffle(300).repeat() \
            .batch(batch_size).map(self._resize_clips).prefetch(1)
        self.te_buffer = tf.data.TFRecordDataset([self.test_filename]).map(self._extract_fn).batch(batch_size) \
            .map(self._resize_clips).prefetch(1)
        self.tr_iter = self.tr_buffer.make_initializable_iterator()
        self.te_iter = self.te_buffer.make_initializable_iterator()
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.tr_iter.initializer)
        self.sess.run(self.te_iter.initializer)
        self.next_tr_batch = self.tr_iter.get_next()
        self.next_te_batch = self.te_iter.get_next()

    def get_train_batch(self):
        batch = self.sess.run(self.next_tr_batch)
        return batch

    def get_test_batch(self):
        try:
            batch = self.sess.run(self.next_te_batch)
            return batch
        except tf.errors.OutOfRangeError:
            self.sess.run(self.te_iter.initializer)
            return None

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        feature = {'vid_clip': tf.FixedLenFeature([], tf.string),
                   'nchannels': tf.FixedLenFeature([], tf.int64),
                   'height': tf.FixedLenFeature([], tf.int64),
                   'width': tf.FixedLenFeature([], tf.int64)}
        # Extract the data record
        sample = tf.parse_single_example(tfrecord, feature)
        clip = tf.decode_raw(sample['vid_clip'], tf.uint8)
        self.nchannels = sample['nchannels']
        self.height = sample['height']
        self.width = sample['width']
        clip = tf.cast(tf.reshape(clip, tf.cast([self._tvol, self.height, self.width, self.nchannels], tf.int64)),
                       tf.float64)
        clip = clip / tf.constant(255., tf.float64) - self.mean_frame
        return clip

    @staticmethod
    def _resize_clips(clips):
        clips = tf.reshape(tf.transpose(clips, [0, 2, 3, 1, 4]),
                           [tf.shape(clips)[0], tf.shape(clips)[2], tf.shape(clips)[3],
                            tf.shape(clips)[1] * tf.shape(clips)[4]])
        clips = tf.image.resize_bilinear(clips, (227, 227), align_corners=True)
        clips = tf.transpose(tf.reshape(
            clips,
            [tf.shape(clips)[0], tf.shape(clips)[1], tf.shape(clips)[2], tf.shape(clips)[3], tf.shape(clips)[4]]),
            [0, 3, 1, 2, 4])
        return clips
