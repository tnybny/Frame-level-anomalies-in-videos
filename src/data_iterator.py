import os
import abc
import tensorflow as tf
import re
from utils import get_mean_frame


class DataIterator(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_dir, ext, batch_size, tvol=10):
        self.tvol = tvol
        self.batch_size = batch_size
        self.ext = ext
        train_dir = os.path.join(data_dir, 'Train')
        train_dirs = sorted(
            [os.path.join(train_dir, d) for d in os.listdir(train_dir) if re.match(r'Train[0-9][0-9][0-9]$', d)])
        assert len(train_dirs) >= 1

        self.mean_frame = get_mean_frame(train_dirs, ext)
        self.height, self.width, self.nchannels = self.mean_frame.shape

        self.train_filename = os.path.join('data.nosync', 'train.tfrecords')
        self.test_filename = os.path.join('data.nosync', 'test.tfrecords')

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
        self.tr_buffer = tf.data.TFRecordDataset([self.train_filename]).map(self._extract_fn, num_parallel_calls=4)\
            .shuffle(32).repeat().batch(batch_size).map(self._resize_clips, num_parallel_calls=4).prefetch(1)
        self.te_buffer = tf.data.TFRecordDataset([self.test_filename]).map(self._extract_fn, num_parallel_calls=4)\
            .batch(batch_size).map(self._resize_clips, num_parallel_calls=4).prefetch(1)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.tr_buffer.output_types,
                                                            self.tr_buffer.output_shapes)
        self.next_batch = self.iterator.get_next()
        self.tr_iter = self.tr_buffer.make_one_shot_iterator()
        self.te_iter = self.te_buffer.make_initializable_iterator()

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        feature = {}
        if self.ext == 'tif':
            feature['vid_clip'] = tf.FixedLenFeature([], tf.string)
        else:  # bmp, gif, jpg, png
            for j in range(self.tvol):
                feature['vid_clip_' + str(j).zfill(2)] = tf.FixedLenFeature([], tf.string)
        # Extract the data record
        sample = tf.parse_single_example(tfrecord, feature)
        if self.ext == 'tif':
            clip = tf.decode_raw(sample['vid_clip'], tf.uint8)
            clip = tf.cast(tf.reshape(clip, [self.tvol, self.height, self.width, self.nchannels]), tf.float32)
        else:
            clip = tf.cast(tf.stack([tf.image.decode_image(sample['vid_clip_' + str(j).zfill(2)])
                                     for j in range(self.tvol)], axis=0), tf.float32)
        clip = tf.reshape(tf.transpose(clip / tf.constant(255., tf.float32) - self.mean_frame, [1, 2, 0, 3]),
                          [tf.shape(clip)[1], tf.shape(clip)[2], tf.shape(clip)[3] * tf.shape(clip)[0]])
        return clip

    @staticmethod
    def _resize_clips(clips):
        clips = tf.image.resize_bilinear(clips, [227, 227], align_corners=True)
        return clips


class DataIteratorStae(DataIterator):
    def __init__(self, data_dir, ext, batch_size, tvol=10):
        super(DataIteratorStae, self).__init__(data_dir, ext, batch_size, tvol=tvol)
        self.tr_buffer = tf.data.TFRecordDataset([self.train_filename]).map(self._extract_fn, num_parallel_calls=4)\
            .shuffle(32).repeat().batch(batch_size).map(self._resize_clips, num_parallel_calls=4).prefetch(1)
        self.te_buffer = tf.data.TFRecordDataset([self.test_filename]).map(self._extract_fn).batch(batch_size) \
            .map(self._resize_clips).prefetch(1)
        self.handle = tf.placeholder(tf.string, shape=[])
        self.iterator = tf.data.Iterator.from_string_handle(self.handle, self.tr_buffer.output_types,
                                                            self.tr_buffer.output_shapes)
        self.next_batch = self.iterator.get_next()
        self.tr_iter = self.tr_buffer.make_one_shot_iterator()
        self.te_iter = self.te_buffer.make_initializable_iterator()

    def _extract_fn(self, tfrecord):
        # Extract features using the keys set during creation
        feature = {}
        if self.ext == 'tif':
            feature['vid_clip'] = tf.FixedLenFeature([], tf.string)
        else:  # bmp, gif, jpg, png
            for j in range(self.tvol):
                feature['vid_clip_' + str(j).zfill(2)] = tf.FixedLenFeature([], tf.string)
        # Extract the data record
        sample = tf.parse_single_example(tfrecord, feature)
        if self.ext == 'tif':
            clip = tf.decode_raw(sample['vid_clip'], tf.uint8)
            clip = tf.cast(tf.reshape(clip, [self.tvol, self.height, self.width, self.nchannels]), tf.float32)
        else:
            clip = tf.cast(tf.stack([tf.image.decode_image(sample['vid_clip_' + str(j).zfill(2)])
                                     for j in range(self.tvol)], axis=0), tf.float32)
        clip = clip / tf.constant(255., tf.float32) - self.mean_frame
        return clip

    @staticmethod
    def _resize_clips(clips):
        t, h, w, c = tf.shape(clips)[1], tf.shape(clips)[2], tf.shape(clips)[3], tf.shape(clips)[4]
        clips = tf.reshape(tf.transpose(clips, [0, 2, 3, 1, 4]),
                           [tf.shape(clips)[0], h, w, t * c])
        clips = tf.image.resize_bilinear(clips, (227, 227), align_corners=True)
        clips = tf.transpose(tf.reshape(clips, [tf.shape(clips)[0], 227, 227, t, c]), [0, 3, 1, 2, 4])
        return clips
