import tensorflow as tf
import os
from tensorflow.python.ops import init_ops
from max_unpool import max_pool_with_argmax, unpool

# network architecture definition
NCHANNELS = 1
CONV1 = 512
CONV2 = 256
CONV3 = 128
DECONV1 = 256
DECONV2 = 512
WIDTH = 227
HEIGHT = 227


class ConvAE2d(object):
    def __init__(self, tvol, alpha, batch_size, lambd):
        self.tvol = tvol
        self.x_ = tf.placeholder(tf.float32, [None, HEIGHT, WIDTH, self.tvol * NCHANNELS])
        self.phase = tf.placeholder(tf.bool, name='is_training')

        self.batch_size = batch_size
        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.params = {
            "c_w1": tf.get_variable("c_weight1", shape=[15, 15, NCHANNELS * self.tvol, CONV1], initializer=w_init),
            "c_b1": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV1]), name="c_bias1"),
            "c_w2": tf.get_variable("c_weight2", shape=[4, 4, CONV1, CONV2], initializer=w_init),
            "c_b2": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV2]), name="c_bias2"),
            "c_w3": tf.get_variable("c_weight3", shape=[3, 3, CONV2, CONV3], initializer=w_init),
            "c_b3": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV3]), name="c_bias3"),
            "c_w_3": tf.get_variable("c_weight_3", shape=[3, 3, DECONV1, CONV3], initializer=w_init),
            "c_b_3": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[DECONV1]), name="c_bias_3"),
            "c_w_2": tf.get_variable("c_weight_2", shape=[4, 4, DECONV2, DECONV1], initializer=w_init),
            "c_b_2": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[DECONV2]), name="c_bias_2"),
            "c_w_1": tf.get_variable("c_weight_1", shape=[15, 15, self.tvol * NCHANNELS, DECONV2], initializer=w_init),
            "c_b_1": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.tvol * NCHANNELS]), name="c_bias_1")
        }

        shapes, masks = [], []
        self.conved, shapes, masks = self.spatial_encoder(self.x_, shapes, masks)
        self.y = self.spatial_decoder(self.conved, shapes, masks)

        self.per_frame_recon_errors = tf.reduce_sum(tf.square(self.x_ - self.y), axis=[1, 2])

        self.reconstruction_loss = 0.5 * tf.reduce_mean(self.per_frame_recon_errors)
        self.vars = tf.trainable_variables()
        self.regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if 'bias' not in v.name])
        self.loss = self.reconstruction_loss + lambd * self.regularization_loss
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(alpha, epsilon=1e-6).minimize(self.loss)

        self.saver = tf.train.Saver()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    @staticmethod
    def conv2d(x, w, b, activation=tf.nn.tanh, strides=1, pad='VALID', phase=True):
        """
        Build a convolutional layer. Does convolution, activation, batchnorm and pooling
        :param x: input
        :param w: filter
        :param b: bias
        :param activation: activation func
        :param strides: the stride when filter is scanning through image
        :param pad: whether padding is of type 'SAME' or 'VALID'
        :param phase: training phase or not
        :return: a convolutional layer representation
        """
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding=pad)
        x = tf.nn.bias_add(x, b)
        param_init = {'beta': init_ops.constant_initializer(0.75), 'gamma': init_ops.constant_initializer(1e-4)}
        x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=phase,
                                         updates_collections=None, param_initializers=param_init)
        return activation(x)

    @staticmethod
    def deconv2d(x, w, b, out_shape, activation=tf.nn.tanh, strides=1, pad='VALID', phase=True, last=False):
        """
        Build a deconvolutional layer
        :param x: input
        :param w: filter
        :param b: bias
        :param out_shape: shape of output tensor after deconvolution
        :param activation: activation func
        :param strides: the stride when filter is scanning
        :param pad: whether padding is of type 'SAME' or 'VALID'
        :param phase: training phase or not
        :param last: last layer of the network or not
        :return: a deconvolutional layer representation
        """
        x = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=[1, strides, strides, 1], padding=pad)
        x = tf.nn.bias_add(x, b)
        if last:
            return activation(x)
        else:
            param_init = {'beta': init_ops.constant_initializer(0.75), 'gamma': init_ops.constant_initializer(1e-4)}
            x = tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, is_training=phase,
                                             updates_collections=None, param_initializers=param_init)
            return activation(x)

    def spatial_encoder(self, x, shapes, masks):
        """
        Build a spatial encoder that performs convolutions and poolings
        :param x: tensor of input image of shape (batch_size, HEIGHT, WIDTH, self.tvol * NCHANNELS)
        :param shapes: list of shapes of convolved objects, used to inform deconvolution output shapes
        :param masks: list of argmax masks from max pooling ops
        :return: convolved representation of shape (batch_size, h, w, CONV3)
        """
        conv1 = self.conv2d(x, self.params['c_w1'], self.params['c_b1'], activation=tf.nn.tanh, strides=4,
                            pad='VALID', phase=self.phase)
        pool1, mask1 = max_pool_with_argmax(conv1)
        conv2 = self.conv2d(pool1, self.params['c_w2'], self.params['c_b2'], activation=tf.nn.tanh, strides=1,
                            pad='VALID', phase=self.phase)
        pool2, mask2 = max_pool_with_argmax(conv2)
        conv3 = self.conv2d(pool2, self.params['c_w3'], self.params['c_b3'], activation=tf.nn.tanh, strides=1,
                            pad='VALID', phase=self.phase)
        shapes.extend([conv1.get_shape().as_list(), pool1.get_shape().as_list(), conv2.get_shape().as_list(),
                       pool2.get_shape().as_list()])
        masks.extend([mask1, mask2])
        return conv3, shapes, masks

    def spatial_decoder(self, x, shapes, masks):
        """
        Build a spatial decoder that performs deconvolutions and unpoolings
        :param x: tensor of some transformed representation of input of shape (batch_size, h, w, CONV3)
        :param shapes: list of shapes of convolved objects, used to inform deconvolution output shapes
        :param masks: list of argmax masks from max pooling ops for max unpooling ops
        :return: deconvolved representation of shape (batch_size, HEIGHT, WEIGHT, self.tvol * NCHANNELS)
        """
        _, newh, neww, _ = shapes[-1]
        deconv1 = self.deconv2d(x, self.params['c_w_3'], self.params['c_b_3'],
                                [self.batch_size, newh, neww, DECONV1],
                                activation=tf.nn.tanh, strides=1, pad='VALID', phase=self.phase)
        mask = masks[-1]
        unpool1 = unpool(deconv1, mask=mask)
        _, newh, neww, _ = shapes[-3]
        deconv2 = self.deconv2d(unpool1, self.params['c_w_2'], self.params['c_b_2'],
                                [self.batch_size, newh, neww, DECONV2],
                                activation=tf.nn.tanh, strides=1, pad='VALID', phase=self.phase)
        mask = masks[-2]
        unpool2 = unpool(deconv2, mask=mask)
        deconv3 = self.deconv2d(unpool2, self.params['c_w_1'], self.params['c_b_1'],
                                [self.batch_size, HEIGHT, WIDTH, self.tvol * NCHANNELS],
                                activation=tf.nn.tanh, strides=4, pad='VALID', phase=self.phase, last=True)
        return deconv3

    def get_loss(self, x, is_training):
        return self.loss.eval(feed_dict={self.x_: x, self.phase: is_training}, session=self.sess)

    def step(self, x, is_training):
        self.sess.run(self.optimizer, feed_dict={self.x_: x, self.phase: is_training})

    def get_reconstructions(self, x, is_training):
        return self.sess.run([self.y, self.per_frame_recon_errors], feed_dict={self.x_: x, self.phase: is_training})

    def get_recon_errors(self, x, is_training):
        return self.per_frame_recon_errors.eval(feed_dict={self.x_: x, self.phase: is_training},
                                                session=self.sess)

    def save_model(self, path):
        self.saver.save(self.sess, os.path.join(path, "model.ckpt"))

    def restore_model(self, path):
        self.saver.restore(self.sess, os.path.join(path, "model.ckpt"))

    def batch_reconstruct(self, x):
        return self.y.eval(feed_dict={self.x_: x, self.phase: False}, session=self.sess)

    def batch_train(self, xbatch):
        self.step(xbatch, is_training=True)
        return self.get_loss(xbatch, is_training=False)
