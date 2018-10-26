import tensorflow as tf
import os

# network architecture definition
CONV1 = 512
CONV2 = 256
CONV3 = 128
DECONV1 = 256
DECONV2 = 512


class Experiment(object):
    def __init__(self, data, alpha, lambd=0.0):
        self.ch = data.nchannels
        self.tvol = data.tvol
        self.x_ = data.next_batch
        self.h, self.w = tf.shape(self.x_)[1], tf.shape(self.x_)[2]
        self.handle = data.handle
        self.tr_iter = data.tr_iter
        self.te_iter = data.te_iter
        self.phase = tf.placeholder(tf.bool, name='is_training')

        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.params = {
            "c_w1": tf.get_variable("c_weight1", shape=[11, 11, self.tvol * self.ch, CONV1], initializer=w_init),
            "c_b1": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV1]), name="c_bias1"),
            "c_w2": tf.get_variable("c_weight2", shape=[5, 5, CONV1, CONV2], initializer=w_init),
            "c_b2": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV2]), name="c_bias2"),
            "c_w3": tf.get_variable("c_weight3", shape=[3, 3, CONV2, CONV3], initializer=w_init),
            "c_b3": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV3]), name="c_bias3"),
            "c_w_3": tf.get_variable("c_weight_3", shape=[3, 3, CONV3, DECONV1], initializer=w_init),
            "c_b_3": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[DECONV1]), name="c_bias_3"),
            "c_w_2": tf.get_variable("c_weight_2", shape=[3, 3, DECONV1, DECONV2], initializer=w_init),
            "c_b_2": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[DECONV2]), name="c_bias_2"),
            "c_w_1": tf.get_variable("c_weight_1", shape=[3, 3, DECONV2, self.tvol * self.ch],
                                     initializer=w_init),
            "c_b_1": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[self.tvol * self.ch]),
                                 name="c_bias_1")
        }

        shapes = []
        self.conved, shapes = self.spatial_encoder(self.x_, shapes)
        self.y = self.spatial_decoder(self.conved, shapes)

        self.per_pixel_recon_errors = tf.square(self.x_ - self.y)
        self.per_frame_recon_errors = tf.reduce_sum(self.per_pixel_recon_errors, axis=[1, 2])

        self.reconstruction_loss = 0.5 * tf.reduce_mean(self.per_frame_recon_errors)

        # l2 regularizer
        self.vars = tf.trainable_variables()
        self.regularization_loss = tf.add_n([tf.nn.l2_loss(v) for v in self.vars if 'bias' not in v.name])

        self.loss = self.reconstruction_loss + lambd * self.regularization_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = tf.train.AdamOptimizer(alpha, epsilon=1e-6).minimize(self.loss)

        self.saver = tf.train.Saver()

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.te_iter.initializer)
        self.training_handle = self.sess.run(data.tr_iter.string_handle())
        self.testing_handle = self.sess.run(data.te_iter.string_handle())

    @staticmethod
    def conv2d(x, w, b, activation=tf.nn.relu, strides=1, phase=True):
        """
        Build a convolutional layer
        :param x: input
        :param w: filter
        :param b: bias
        :param activation: activation func
        :param strides: the stride when filter is scanning through image
        :param phase: training phase or not
        :return: a convolutional layer representation
        """
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        x = activation(x)
        x = tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True, is_training=phase,
                                         updates_collections=None)
        return x

    @staticmethod
    def deconv2d(x, w, b, out_shape, activation=tf.nn.relu, strides=1, phase=True, last=False):
        """
        Build a deconvolutional layer composed of NN-resizing + convolution
        :param x: input
        :param w: filter
        :param b: bias
        :param out_shape: output height and width after NN-resizing
        :param activation: activation func
        :param strides: the stride when filter is scanning
        :param phase: training phase or not
        :param last: last layer of the network or not
        :return: a deconvolutional layer representation
        """
        x = tf.image.resize_images(x, out_shape, method=tf.image.ResizeMethod.BILINEAR, align_corners=True)
        x = tf.nn.conv2d(x, w, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        if not last:
            x = activation(x)
            x = tf.contrib.layers.batch_norm(x, decay=0.99, center=True, scale=True, is_training=phase,
                                             updates_collections=None)
        return x

    def spatial_encoder(self, x, shapes):
        """
        Build a spatial encoder that performs convolutions and poolings
        :param x: tensor of input image of shape (batch_size, HEIGHT, WIDTH, self.tvol * NCHANNELS)
        :param shapes: list of shapes of convolved objects, used to inform deconvolution output shapes
        :return: convolved representation of shape (batch_size, h, w, CONV3)
        """
        conv1 = self.conv2d(x, self.params['c_w1'], self.params['c_b1'], activation=tf.nn.relu, strides=4,
                            phase=self.phase)
        conv2 = self.conv2d(conv1, self.params['c_w2'], self.params['c_b2'], activation=tf.nn.relu, strides=2,
                            phase=self.phase)
        conv3 = self.conv2d(conv2, self.params['c_w3'], self.params['c_b3'], activation=tf.nn.relu, strides=1,
                            phase=self.phase)
        shapes.extend([conv1.get_shape().as_list(), conv2.get_shape().as_list()])
        return conv3, shapes

    def spatial_decoder(self, x, shapes):
        """
        Build a spatial decoder that performs deconvolutions and unpoolings
        :param x: tensor of some transformed representation of input of shape (batch_size, h, w, CONV3)
        :param shapes: list of shapes of convolved objects, used to inform deconvolution output shapes
        :return: deconvolved representation of shape (batch_size, HEIGHT, WEIGHT, self.tvol * NCHANNELS)
        """
        _, newh, neww, _ = shapes[-1]
        deconv1 = self.deconv2d(x, self.params['c_w_3'], self.params['c_b_3'], [newh, neww],
                                activation=tf.nn.relu, strides=1, phase=self.phase)
        _, newh, neww, _ = shapes[-2]
        deconv2 = self.deconv2d(deconv1, self.params['c_w_2'], self.params['c_b_2'], [newh, neww],
                                activation=tf.nn.relu, strides=1, phase=self.phase)
        deconv3 = self.deconv2d(deconv2, self.params['c_w_1'], self.params['c_b_1'], [self.h, self.w],
                                activation=tf.nn.relu, strides=1, phase=self.phase, last=True)
        return deconv3

    def get_loss(self):
        try:
            return self.loss.eval(feed_dict={self.phase: False, self.handle: self.testing_handle},
                                  session=self.sess)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.te_iter.initializer)
            return None

    def batch_train(self):
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.phase: True,
                                                                        self.handle: self.training_handle})
        return loss

    def get_reconstructions(self):
        try:
            return self.sess.run([self.x_, self.y],
                                 feed_dict={self.phase: False, self.handle: self.testing_handle})
        except tf.errors.OutOfRangeError:
            self.sess.run(self.te_iter.initializer)
            return None, None

    def get_recon_errors(self):
        try:
            return self.sess.run([self.per_pixel_recon_errors, self.per_frame_recon_errors],
                                 feed_dict={self.phase: False, self.handle: self.testing_handle})
        except tf.errors.OutOfRangeError:
            self.sess.run(self.te_iter.initializer)
            return None, None

    def save_model(self, path):
        self.saver.save(self.sess, os.path.join(path, "model.ckpt"))

    def restore_model(self, path):
        self.saver.restore(self.sess, os.path.join(path, "model.ckpt"))

    def batch_reconstruct(self):
        try:
            return self.y.eval(feed_dict={self.phase: False, self.handle: self.testing_handle}, session=self.sess)
        except tf.errors.OutOfRangeError:
            self.sess.run(self.te_iter.initializer)
            return None
