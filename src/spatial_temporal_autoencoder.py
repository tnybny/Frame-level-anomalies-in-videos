import tensorflow as tf
from conv_lstm_cell import ConvLSTMCell

# network architecture definition
NCHANNELS = 1
CONV1 = 128
CONV2 = 64
DECONV1 = 128
DECONV2 = 1
WIDTH = 227
HEIGHT = 227
TVOL = 10
NUM_RNN_LAYERS = 3


class SpatialTemporalAutoencoder(object):
    def __init__(self, alpha, batch_size, lambd):
        self.x_ = tf.placeholder(tf.float32, [None, TVOL, HEIGHT, WIDTH, NCHANNELS])
        self.y_ = tf.placeholder(tf.float32, [None, TVOL, HEIGHT, WIDTH, NCHANNELS])
        self.phase = tf.placeholder(tf.bool, name='is_training')
        # usually y_ = x_ if reconstruction error objective

        self.batch_size = batch_size
        w_init = tf.contrib.layers.xavier_initializer_conv2d()
        self.params = {
            "c_w1": tf.get_variable("c_weight1", shape=[11, 11, NCHANNELS, CONV1], initializer=w_init),
            "c_b1": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV1]), name="c_bias1"),
            "c_w2": tf.get_variable("c_weight2", shape=[5, 5, CONV1, CONV2], initializer=w_init),
            "c_b2": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[CONV2]), name="c_bias2"),
            "c_w3": tf.get_variable("c_weight3", shape=[5, 5, DECONV1, CONV2], initializer=w_init),
            "c_b3": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[DECONV1]), name="c_bias3"),
            "c_w4": tf.get_variable("c_weight4", shape=[11, 11, DECONV2, DECONV1], initializer=w_init),
            "c_b4": tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[DECONV2]), name="c_bias4")
        }

        self.conved = self.spatial_encoder(self.x_)
        self.convLSTMed = self.temporal_encoder_decoder(self.conved)
        self.y = self.spatial_decoder(self.convLSTMed)
        self.y = tf.reshape(self.y, shape=[-1, TVOL, HEIGHT, WIDTH, NCHANNELS])

        self.per_frame_recon_errors = tf.reduce_mean(tf.pow(self.y_ - self.y, 2), axis=[2, 3, 4])

        self.reconstruction_loss = tf.reduce_mean(tf.pow(self.y_ - self.y, 2))
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
        x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
        return activation(x)

    @staticmethod
    def deconv2d(x, w, b, out_shape, activation=tf.nn.relu, strides=1, phase=True, last=False):
        """
        Build a deconvolutional layer
        :param x: input
        :param w: filter
        :param b: bias
        :param out_shape: shape of output tensor
        :param activation: activation func
        :param strides: the stride when filter is scanning
        :param phase: training phase or not
        :param last: last layer of the network or not
        :return: a deconvolutional layer representation
        """
        x = tf.nn.conv2d_transpose(x, w, output_shape=out_shape, strides=[1, strides, strides, 1], padding='VALID')
        x = tf.nn.bias_add(x, b)
        if last:
            return x
        else:
            x = tf.contrib.layers.batch_norm(x, center=True, scale=True, is_training=phase)
            return activation(x)

    def spatial_encoder(self, x):
        """
        Build a spatial encoder that performs convolutions
        :param x: tensor of input image of shape (batch_size, TVOL, HEIGHT, WIDTH, NCHANNELS)
        :return: convolved representation of shape (batch_size * TVOL, h, w, c)
        """
        _, _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=[-1, h, w, c])
        conv1 = self.conv2d(x, self.params['c_w1'], self.params['c_b1'], activation=tf.nn.relu, strides=4,
                            phase=self.phase)
        conv2 = self.conv2d(conv1, self.params['c_w2'], self.params['c_b2'], activation=tf.nn.relu, strides=2,
                            phase=self.phase)
        return conv2

    def temporal_encoder_decoder(self, x):
        """
        Build a temporal encoder-decoder network that uses convLSTM layers to perform sequential operation
        :param x: convolved representation of input volume of shape (batch_size * TVOL, h, w, c)
        :return: convLSTMed representation (batch_size, TVOL, h, w, c)
        """
        _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=[-1, TVOL, h, w, c])
        x = tf.unstack(x, axis=1)
        num_filters = [64, 32, 64]
        filter_sizes = [[3, 3], [3, 3], [3, 3]]
        cell = tf.nn.rnn_cell.MultiRNNCell(
            [ConvLSTMCell(shape=[h, w], num_filters=num_filters[i], filter_size=filter_sizes[i], layer_id=i)
             for i in xrange(NUM_RNN_LAYERS)])
        states_series, _ = tf.nn.static_rnn(cell, x, dtype=tf.float32)
        output = tf.transpose(tf.stack(states_series, axis=0), [1, 0, 2, 3, 4])
        return output

    def spatial_decoder(self, x):
        """
        Build a spatial decoder that performs deconvolutions on the input
        :param x: tensor of some transformed representation of input of shape (batch_size, TVOL, h, w, c)
        :return: deconvolved representation of shape (batch_size * TVOL, HEIGHT, WEIGHT, NCHANNELS)
        """
        _, _, h, w, c = x.get_shape().as_list()
        x = tf.reshape(x, shape=[-1, h, w, c])
        deconv1 = self.deconv2d(x, self.params['c_w3'], self.params['c_b3'],
                                [self.batch_size * TVOL, 55, 55, DECONV1],
                                activation=tf.nn.relu, strides=2, phase=self.phase)
        deconv2 = self.deconv2d(deconv1, self.params['c_w4'], self.params['c_b4'],
                                [self.batch_size * TVOL, HEIGHT, WIDTH, DECONV2],
                                activation=tf.nn.relu, strides=4, phase=self.phase, last=True)
        return deconv2

    def get_loss(self, x, is_training):
        return self.loss.eval(feed_dict={self.x_: x, self.y_: x, self.phase: is_training}, session=self.sess)

    def step(self, x, is_training):
        self.sess.run(self.optimizer, feed_dict={self.x_: x, self.y_: x, self.phase: is_training})

    def get_recon_errors(self, x, is_training):
        return self.per_frame_recon_errors.eval(feed_dict={self.x_: x, self.y_: x, self.phase: is_training},
                                                session=self.sess)

    def save_model(self):
        self.saver.save(self.sess, "models/model.ckpt")

    def restore_model(self):
        self.saver.restore(self.sess, "models/model.ckpt")

    def batch_predict(self, x):
        return self.y.eval(feed_dict={self.x_: x, self.phase: False}, session=self.sess)

    def batch_train(self, xbatch, ybatch=None):
        self.step(xbatch, is_training=True)
        return self.get_loss(xbatch, is_training=False)
