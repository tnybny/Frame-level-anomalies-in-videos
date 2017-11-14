import tensorflow as tf


class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, shape, num_filters, filter_size, layer_id):
        """
        :param shape: (list) spatial dimensions [H, W]
        :param num_filters: (int) number of output feature maps
        :param filter_size: (list) dims of filter [F, F]
        """
        super(ConvLSTMCell, self).__init__()
        self.shape = shape
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.size = tf.TensorShape(shape + [self.num_filters])
        self.feature_axis = self.size.ndims
        self.layer_id = layer_id

    def call(self, x, state):
        """
        Perform convLSTM cell ops given input at a given state
        :param x: (tensor) input image of shape [batch_size, timesteps, H, W, channels]
        :param state: (tuple) previous memory and hidden states of the cell
        :return: new state after performing convLSTM ops given input and previous state
        """
        c, h = state

        x = tf.concat([x, h], axis=self.feature_axis)
        n = x.shape[-1]
        m = 4 * self.num_filters if self.num_filters > 1 else 4
        W = tf.get_variable("l_weight_" + str(self.layer_id), self.filter_size + [n, m])
        y = tf.nn.convolution(x, W, padding="SAME")
        y += tf.get_variable("l_bias_" + str(self.layer_id), [m], initializer=tf.zeros_initializer())
        j, i, f, o = tf.split(y, 4, axis=self.feature_axis)

        f = tf.sigmoid(f)
        i = tf.sigmoid(i)
        c = c * f + i * tf.nn.tanh(j)

        o = tf.sigmoid(o)
        h = o * tf.nn.tanh(c)

        state = tf.nn.rnn_cell.LSTMStateTuple(c, h)

        return h, state

    @property
    def state_size(self):
        return tf.nn.rnn_cell.LSTMStateTuple(self.size, self.size)

    @property
    def output_size(self):
        return self.size
