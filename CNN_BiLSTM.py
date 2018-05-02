import tensorflow as tf
import numpy as np
import Config
from tensorflow.contrib import rnn


class CNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # alphabet_size = 69

        print "making place holders"
        # Placeholders for input, output and dropout
        # self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.batch_size = tf.placeholder(tf.int32, [],name="batch_size")
        self.pad = tf.placeholder(tf.float32, [None, 1, embedding_size, 1], name='pad')
        self.real_len = tf.placeholder(tf.int32, [None], name='real_len')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        print "making embd layer"
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # self.Q = tf.concat(
            #     [
            #         tf.zeros([1, alphabet_size]),  # Zero padding vector for out of alphabet characters
            #         tf.one_hot(range(alphabet_size), alphabet_size, 1.0, 0.0)
            #         # one-hot vector representation for alphabets
            #     ], 0,
            #     name='Q')
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # self.embedded_chars = tf.nn.embedding_lookup(self.Q, self.input_x)
            embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        print "making conv layer---training starting"

        pooled_outputs = []
        reduced = np.int32(np.ceil((sequence_length) * 1.0 / Config.max_pool_size))
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):

                num_prio = (filter_size - 1) // 2
                num_post = (filter_size - 1) - num_prio
                pad_prio = tf.concat([self.pad] * num_prio, 1)
                pad_post = tf.concat([self.pad] * num_post, 1)
                emb_pad = tf.concat([pad_prio, embedded_chars_expanded, pad_post], 1)

                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                print "filter shape : ",filter_shape
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                # conv = tf.nn.conv2d(
                #     self.embedded_chars_expanded,
                #     W,
                #     strides=[1, 1, 1, 1],
                #     padding="VALID",
                #     name="conv")
                conv = tf.nn.conv2d(
                    emb_pad,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")


                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling over the outputs
                # pooled = tf.nn.max_pool(
                #     h,
                #     ksize=[1, sequence_length - filter_size + 1, 1, 1],
                #     strides=[1, 1, 1, 1],
                #     padding='VALID',
                #     name="pool")
                pooled = tf.nn.max_pool(h, ksize=[1, Config.max_pool_size, 1, 1], strides=[1, Config.max_pool_size, 1, 1],
                                        padding='SAME', name='pool')
                pooled = tf.reshape(pooled, [-1, reduced, num_filters])
                pooled_outputs.append(pooled)
        print "combining pool funcs"
        # Combine all the pooled features
        # num_filters_total = num_filters * len(filter_sizes)
        # print "num filtersssssssss:",num_filters_total
        # self.h_pool = tf.concat(self.h_pool_flat, 2)
        # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # Add dropout
        # with tf.name_scope("dropout"):
            # self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        pooled_concat = tf.concat(pooled_outputs, 2)
        pooled_concat = tf.nn.dropout(pooled_concat, self.dropout_keep_prob)
        print "making dropout layer"

        def lstm_fw_cell():
            lstm_forward_cell = tf.contrib.rnn.BasicLSTMCell(num_units=Config.hidden_unit,state_is_tuple = True,forget_bias=1.0)
            lstm_forward_cell = tf.contrib.rnn.DropoutWrapper(lstm_forward_cell, output_keep_prob=self.dropout_keep_prob)
            return lstm_forward_cell
        lstm_forward_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell() for _ in range(1)],
                                                     state_is_tuple=True)

        self._initial_forward_state = lstm_forward_cell_m.zero_state(self.batch_size, tf.float32)

        def lstm_bw_cell():
            lstm_backward_cell = tf.contrib.rnn.BasicLSTMCell(num_units=Config.hidden_unit,state_is_tuple = True,forget_bias=1.0)
            lstm_backward_cell = tf.contrib.rnn.DropoutWrapper(lstm_backward_cell, output_keep_prob=self.dropout_keep_prob)
            return lstm_backward_cell
        lstm_backward_cell_m = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell() for _ in range(1)],
                                                     state_is_tuple=True)
        self._initial_backward_state = lstm_backward_cell_m.zero_state(self.batch_size, tf.float32)


        inputs = [tf.squeeze(input_, [1]) for input_ in
                  tf.split(pooled_concat, num_or_size_splits=int(reduced), axis=1)]
        # outputs, state = tf.contrib.rnn.static_rnn(lstm_forward_cell, inputs, initial_state=self._initial_state,
        #                                            sequence_length=self.real_len)

        # with tf.name_scope("bw"), tf.variable_scope("bw"):
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_forward_cell_m, lstm_backward_cell_m, inputs,
                                                         initial_state_fw=self._initial_forward_state,
                                                         initial_state_bw=self._initial_backward_state, dtype=tf.float32,
                                                         sequence_length=self.real_len)

        # print tf.Session()
        # Collect the appropriate last words into variable output (dimension = batch x embedding_size)

        output = outputs[0]
        print "outputs shape is .............."
        print len(outputs)
        print "output shape is .............."
        # print tf.Session().run(output.get_shape())
        with tf.variable_scope('Output'):
            tf.get_variable_scope().reuse_variables()
            one = tf.ones([1, Config.hidden_unit*2], tf.float32)
            for i in range(1, len(outputs)):
                ind = self.real_len < (i + 1)
                ind = tf.to_float(ind)
                ind = tf.expand_dims(ind, -1)
                print "ind shape is .............."
                # print tf.Session().run(ind.get_shape())
                mat = tf.matmul(ind, one)
                print "mat shape is .............."
                # print tf.Session().run(mat.get_shape())
                output = tf.add(tf.multiply(output, mat), tf.multiply(outputs[i], 1.0 - mat))

        print "loss calculation"
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            # W = tf.get_variable(
            #     "W",
            #     shape=[num_filters_total, num_classes],
            #     initializer=tf.contrib.layers.xavier_initializer())
            self.Wnew = tf.Variable(tf.truncated_normal([Config.hidden_unit*2, num_classes], stddev=0.1), name='W')
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.scores = tf.nn.xw_plus_b(output, self.Wnew, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        print "calc accuracy"
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")