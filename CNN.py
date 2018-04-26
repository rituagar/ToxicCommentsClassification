import tensorflow as tf
import numpy as np
import Config


class CNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        print "making place holders"
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None,num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        print "making embd layer"
        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)


        # Create a convolution + maxpool layer for each filter size
        print "making conv layer---training starting"

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                print "filter shape : ",filter_shape
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                print "W shape:"
                # print tf.Session().run(W.get_shape())
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                print "b shape:"
                # print tf.Session().run(b.get_shape())
                # print "making conv layer"
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # print "making relu layer"
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # print "h is :",tf.Session().run(h.get_shape());
                # print "making pooling layer"
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                #print "Pooled shape :",tf.Session().run(pooled.get_shape());
        print "combining pool funcs"
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        print "num filtersssssssss:",num_filters_total
        self.h_pool = tf.concat(pooled_outputs, 3)
        # print "h pool shape : ",tf.Session().run(self.h_pool.get_shape())
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
        # print "h pool flat shape : ",tf.Session().run(self.h_pool_flat.get_shape())

        print "making dropout layer"
        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)
            #print "dropout layer : ",tf.Session().run(self.h_drop.get_shape())
        print "loss calculation"
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            # print "output W shape : ",tf.Session().run(W.get_shape())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            # print "output b shape : ",tf.Session().run(b.get_shape())

            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # print "output score shape : ",tf.Session().run(self.scores.get_shape())

            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # print "output pred shape : ",tf.Session().run(self.predictions.get_shape())

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
        print "calc accuracy"
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")