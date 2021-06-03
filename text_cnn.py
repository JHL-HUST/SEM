# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification/regression.
    Uses an embedding layer, followed by a convolutional, max-pooling, fully-connected (and softmax) layer.
    """
    def __init__(
      self, sequence_length, model_type='clf', num_classes=2, vocab_size=10000,
      embedding_size=300, filter_sizes=0, num_filters=0, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.int64, [None], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.learning_rate = tf.placeholder(tf.float32, name="learning_rate")

        # self.is_train = tf.placeholder(tf.bool, name='is_train')

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # When trainable parameter equals True the embedding vector is non-static, otherwise is static
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W", trainable=True)
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x) # [None, sequence_length, embedding_size]
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1) # [None, sequence_length, embedding_size, 1]

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob, name='text_vector')
        # self.h_drop = tf.cond(tf.equal(self.is_train, tf.constant(True)), 
        #             lambda: tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob), 
        #             lambda: self.h_pool_flat)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            if model_type == 'clf':
                self.predictions = tf.argmax(self.scores, 1, name="predictions")
            elif model_type == 'reg':
                self.predictions = tf.reduce_max(self.scores, 1, name="predictions")
                self.predictions = tf.expand_dims(self.predictions, -1)

        # Calculate mean cross-entropy loss, or root-mean-square error loss
        with tf.name_scope("loss"):
            if model_type == 'clf':
                losses = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=tf.one_hot(self.input_y, depth=num_classes),logits=self.scores))
                self.loss = losses + l2_reg_lambda * l2_loss
            elif model_type == 'reg':
                print('Not implement yet...')
                # losses = tf.sqrt(tf.losses.mean_squared_error(predictions=self.predictions, labels=self.input_y))
                # self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            if model_type == 'clf':
                self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.input_y, self.predictions), tf.float32))
            elif model_type == 'reg':
                print('Not implement yet!')
                self.accuracy = tf.constant(0.0, name="accuracy")
