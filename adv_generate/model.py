"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import set_loader
import facenet
import importlib
weight_decay = 0.0
keep_probability = 1.0
embedding_size = 128
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
batch_size = 20
class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder('float32', [batch_size, 160, 160, 3], name='x_input_input_to_facenet')
    self.reference_input = tf.placeholder('float32', [batch_size, 160, 160, 3], name='reference_input_input_to_facenet')
    self.y_input = tf.placeholder(tf.int64, [batch_size], name='true_label')
    self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

    out_160 = set_loader.prewhitenfacenet(self.x_input)
    t_target_image_160 = set_loader.prewhitenfacenet(self.reference_input)
    image_batch1 = tf.identity(out_160, 'input')
    image_batch2 = tf.identity(t_target_image_160, 'input')

    model_def = 'inception_resnet_v1'
    network = importlib.import_module(model_def)
    # print('Building training graph')

    # Build the inference graph
    prelogits1, _ = network.inference(image_batch1, keep_probability,
                                      phase_train=self.phase_train_placeholder, bottleneck_layer_size=embedding_size,
                                      weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
    prelogits2, _ = network.inference(image_batch2, keep_probability,
                                      phase_train=self.phase_train_placeholder, bottleneck_layer_size=embedding_size,
                                      weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
    # logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
    #         weights_initializer=slim.initializers.xavier_initializer(),
    #         weights_regularizer=slim.l2_regularizer(args.weight_decay),
    #         scope='Logits', reuse=False)

    embeddings1 = tf.nn.l2_normalize(prelogits1, 1, 1e-10, name='embeddings')
    embeddings2 = tf.nn.l2_normalize(prelogits2, 1, 1e-10, name='embeddings')

    distance1 = tf.reduce_sum(tf.square(embeddings1 - embeddings2), axis=1)
    self.pre_softmax = tf.transpose(set_loader.softmax(distance1))    #
    self.d=distance1
    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)   #

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  @staticmethod
  def _weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

  @staticmethod
  def _bias_variable(shape):
      initial = tf.constant(0.1, shape = shape)
      return tf.Variable(initial)

  @staticmethod
  def _conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

  @staticmethod
  def _max_pool_2x2( x):
      return tf.nn.max_pool(x,
                            ksize = [1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

