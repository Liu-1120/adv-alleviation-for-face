"""
The model is adapted from the tensorflow tutorial:
https://www.tensorflow.org/get_started/mnist/pros
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf

from model_resgn import SRGAN_g

from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim

batch_size = 20
class Model(object):
  def __init__(self):
    self.x_input = tf.placeholder('float32', [batch_size, 299, 299, 3], name='x_input_input_to_facenet')
    # self.r_input = tf.placeholder('float32', [batch_size, 299, 299, 3], name='reference_input_input_to_facenet')
    self.y_input = tf.placeholder(tf.int64, [batch_size], name='true_label')
    # self.phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    # input_v3 = self.r_input * 2-1
    # out_160 = set_loader.prewhitenfacenet(self.x_input)
    # t_target_image_160 = set_loader.prewhitenfacenet(self.reference_input)
    #resgn
    # self.net_g = SRGAN_g(self.x_input, is_train=False, reuse=tf.AUTO_REUSE)
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points = inception.inception_v3(
            self.x_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)

    # with slim.arg_scope(inception.inception_v3_arg_scope()):
    #     _, end_points = inception.inception_v3(
    #         self.net_g.outputs, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    self.pre_softmax = end_points['Predictions']
    #calculate the predict label

    # self.pre_softmax = tf.transpose(set_loader.softmax(distance1))    #

    y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=self.y_input, logits=self.pre_softmax)

    self.xent = tf.reduce_sum(y_xent)   #

    self.y_pred = tf.argmax(self.pre_softmax, 1)

    self.correct_prediction = tf.equal(self.y_pred, self.y_input)

    self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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

