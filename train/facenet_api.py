"""Performs face alignment and calculates L2 distance between the embeddings of images."""

# MIT License
# 
# Copyright (c) 2016 David Sandberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from scipy import misc
import tensorflow as tf
import numpy as np
import sys
import os
import copy
import argparse
import facenet
import importlib
from tensorflow.python.platform import gfile
from inception_resnet_v1 import *
def facenet_api(img):
    # network = importlib.import_module('inception_resnet_v1')


    with tf.Graph().as_default():
        prelogits, _ = inception_resnet_v1_1(img, is_training=False, dropout_keep_prob=0.8,
                                             bottleneck_layer_size=128, reuse=None)

        embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        saver = tf.train.Saver(tf.trainable_variables(), max_to_keep=3)
        with tf.Session() as sess:
            # Load the model
            # images = np.zeros((16, 160, 160, 3))
            #facenet.load_model(model)
            # nrof_samples = len(img)
            # print(nrof_samples)
             #images = np.zeros((nrof_samples, 160, 160, 3))
             #Load the images
            # for i in range(16):
            #     if img.ndim == 2:
            #        img = facenet.to_rgb(img)
            #     img = facenet.prewhiten(img)
            #     img = facenet.crop(img, False, 160)
            #     img = facenet.flip(img, False)
            #     images[i,:,:,:] = img

            # min_pixel = np.min(images)
            # max_pixel = np.max(images)
            # images = (images - min_pixel) / (max_pixel - min_pixel)
            # Get input and output tensors
            model = "/home/fan/face_adv/facenet/src/models/20180402-114759/"
            # model = "/home/fan/face_adv/facenet/src/models/20180402-114759/20180402-114759.pb"
            # Load the model
            # facenet.load_model(model)
            model_exp = os.path.expanduser(model)
            print('Model directory: %s' % model_exp)
            meta_file, ckpt_file = facenet.get_model_filenames(model_exp)

            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)


            saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
            # Build the inference graph


            emb = sess.run(embeddings)
            print(emb)
        return emb

