"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from scipy.misc import imsave
import tensorflow as tf
import numpy as np
import os
import facenet
import set_loader
def save_img(images, filepaths, output_dir):

    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (images[i, :, :, :] * 255.0).astype(np.uint8)
            imsave(f, img, format='PNG')

class LinfPGDAttack:
  def __init__(self, model, epsilon, k, a, random_start, loss_func):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.k = k
    self.a = a
    self.rand = random_start

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == 'cw':
      label_mask = tf.one_hot(model.y_input,
                              2,
                              on_value=1.0,
                              off_value=0.0,
                              dtype=tf.float32)
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]

  def perturb(self, x_nat, reference, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
    else:
      x = np.copy(x_nat)

    for i in range(self.k):
      grad, softmax = sess.run([self.grad, self.model.pre_softmax], feed_dict={self.model.x_input: x,
                                            self.model.reference_input: reference,
                                            self.model.y_input: y,
                                            self.model.phase_train_placeholder:False})

      x += self.a * np.sign(grad)

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon) 
    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  from tensorflow.examples.tutorials.mnist import input_data

  from model import Model

  with open('config.json') as config_file:
    config = json.load(config_file)

  model = Model()
  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['k'],
                         config['a'],
                         config['random_start'],
                         config['loss_func'])

  inception_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')
  saver = tf.train.Saver(inception_vars, max_to_keep=3)
  
  with tf.Session() as sess:
    # Restore the checkpoint
    pretrained_model = 'models/facenet/20170512-110547/'
    if pretrained_model:
        print('Restoring pretrained model: %s' % pretrained_model)
        # facenet.load_model(pretrained_model)

        model_exp = os.path.expanduser(pretrained_model)
        print('Model directory: %s' % model_exp)
        _, ckpt_file = facenet.get_model_filenames(model_exp)

        # print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver.restore(sess, os.path.join(model_exp, ckpt_file))

    x_adv = [] # adv accumulator

    # faces1, faces2, labels, filepaths1, filepaths2 = set_loader.load_testset(200)
    faces1, faces2, labels, filepaths1, filepaths2, label_index = set_loader.load_testset(20)
    #faces1, faces2, labels, filepaths = set_loader.load_testset(200)
    
    batch_size = config['batch_size']
    print("^"*10)
    print(len(faces1))
    num_examples = len(faces1)
    batch_number = int(math.ceil(num_examples/batch_size))
    accuracy_1 = 0
    accuracy_2 = 0
    accuracy_3 = 0
    accuracy_4 = 0
    accuracy_5 = 0
    # list_file = open('/media/fan/胡浩棋/3/lfw/impersation_true.txt', 'w')
    for ibatch in range(batch_number):
        bstart = ibatch * batch_size
        bend = min(bstart + batch_size, num_examples)
        print('batch size: {}'.format(bend - bstart))

        faces1_batch = faces1[bstart:bend, :]
        faces2_batch = faces2[bstart:bend, :]
        labels_b = label_index[bstart:bend]
        labels_batch = labels[bstart:bend]
        # print(len(faces1_batch))
        # faces1, faces2, labels, filepaths,_,_ = set_loader.load_testset(200)
        x_batch_adv = attack.perturb(faces1_batch, faces2_batch, labels_b, sess)
        print('Storing examples')
        path = config['store_adv_path']
        filepaths1_batch = filepaths1[bstart:bend]
        filepaths2_batch=filepaths2[bstart:bend]
        save_img(x_batch_adv, filepaths1_batch, path)
        print('Examples stored in {}'.format(path))
        feed_dict = {model.x_input: faces1_batch,
                     model.reference_input: faces2_batch,
                     model.y_input: labels_b,
                     model.phase_train_placeholder: False}
        real_labels,d= sess.run([model.pre_softmax,model.d], feed_dict=feed_dict)
        accuracy1 = np.mean(
            (np.argmax(labels_batch, axis=-1)) == (np.argmax(real_labels, axis=-1))
        )


        
        feed_dict = {model.x_input: x_batch_adv,
                     model.reference_input: faces2_batch,
                     model.y_input: labels_b,
                     model.phase_train_placeholder: False}
        adversarial_labels = sess.run(
            model.pre_softmax, feed_dict=feed_dict)
        # print(adversarial_labels)
        same_faces_index = np.where((np.argmax(labels_batch, axis=-1) == 0))
        different_faces_index = np.where((np.argmax(labels_batch, axis=-1) == 1))
        accuracy2 = np.mean(
            (np.argmax(labels_batch[same_faces_index], axis=-1)) ==
            (np.argmax(real_labels[same_faces_index], axis=-1))
        )

        accuracy3 = np.mean(
            (np.argmax(labels_batch[different_faces_index], axis=-1)) == (
                np.argmax(real_labels[different_faces_index], axis=-1))
        )

        accuracy4 = np.mean(
            (np.argmax(labels_batch[same_faces_index], axis=-1)) ==
            (np.argmax(adversarial_labels[same_faces_index], axis=-1))
        )

        accuracy5 = np.mean(
            (np.argmax(labels_batch[different_faces_index], axis=-1)) == (
                np.argmax(adversarial_labels[different_faces_index], axis=-1))
        )
        accuracy_1 += accuracy1
        accuracy_2 += accuracy2
        accuracy_3 += accuracy3
        accuracy_4 += accuracy4
        accuracy_5 += accuracy5
    accuracy1_a = accuracy_1 / batch_number
    accuracy2_a = accuracy_2 / batch_number
    accuracy3_a = accuracy_3 / batch_number
    accuracy4_a = accuracy_4 / batch_number
    accuracy5_a = accuracy_5 / batch_number
    print(batch_number)
    print('Accuracy: ' + str(accuracy1_a * 100) + '%')
    
    print('Accuracy legitimate examples for '
            + 'same person faces (dodging): '
            + str(accuracy2_a * 100)
            + '%')

    print('Accuracy legitimate examples for '
            + 'different people faces (impersonation): '
            + str(accuracy3_a * 100)
            + '%')

    print('Accuracy against adversarial examples for '
            + 'same person faces (dodging): '
            + str(accuracy4_a * 100)
            + '%')

    print('Accuracy against adversarial examples for '
            + 'different people faces (impersonation): '
            + str(accuracy5_a * 100)
            + '%')
