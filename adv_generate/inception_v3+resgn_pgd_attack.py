"""
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from scipy.misc import imsave
import tensorflow as tf
import numpy as np
import os
import cv2
from scipy.misc import imread
import tensorlayer as tl
from tensorflow.contrib.slim.nets import inception
slim = tf.contrib.slim
# slim = tf.contrib.slim
def save_img(images, filepaths, output_dir):
    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (images[i, :, :, :])
            cv2.imwrite(os.path.join(output_dir, filename), img)
def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.
  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]
  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
      with tf.gfile.Open(filepath) as f:
          print(filepath)
          # image = imread(f, mode='RGB').astype(np.float) / 255.0
          image = cv2.imread(filepath)
      # Images for inception classifier are normalized to be in [-1, 1] interval.
      # images[idx, :, :, :] = image * 2.0 - 1.0
      images[idx, :, :, :] = image
      filenames.append(os.path.basename(filepath))
      idx += 1
      if idx == batch_size:
          yield filenames, images
          filenames = []
          images = np.zeros(batch_shape)
          idx = 0
  if idx > 0:
      yield filenames, images

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
            wrong_logit = tf.reduce_max((1 - label_mask) * model.pre_softmax, axis=1)
            loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
        else:
            print('Unknown loss function. Defaulting to cross-entropy')
            loss = model.xent

        self.grad = tf.gradients(loss, model.x_input)[0]

    def perturb(self, x_nat, y, sess):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if self.rand:
            x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)

        for i in range(self.k):
            grad, softmax = sess.run([self.grad, self.model.pre_softmax], feed_dict={self.model.x_input: x,
                                                                                     self.model.y_input: y})

            x += self.a * np.sign(grad)

            x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
        return x


if __name__ == '__main__':
    import json
    import skimage
    import math

    from tensorflow.examples.tutorials.mnist import input_data

    from model_resgn_inception_v3 import Model

    with open('config.json') as config_file:
        config = json.load(config_file)
    batch_shape = [20, 299, 299, 3]
    # input_dir = 'E:\\adv_dataset\\adv_generate\\inception_test\\resgn_clean\\'
    input_dir = ''
    r_input = tf.placeholder('float32', [20, 299, 299, 3], name='reference_input_input_to_facenet')
    # input_v3 = r_input * 2 - 1
    with slim.arg_scope(inception.inception_v3_arg_scope()):
        _, end_points_label = inception.inception_v3(
            r_input, num_classes=1001, is_training=False, reuse=tf.AUTO_REUSE)
    predicted_label = tf.argmax(end_points_label['Predictions'], 1)
    # model = Model()
    # attack = LinfPGDAttack(model,
    #                        config['epsilon'],
    #                        config['k'],
    #                        config['a'],
    #                        config['random_start'],
    #                        config['loss_func'])

    # inception_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV3')
    # saver = tf.train.Saver(inception_vars)
    # saver = tf.train.Saver(slim.get_model_variables(scope='InceptionV3'))
    # session_creator = tf.train.ChiefSessionCreator(
    #     scaffold=tf.train.Scaffold(saver=saver),
    #     checkpoint_filename_with_path='D:\\Comdefend-master\\inception_v3.ckpt')
    # with tf.Session() as sess1:
    #     tl.files.load_and_assign_npz(sess=sess1,
    #                              name='E:\\models_mse_facenet_1.14\\facenet_pgd_triplet_mse_12.26\\1230\\g_srgan_softmax150.npz',
    #                              network=model.net_g)

    # with tf.train.MonitoredSession(session_creator=session_creator) as sess:
    with tf.Session() as sess:
        #     # Restore the checkpoint
        #     pretrained_model = 'D:\\Comdefend-master\\adv_v3_model\\'
        #     if pretrained_model:
        #         print('Restoring pretrained model: %s' % pretrained_model)
        #         # facenet.load_model(pretrained_model)
        #
        #         # model_exp = os.path.expanduser(pretrained_model)
        #         # print('Model directory: %s' % model_exp)
        #         # _, ckpt_file = facenet.get_model_filenames(model_exp)
        #         #
        #         # # print('Metagraph file: %s' % meta_file)
        #         # print('Checkpoint file: %s' % ckpt_file)
        #         saver.restore(sess, 'D:\\Comdefend-master\\adv_v3_model\\adv_inception_v3.ckpt')
        #     tl.files.load_and_assign_npz(sess=sess,
        #                                  name='E:\\models_mse_facenet_1.14\\facenet_pgd_triplet_mse_12.26\\1230\\g_srgan_softmax150.npz',
        #                                  network=model.net_g)
            x_adv = []  # adv accumulator
            accuracy_1 = 0
            for filenames, images in load_images(input_dir, batch_shape):
                # labels = sess.run(predicted_label, feed_dict={r_input: images})
                # x_batch_adv = attack.perturb(images, labels, sess)
                # x_batch_adv = skimage.util.random_noise(images, mode='gaussian', seed=None, clip=True)
                x_batch_adv = images + np.random.uniform(0, 25, (299, 299, 3))
                # print('max(x_batch_adv)')
                # print(np.max(x_batch_adv[0, :, :, :]))
                # print(np.min(x_batch_adv[0, :, :, :]))
                print(x_batch_adv)
                print('Storing examples')
                path = config['store_adv_path']
                # save_img((x_batch_adv + 1) / 2, filenames, path)
                save_img(x_batch_adv, filenames, path)
            #     label_y, correct_num = sess.run([model.correct_prediction, model.num_correct], feed_dict={model.x_input: x_batch_adv,
            #                                                       model.y_input: labels})
            #     # correct_prediction = np.equal(labels, np.argmax(labels_adv))
            #
            #     # num_correct = np.sum(correct_prediction)
            #     # accuracy = np.mean(correct_prediction)
            #     accuracy_1 = accuracy_1 + correct_num
            #     print(correct_num)
            #     print(label_y)
            # print(accuracy_1)



