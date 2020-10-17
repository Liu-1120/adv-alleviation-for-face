#!/usr/bin/env python
#-*- coding: utf-8 -*-
#File:

import sys
import argparse
import tensorflow as tf
import tqdm
import numpy as np
import cv2
import os
import glob
import lfw
from sklearn import metrics
from scipy.optimize import brentq
from scipy import interpolate

# import lfw as lfw
# import align.detect_face as FaceDet
import facenet

class Model():

    def __init__(self):
        # from models import inception_resnet_v1  # facenet model     #             修改1
        import inception_resnet_v1
        self.network = inception_resnet_v1

        self.image_batch = tf.placeholder(tf.uint8, shape=[None, 160, 160, 3], name='images')

        image = (tf.cast(self.image_batch, tf.float32) - 127.5) / 128.0
        prelogits, _ ,_,_= self.network.inference(image, 1.0, False, bottleneck_layer_size=128)          #修改2，512 to 128
        self.embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
        pretrained_model = 'models\\facenet\\20170512-110547\\'
        self.sess = tf.Session()
        saver = tf.train.Saver()
        model_exp = os.path.expanduser(pretrained_model)
        print('Model directory: %s' % model_exp)
        _, ckpt_file = facenet.get_model_filenames(model_exp)

        # print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver.restore(self.sess, os.path.join(model_exp, ckpt_file))
        # saver.restore(self.sess, 'models/20180402-114759/model-20180402-114759.ckpt-275')

    def compute_victim(self, lfw_160_path, name):
        imgfolder = os.path.join(lfw_160_path, name)
        assert os.path.isdir(imgfolder), imgfolder
        images = glob.glob(os.path.join(imgfolder, '*.png')) + glob.glob(os.path.join(imgfolder, '*.jpg'))
        image_batch = [cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1] for f in images]
        for img in image_batch:
            assert img.shape[0] == 160 and img.shape[1] == 160, \
                "--data should only contain 160x160 images. Please read the README carefully."
        embeddings = self.eval_embeddings(image_batch)
        self.victim_embeddings = embeddings
        return embeddings

    def structure(self, input_tensor):
        """
        Args:
            input_tensor: NHWC
        """
        rnd = tf.random_uniform((), 135, 160, dtype=tf.int32)
        rescaled = tf.image.resize_images(
            input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_rem = 160 - rnd
        w_rem = 160 - rnd
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [
                        pad_left, pad_right], [0, 0]])
        padded.set_shape((input_tensor.shape[0], 160, 160, 3))
        output = tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(0.9),
                         lambda: padded, lambda: input_tensor)
        return output

    def build_pgd_attack(self, eps):
        victim_embeddings = tf.constant(self.victim_embeddings, dtype=tf.float32)

        def one_step_attack(image, grad):
            """
            core components of this attack are:
            (a) PGD adversarial attack (https://arxiv.org/pdf/1706.06083.pdf)
            (b) momentum (https://arxiv.org/pdf/1710.06081.pdf)
            (c) input diversity (https://arxiv.org/pdf/1803.06978.pdf)
            """
            orig_image = image
            image = self.structure(image)
            image = (image - 127.5) / 128.0
            image = image + tf.random_uniform(tf.shape(image), minval=-1e-2, maxval=1e-2)
            prelogits, _ ,t1,t2= self.network.inference(image, 1.0, False, bottleneck_layer_size=128)         #512 to 128
            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')
            t1=tf.reshape(t1,[-1,3*3*1792])
            print(prelogits.shape)

            embeddings = tf.reshape(embeddings[0], [128, 1])      #512 to 128
            objective = tf.reduce_mean(tf.matmul(victim_embeddings, embeddings))  # to be maximized
            noise, = tf.gradients(objective, orig_image)

            noise = noise / tf.reduce_mean(tf.abs(noise), [1, 2, 3], keep_dims=True)
            noise = 0.1 * grad + noise

            adv = tf.clip_by_value(orig_image + tf.sign(noise) * 1.0, lower_bound, upper_bound)
            return adv, noise

        input = tf.to_float(self.image_batch)
        lower_bound = tf.clip_by_value(input - eps, 0, 255.)
        upper_bound = tf.clip_by_value(input + eps, 0, 255.)

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            adv, _ = tf.while_loop(
                lambda _, __: True, one_step_attack,
                (input, tf.zeros_like(input)),
                back_prop=False,
                maximum_iterations=200,
                parallel_iterations=1)
        self.adv_image = adv
        return adv

    def eval_attack(self, img):
        # img: single HWC image
        out = self.sess.run(
            self.adv_image, feed_dict={self.image_batch: [img]})[0]
        return out

    def eval_embeddings(self, batch_arr):
        return self.sess.run(self.embeddings, feed_dict={self.image_batch: batch_arr})

    def distance_to_victim(self, img):
        emb = self.eval_embeddings([img])
        dist = np.dot(emb, self.victim_embeddings.T).flatten()
        stats = np.percentile(dist, [10, 30, 50, 70, 90])
        return stats


def validate_on_lfw(model, lfw_160_path):
    # Read the file containing the pairs used for testing
    pairs = lfw.read_pairs('validation-LFW-pairs.txt')
    # Get the paths for the corresponding images
    paths, actual_issame = lfw.get_paths(lfw_160_path, pairs)
    num_pairs = len(actual_issame)

    all_embeddings = np.zeros((num_pairs * 2, 128), dtype='float32')
    for k in tqdm.trange(num_pairs):
        img1 = cv2.imread(paths[k * 2], cv2.IMREAD_COLOR)[:, :, ::-1]
        img2 = cv2.imread(paths[k * 2 + 1], cv2.IMREAD_COLOR)[:, :, ::-1]
        batch = np.stack([img1, img2], axis=0)
        embeddings = model.eval_embeddings(batch)
        all_embeddings[k * 2: k * 2 + 2, :] = embeddings

    tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(
        all_embeddings, actual_issame, distance_metric=1, subtract_mean=True)

    print('Accuracy: %2.5f+-%2.5f' % (np.mean(accuracy), np.std(accuracy)))
    print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))

    auc = metrics.auc(fpr, tpr)
    print('Area Under Curve (AUC): %1.3f' % auc)
    eer = brentq(lambda x: 1. - x - interpolate.interp1d(fpr, tpr)(x), 0., 1.)
    print('Equal Error Rate (EER): %1.3f' % eer)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--data', help='path to MTCNN-aligned LFW dataset',
            default='/LFW_all/lfw_align_mtcnnpy_160/')
    parser.add_argument('--eps', type=int, default=8, help='maximum pixel perturbation')
    # parser.add_argument('--output', help='output image', default='C:/Users/hhq13/Desktop/LOT/adv_output3/')  #C:/Users/hhq13/Desktop/LOT/adv_output3/'
    parser.add_argument('--target', default='Courtney_Love')
    args = parser.parse_args()

    model = Model()
    # facev5path=''
    # lfw_path='C:/Users/hhq13/Desktop/facenet-test/LFW_all/lfw_all'
    youtubepath=''
    # victim = model.compute_victim(args.data, args.target)
    victim = model.compute_victim(youtubepath, args.target)
    print("Number of victim samples (the more the better): {}".format(len(victim)))
    model.build_pgd_attack(args.eps)
    n=1

    pairs=lfw.read_pairs('youtube.txt')
    print(len(pairs))
    for pair in pairs:
        if not os.path.exists("youtube_adv2/" + pair[0]):
            os.mkdir("youtube_adv2/" + pair[0])
    for pair in pairs:
        path = os.path.join(youtubepath, pair[0],pair[1])
        print(path)
        img = cv2.imread(path)[:, :, ::-1]
        out = model.eval_attack(img)
        # cv2.imwrite(args.output +pair[0], out[:, :, ::-1])
        cv2.imwrite("youtube_adv/"+pair[0]+"/"+pair[1], out[:, :, ::-1])
        print('n= ' + str(n))
        n=n+1



