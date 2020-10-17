#!usr/bin/python
# -*- coding: utf-8 -*-
import facenet
import tensorflow as tf
import numpy as np
import os
from scipy.misc import imsave
from cleverhans.model import Model
from cleverhans.attacks import FastGradientMethod
# import gc
import set_loader
import math
batch_size1 = 36
batch_shape = [36, 160, 160, 3]

def save_img(images, filepaths, output_dir):

    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        print(filename)
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (images[i, :, :, :] * 255.0).astype(np.uint8)
            imsave(f, img, format='PNG')


class InceptionResnetV1Model(Model):
    model_path = "/home/fan/facenet_adversarial_faces/models/facenet/20170512-110547/20170512-110547.pb"

    def __init__(self):
        super(InceptionResnetV1Model, self).__init__()

        # Load Facenet CNN
        facenet.load_model(self.model_path)
        # Save input and output tensors references

        graph = tf.get_default_graph()

        self.face_input = graph.get_tensor_by_name("input:0")
        self.embedding_output = graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        self.batch_size = graph.get_tensor_by_name("batch_size:0")
        # self.face_input = tf.placeholder(tf.int32, shape=[None, 160, 160, 3])
        # self.embedding_output = tf.placeholder(tf.float32, shape=(None, 128))

    def convert_to_classifier(self):
        # Create victim_embedding placeholder
        self.victim_embedding_input = tf.placeholder(
            tf.float32,
            shape=(None, 128))

        # Squared Euclidean Distance between embeddings
        distance = tf.reduce_sum(
            tf.square(self.embedding_output - self.victim_embedding_input),
            axis=1)

        # Convert distance to a softmax vector
        # 0.99 out of 4 is the distance threshold for the Facenet CNN
        threshold = 0.48
        score = tf.where(
            distance > threshold,
            0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.softmax_output)
        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))

output_dir = ''

with tf.Graph().as_default():
    with tf.Session() as sess:        
        # embedding_output = tf.placeholder(tf.float32, shape=(None, 128))
        # Load model
        model = InceptionResnetV1Model()
        # Convert to classifier
        model.convert_to_classifier()
        # graph = tf.get_default_graph()

        # face1 和 face2 可能是同人也可能不是同人，根据lables的值，True/[1,0]：同人；False/[0,1]表示非同人
        #目标攻击时加载数据
        faces1, faces2, labels, filepaths1, filepaths2, label_index = set_loader.load_testset(200)
        num_examples = len(faces1)
        batch_number = int(math.ceil(num_examples / batch_size1))
        # thresholds = np.arange(0, 4, 0.01)
        # threshold = 1.1
        # x_adv = []
        accuracy_1 = 0
        accuracy_2 = 0
        accuracy_3 = 0
        accuracy_4 = 0
        accuracy_5 = 0
        steps = 1
        eps = 0.02
        alpha = eps/steps
        fgsm = FastGradientMethod(model)
        # generate the adversarial examples
        for ibatch in range(batch_number):
            bstart = ibatch * batch_size1
            bend = min(bstart + batch_size1, num_examples)
            print('batch size: {}'.format(bend - bstart))
            index = int(batch_size1 / 2)

            faces1_batch = faces1[bstart:bend, :]
            faces2_batch = faces2[bstart:bend, :]
            labels_batch = labels[bstart:bend]
            filepaths1_batch = filepaths1[bstart:bend]
            #非目标攻击时加载数据
            #faces1, faces2, labels, filepaths = set_loader.load_testset(200)
            #print(label_index)
            # Create victims' embeddings using Facenet itself

            feed_dict = {model.face_input: faces2_batch,
                         model.phase_train_placeholder: False}
            # faces2 的 embeddings 值
            victims_embeddings = sess.run(
                model.embedding_output, feed_dict=feed_dict)

            # Define FGSM for the model
           
            fgsm_params = {'eps': alpha,
                           'clip_min': 0.,
                           'clip_max': 1.}
            adv_x = fgsm.generate(model.face_input, **fgsm_params)

            adv = faces1_batch
            for i in range(steps):
                print("FGSM step " + str(i + 1))
                feed_dict = {model.face_input: adv,
                             model.victim_embedding_input: victims_embeddings,
                             model.phase_train_placeholder: False}
                adv = sess.run(adv_x, feed_dict=feed_dict)



            feed_dict = {model.face_input: faces1_batch,
                         model.victim_embedding_input: victims_embeddings,
                         model.phase_train_placeholder: False,
                         model.batch_size: 36}
            save_img(adv, filepaths1_batch, output_dir)

            real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
            accuracy1 = np.mean(
                (np.argmax(labels_batch, axis=-1)) == (np.argmax(real_labels, axis=-1))
            )

            feed_dict = {model.face_input: adv,
                         model.victim_embedding_input: victims_embeddings,
                         model.phase_train_placeholder: False,
                         model.batch_size: 36}
            adversarial_labels = sess.run(
                model.softmax_output, feed_dict=feed_dict)
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
            del adv
            # gc.collect()
        accuracy1_a = accuracy_1 / batch_number
        accuracy2_a = accuracy_2 / batch_number
        accuracy3_a = accuracy_3 / batch_number
        accuracy4_a = accuracy_4 / batch_number
        accuracy5_a = accuracy_5 / batch_number
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
