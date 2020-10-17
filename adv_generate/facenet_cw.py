#!usr/bin/python
# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorlayer as tl
import numpy as np
import os
from scipy.misc import imsave
from cleverhans.model import Model
from cleverhans.attacks_cw import FastGradientMethod, CarliniWagnerL2
import facenet
import set_loader
import importlib
import math
batch_size = 16
batch_shape = [16, 160, 160, 3]
weight_decay = 0.0
keep_probability = 1.0
embedding_size = 128
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def save_img(images, filepaths, output_dir):

    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (images[i] * 255.0).astype(np.uint8)
            imsave(f, img, format='PNG')

def prewhitenfacenet(imgs):
    image_size = (160, 160)
    # image, control = imgs.dequeue()
    images = []
    for img1 in tf.unstack(imgs):
        img2 = tf.image.per_image_standardization(img1)
        images.append(img2)
    return images

class InceptionResnetV1Model(Model):
    model_path = "/facenet/20170512-110547/20170512-110547.pb"

    def __init__(self):
        super(InceptionResnetV1Model, self).__init__()




    def __call__(self, x1, x2):
        # Create victim_embedding placeholder
        # self.victim_embedding_input = victim_embedding_input
        # self.embedding_output = embedding_output
        # Squared Euclidean Distance between embeddings
        out_160 = prewhitenfacenet(x1)
        image_batch1 = tf.identity(out_160, 'input')
        out1_160 = prewhitenfacenet(x2)
        image_batch2 = tf.identity(out1_160, 'input')
        model_def = 'inception_resnet_v1'
        network = importlib.import_module(model_def)
        # print('Building training graph')

        # Build the inference graph
        prelogits1, _ = network.inference(image_batch1, keep_probability,
                                          phase_train=False, bottleneck_layer_size=embedding_size,
                                          weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
        embeddings1 = tf.nn.l2_normalize(prelogits1, 1, 1e-10, name='embeddings')
        prelogits2, _ = network.inference(image_batch2, keep_probability,
                                          phase_train=False, bottleneck_layer_size=embedding_size,
                                          weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
        embeddings2 = tf.nn.l2_normalize(prelogits2, 1, 1e-10, name='embeddings')
        distance = tf.reduce_sum(
            tf.square(embeddings1 - embeddings2),
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

        return self.softmax_output
    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))

output_dir = ''

with tf.Graph().as_default():
    with tf.Session() as sess:        
        # embedding_output = tf.placeholder(tf.float32, shape=(None, 128))
        # Load model
        model = InceptionResnetV1Model()
        # Convert to classifier
        # model.convert_to_classifier()

        # face1 和 face2 可能是同人也可能不是同人，根据lables的值，True/[1,0]：同人；False/[0,1]表示非同人
        #目标攻击时加载数据
        # faces1, faces2, labels, filepaths1, filepaths2 = set_loader.load_construct_testset_1(200)
        faces1, faces2, labels, filepaths1, filepaths2, label_index = set_loader.load_testset(16)
        #非目标攻击时加载数据
        #faces1, faces2, labels, filepaths = set_loader.load_testset(200)

        num_examples = len(faces1)
        batch_number = int(math.ceil(num_examples / batch_size))
        # thresholds = np.arange(0, 4, 0.01)
        # threshold = 1.1
        # x_adv = []
        accuracy_1 = 0
        accuracy_2 = 0
        accuracy_3 = 0
        accuracy_4 = 0
        accuracy_5 = 0
        # generate the adversarial examples
        for ibatch in range(batch_number):

            bstart = ibatch * batch_size
            bend = min(bstart + batch_size, num_examples)
            print('batch size: {}'.format(bend - bstart))
            index = int(batch_size / 2)

            faces1_batch = faces1[bstart:bend, :]
            faces2_batch = faces2[bstart:bend, :]
            labels_batch = labels[bstart:bend]
            filepaths1_batch = filepaths1[bstart:bend]
            # Create victims' embeddings using Facenet itself
            # graph = tf.get_default_graph()
            x_input1 = tf.placeholder(tf.float32, shape=batch_shape)
            x_input2 = tf.placeholder(tf.float32, shape=batch_shape)
            prediction = model(x_input2, x_input1)

            # prediction = sess.run(predictions, feed_dict={phase_train_placeholder: False})
            # Define FGSM for the model
            steps = 1
            cw_params = {'binary_search_steps': 10,
                         'abort_early': True,
                         'max_iterations': 5000,
                         'learning_rate': 0.01,
                         'batch_size': 16,
                         'initial_const': 0.1,
                         'nb_classes': 2,
                         'confidence': 0.9,
                         'clip_min': 0.0,
                         'clip_max': 1.0
                         # 'y_target': [[]],
                         # 'phase_train_placeholder': False,
                         # 'model.victim_embedding_input': victims_embeddings,
                         }
            inception_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')
            saver = tf.train.Saver(inception_vars, max_to_keep=3)
            # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
            pretrained_model = '/models/facenet/20170512-110547/'
            if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                # facenet.load_model(pretrained_model)

                model_exp = os.path.expanduser(pretrained_model)
                print('Model directory: %s' % model_exp)
                _, ckpt_file = facenet.get_model_filenames(model_exp)

                # print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                saver.restore(sess, os.path.join(model_exp, ckpt_file))

            # x_input = tf.placeholder(tf.float32, shape=batch_shape)
            # output = model.get_probs(x_input)
            # faces3 = faces1 + faces2
            # out = sess.run(output, feed_dict={x_input1: faces1, x_input2: faces2, x_input: faces1})
            # print(out)
            CW = CarliniWagnerL2(model, back='tf', sess=sess)
            # tfe = tf.contrib.eager
            # x = tfe.Variable(faces1)
            # adv_x = self.generate(x, **kwargs)
            # adv_x = CW.generate(x_input1, faces1, x_input2, faces2, **cw_params)
            adv_x = CW.generate(x_input1, faces1_batch, x_input2, faces2_batch, **cw_params)
            # print(faces1[15])
            # print(adv_x[15])
            feed_dict = {x_input1: faces1_batch,
                         x_input2: faces2_batch
                         }
            adv_x = sess.run(adv_x, feed_dict=feed_dict)
            # per, adv = sess.run([prediction, adv_x], feed_dict=feed_dict)
            # print(adv_x)
            # print(faces1)
            # adv = sess.run(adv_x, feed_dict=feed_dict)
            # 针对 faces1 进行攻击
            # adv = faces1
            # for i in range(steps):
            #     print("FGSM step " + str(i + 1))
            #     feed_dict = {model.face_input: adv,
            #                  model.victim_embedding_input: victims_embeddings,
            #                  phase_train_placeholder: False}
            #     adv = sess.run(adv_x, feed_dict=feed_dict)
            #     # print(adv)
            #
            # 保存图片

            save_img(adv_x, filepaths1_batch, output_dir)
            # #参考图片
            # save_img(faces2, filepaths2, target_dir)
            # #干净图片
            # save_img(faces1, filepaths1, clean_dir)
            # Test accuracy of the model
            real_labels = sess.run(model.softmax_output, feed_dict=feed_dict)
            accuracy1 = np.mean(
                (np.argmax(labels_batch, axis=-1)) == (np.argmax(real_labels, axis=-1))
            )
            adv_dict = {x_input1: adv_x,
                         x_input2: faces2_batch
                         }
            adversarial_labels = sess.run(
                model.softmax_output, feed_dict=adv_dict)
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