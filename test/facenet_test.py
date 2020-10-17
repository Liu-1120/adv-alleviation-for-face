# -*- coding: utf-8 -*-
# @Time    : 19-12-29 上午11:02
# @Author  : 范伟琦
# @Software: PyCharm
# @FileName: facenet_test_2.py
#!usr/bin/python
# -*- coding: utf-8 -*-
import facenet
import tensorflow as tf
import numpy as np
import os, math
import tensorflow.contrib.slim as slim
# from networks import sphere_network as network
from scipy.misc import imsave
from cleverhans.model import Model
# from cleverhans.attacks import FastGradientMethod
image_size = 160
import set_loader
import importlib
from scipy import misc
weight_decay = 0.0
keep_probability = 1.0
embedding_size = 128
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
batch_size = 10

def save_img(images, filepaths, output_dir):

    for i, filepath in enumerate(filepaths):
        filename = os.path.basename(filepath)
        with tf.gfile.Open(os.path.join(output_dir, filename), 'w') as f:
            img = (images[i, :, :, :] * 255.0).astype(np.uint8)
            imsave(f, img, format='PNG')


class InceptionResnetV1Model(Model):
    # model_path = "/home/fan/facenet_adversarial_faces/models/facenet/20170512-110547/20170512-110547.pb"

    def __init__(self):
        super(InceptionResnetV1Model, self).__init__()
        self.t_image = tf.placeholder('float32', [batch_size, 160, 160, 3], name='t_image_input_to_SRGAN_generator')
        self.t_target_image = tf.placeholder('float32', [batch_size, 160, 160, 3], name='t_target_image')
        image_batch1 = tf.identity(self.t_image, 'input')
        image_batch2 = tf.identity(self.t_target_image, 'input')
        # Load Facenet CNN
        # facenet.load_model(self.model_path)
        # model_def = 'inception_resnet_v1_new'
        # network = importlib.import_module(model_def)
        # Save input and output tensors references
        model_def = 'inception_resnet_v1'
        network = importlib.import_module(model_def)
        # print('Building training graph')

        # Build the inference graph
        prelogits1, _ = network.inference(image_batch1, keep_probability,
                                          phase_train=False,
                                          bottleneck_layer_size=embedding_size,
                                          weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
        prelogits2, _ = network.inference(image_batch2, keep_probability,
                                          phase_train=False,
                                          bottleneck_layer_size=embedding_size,
                                          weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
        # graph = tf.get_default_graph()
        self.embeddings1 = tf.nn.l2_normalize(prelogits1, 1, 1e-10, name='embeddings')
        self.embeddings2 = tf.nn.l2_normalize(prelogits2, 1, 1e-10, name='embeddings')
        # self.face_input = graph.get_tensor_by_name("input:0")
        # self.embedding_output = graph.get_tensor_by_name("embeddings:0")
        # self.face_input = tf.placeholder(tf.int32, shape=[None, 160, 160, 3])
        # self.embedding_output = tf.placeholder(tf.float32, shape=(None, 128))

    def convert_to_classifier(self):
        # Create victim_embedding placeholder
        # self.victim_embedding_input = tf.placeholder(
        #     tf.float32,
        #     shape=(None, 128))

        # Squared Euclidean Distance between embeddings
        self.distance = tf.reduce_sum(
            tf.square(self.embeddings1 - self.embeddings2),
            axis=1)

        # Convert distance to a softmax vector
        # 0.99 out of 4 is the distance threshold for the Facenet CNN
        # threshold = 0.99
        threshold = 1.1
        score = tf.where(
            self.distance > threshold,
            0.5 + ((self.distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * self.distance / threshold)
        reverse_score = 1.0 - score
        self.softmax_output = tf.transpose(tf.stack([reverse_score, score]))

        # Save softmax layer
        self.layer_names = []
        self.layers = []
        self.layers.append(self.softmax_output)
        self.layer_names.append('probs')

    def fprop(self, x, set_ref=False):
        return dict(zip(self.layer_names, self.layers))

#output_dir = '/home/fan/facenet_adversarial_faces/output1/'

with tf.Graph().as_default():
    with tf.Session() as sess:
        # embedding_output = tf.placeholder(tf.float32, shape=(None, 128))
        # Load model
        model = InceptionResnetV1Model()
        # Convert to classifier
        model.convert_to_classifier()
        # saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
        inception_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')
        # saver_inception_v3 = tf.train.Saver(slim.get_model_variables())
        saver = tf.train.Saver(inception_vars, max_to_keep=3)
        # face1 和 face2 可能是同人也可能不是同人，根据lables的值，True/[1,0]：同人；False/[0,1]表示非同人
        faces1, faces2, labels, filepaths1, filepaths2 = set_loader.load_construct_testset(1)
        # print(faces1)
        # print(faces2)
        pretrained_model = 'E:\\models\\facenet\\20170512-110547\\'
        if pretrained_model:
                print('Restoring pretrained model: %s' % pretrained_model)
                # facenet.load_model(pretrained_model)

                model_exp = os.path.expanduser(pretrained_model)
                print('Model directory: %s' % model_exp)
                _, ckpt_file = facenet.get_model_filenames(model_exp)

                # print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                saver.restore(sess, os.path.join(model_exp, ckpt_file))
        # Create victims' embeddings using Facenet itself
        # graph = tf.get_default_graph()
        # phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        # feed_dict = {model.face_input: faces2,
        #              phase_train_placeholder: False}
        # # faces2 的 embeddings 值
        # victims_embeddings = sess.run(
        #     model.embedding_output, feed_dict=feed_dict)

        # Define FGSM for the model
        #steps = 1
        #eps = 0.03
        #alpha = eps/steps
        #fgsm = FastGradientMethod(model)
       # fgsm_params = {'eps': alpha,
        #               'clip_min': 0.,
         #              'clip_max': 1.}
        #adv_x = fgsm.generate(model.face_input, **fgsm_params)
        # bim = BasicIterativeMethod(model)
        # # bim_params = {'eps_iter': 0.05,
        # #               'nb_iter': 10,
        # #               'clip_min': 0.,
        # #               'clip_max': 1.}
        # adv_x = bim.generate(model.face_input)
        #faces的embedding数值
        #adv = faces1
        # feed_dict = {model.face_input: faces1,
        #              phase_train_placeholder: False}
        # faces1_embedding = sess.run(model.embedding_output, feed_dict=feed_dict)
            # print(adv)
        # dis0 = np.sum(np.square(faces1_embedding - victims_embeddings), axis=1)
        # # 保存图片
        # print(faces1_embedding)
        #save_img(adv, filepaths, output_dir)
        # output_dir = '/home/fan/su/remove_face/test_0.03/impersation_1/'
        # output_dir1 = '/home/fan/facenet_adversarial_faces/datasets/lfw_feitongren_reference/'
        # output_file = '/home/fan/facenet_adversarial_faces/adv_generate/test/0result_txt/pgd_result_resgn_f.txt'
        # Test accuracy of the model
        # batch_size = graph.get_tensor_by_name("batch_size:0")
        num_examples = len(faces1)
        batch_number = int(math.ceil(num_examples / batch_size))
        accuracy_1 = 0
        accuracy_2 = 0
        accuracy_3 = 0
        accuracy_4 = 0
        accuracy_5 = 0
        for ibatch in range(batch_number):
            bstart = ibatch * batch_size
            bend = min(bstart + batch_size, num_examples)
            print('batch size: {}'.format(bend - bstart))

            faces1_batch = faces1[bstart:bend, :]
            faces2_batch = faces2[bstart:bend, :]
            labels_batch = labels[bstart:bend]
            feed_dict = {model.t_image: faces1_batch,
                         model.t_target_image: faces2_batch}
            em1, em2, real_labels, distance_real = sess.run([model.embeddings1, model.embeddings2, model.softmax_output, model.distance], feed_dict=feed_dict)
            # np.savetxt(output_file, real_labels)
            # print(em1)
            # print(em2)
            same_faces_index = np.where((np.argmax(labels_batch, axis=-1) == 0))
            print(distance_real)
            # print(dis0)
            # print(real_labels)
            # path = []
            # path1 = []
            # for i in range(len(labels)):
            #     if (np.argmax(labels[i], axis=-1)) != (np.argmax(real_labels[i], axis=-1)):
            #         path.append(filepaths1[i])
            #         path1.append(filepaths2[i])
            #         # filepaths =
            #         # print(filepaths)
            # # faces = misc.imread(path)
            # faces = facenet.load_data(path, False, False, image_size)
            # save_img(faces, path, output_dir)
            # faces1 = facenet.load_data(path1, False, False, image_size)
            # save_img(faces1, path1, output_dir1)
            accuracy = np.mean(
                (np.argmax(labels_batch[same_faces_index], axis=-1)) == (np.argmax(real_labels[same_faces_index], axis=-1))
            )
            print('Accuracy: ' + str(accuracy * 100) + '%')
            accuracy_1 += accuracy
        accuracy1_a = accuracy_1 / batch_number
        print('the total Accuracy: ' + str(accuracy1_a * 100) + '%')