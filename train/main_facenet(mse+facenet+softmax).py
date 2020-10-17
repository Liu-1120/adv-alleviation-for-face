#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g
from utils import *
from vgg_face_model import *
from config import config, log_config
from facenet_api import *
import facenet
# 参数设置
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
# initialize G
n_epoch_init = config.TRAIN.n_epoch_init
# adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every
ni = int(np.sqrt(batch_size))
margin = config.margin
weight_decay = 0.0
keep_probability = 1.0
embedding_size = 128
RANDOM_ROTATE = 1
RANDOM_CROP = 2
RANDOM_FLIP = 4
FIXED_STANDARDIZATION = 8
FLIP = 16
def train():
    # 创建一个文件夹保存训练好的模型
    save_dir_ginit = "samples/facenet_pgd_loss_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "samples/facenet_pgd_loss_gan".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    checkpoint_dir = "checkpoint/facenet_pgd_loss_12.19"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    # 加载训练集数据
    train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    # 加载facenet参考样本数据集
    # train_reference_img_list = sorted(
    #     tl.files.load_file_list(path=config.TRAIN.reference_img_path, regx='.*.png', printable=False))

    train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    train_lr_imgs = tl.vis.read_images(train_lr_img_list, path=config.TRAIN.lr_img_path, n_threads=32)
    # train_reference_imgs = tl.vis.read_images(train_reference_img_list, path=config.TRAIN.reference_img_path,
    #                                           n_threads=32)
    t_image = tf.placeholder('float32', [batch_size, 160, 160, 3], name='t_image_input_to_SRGAN_generator')
    t_target_image = tf.placeholder('float32', [batch_size, 160, 160, 3], name='t_target_image')
    phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')
    # softmax_output1 = tf.placeholder('float32')
    # softmax_output2 = tf.placeholder('float32')
    # 定义模型
    net_g = SRGAN_g(t_image, is_train=True, reuse=False)

    net_g.print_params(False)
    net_g.print_layers()
    # 加载vggface模型
    data1 = loadmat('vgg-face.mat')
    # # # resize成vggface可以接受的图像尺寸
    # t_target_image_224 = tf.image.resize_images(t_target_image, size=[224, 224], method=0, align_corners=False)
    #
    # t_predict_image_224 = tf.image.resize_images(net_g.outputs, size=[224, 224], method=0, align_corners=False)
    # out_160 = tf.image.resize_images(net_g.outputs, size=[160, 160], method=0, align_corners=False)
    # t_target_image_160 = tf.image.resize_images(t_target_image, size=[160, 160], method=0, align_corners=False)
    out_160 = prewhitenfacenet(net_g.outputs)
    t_target_image_160 = prewhitenfacenet(t_target_image)
    image_batch1 = tf.identity(out_160, 'input')
    image_batch2 = tf.identity(t_target_image_160, 'input')
    # facenet_target_emb2 = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # #facenet_reference_emb2 = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    # facenet_predict_emb2 = tf.get_default_graph().get_tensor_by_name("embeddings:0")

    # net_vgg, vgg_target_emb, vgg_relu_emb = vgg_face_api(data1, (t_target_image_224 + 1) / 2)
    # _, vgg_predict_emb, vgg_predict_relu_emb = vgg_face_api(data1, (t_predict_image_224 + 1) / 2)
    # predicted_out = tf.nn.l2_normalize(vgg_predict_relu_emb, 1, 1e-10, name='embeddings')
    # print(predicted_out)
    # predicted_target_out = tf.nn.l2_normalize(vgg_relu_emb, 1, 1e-10, name='embeddings')
    # net_vgg, vgg_target_emb, vgg_target_emb2 = Vgg19_simple_api((t_target_image_224 + 1) / 2, reuse=False)
    # _, vgg_predict_emb, vgg_predict_emb2 = Vgg19_simple_api((t_predict_image_224 + 1) / 2, reuse=True)
    model_def = 'inception_resnet_v1_new'
    network = importlib.import_module(model_def)
    # print('Building training graph')

    # Build the inference graph
    prelogits1, _, texture_emb1 = network.inference(image_batch1, keep_probability,
                                      phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size,
                                      weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
    prelogits2, _, texture_emb2 = network.inference(image_batch2, keep_probability,
                                      phase_train=phase_train_placeholder, bottleneck_layer_size=embedding_size,
                                      weight_decay=weight_decay, reuse=tf.AUTO_REUSE)
    # logits = slim.fully_connected(prelogits, len(train_set), activation_fn=None,
    #         weights_initializer=slim.initializers.xavier_initializer(),
    #         weights_regularizer=slim.l2_regularizer(args.weight_decay),
    #         scope='Logits', reuse=False)

    embeddings1 = tf.nn.l2_normalize(prelogits1, 1, 1e-10, name='embeddings')
    embeddings2 = tf.nn.l2_normalize(prelogits2, 1, 1e-10, name='embeddings')
    # test inference
    net_g_test = SRGAN_g(t_image, is_train=False, reuse=True)
    # distance1 = tf.reduce_sum(tf.square(facenet_target_emb2 - facenet_predict_emb2))
    # softmax_output1 = softmax(distance1)
    # distance2 = tf.reduce_sum(tf.square(facenet_target_emb2 - facenet_target_emb2))
    # softmax_output2 = softmax(distance2)
    #softmax2 = convert_to_softmax(facenet_reference_emb2, facenet_reference_emb2)
    #print(distance)
    # ###========================== DEFINE TRAIN OPS ==========================###
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    # mse_loss = tl.cost.mean_squared_error(predicted_out, predicted_target_out, is_mean=True)
    vgg_loss = 1e4 * tl.cost.mean_squared_error(texture_emb1, texture_emb2, is_mean=True)
    # distance1 = 1/batch_size * tf.reduce_sum(tf.square(embeddings1 - embeddings2))
    distance1 = tf.reduce_sum(tf.square(embeddings1 - embeddings2), axis=1)
    softmax_output_value1 = tf.transpose(softmax(distance1))
    # distance2 = 1/batch_size * tf.reduce_sum(tf.square(embeddings2 - embeddings2))
    distance2 = tf.reduce_sum(tf.square(embeddings2 - embeddings2), axis=1)
    softmax_output_value2 = tf.transpose(softmax(distance2))
    # softmax_loss = 1e-3 * tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=softmax_output_value2,
    #                                                                                        logits=softmax_output_value1))
    index = tf.arg_max(softmax_output_value2, 1)
    label_mask = tf.one_hot(index,
                            2,
                            on_value=1.0,
                            off_value=0.0,
                            dtype=tf.float32)
    softmax_loss = 1e2 * tf.reduce_mean(-tf.reduce_sum(label_mask * tf.log(softmax_output_value1), 1))
    # softmax_loss = 1e2 * tl.cost.mean_squared_error(embeddings1, embeddings2, is_mean=True)
    # 生成器损失
    g_loss = mse_loss + vgg_loss + softmax_loss

    g_vars = tl.layers.get_variables_with_name('SRGAN_g', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)
    inception_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionResnetV1')
    saver = tf.train.Saver(inception_vars, max_to_keep=3)
    # 前100轮的初始化只优化Mse损失
    g_optim_init = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(mse_loss, var_list=g_vars)
    # SRGAN
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    # 模型恢复
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    pretrained_model = '/home/fan/facenet_adversarial_faces/models/facenet/20170512-110547/'
    if pretrained_model:
        print('Restoring pretrained model: %s' % pretrained_model)
        # facenet.load_model(pretrained_model)

        model_exp = os.path.expanduser(pretrained_model)
        print('Model directory: %s' % model_exp)
        _, ckpt_file = facenet.get_model_filenames(model_exp)

        # print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)
        saver.restore(sess, os.path.join(model_exp, ckpt_file))

    if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']),
                                    network=net_g) is False:
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']),
                                     network=net_g)

    for var in tf.trainable_variables():
        print(var.name)
    # 开始训练
    sample_imgs_h = train_hr_imgs[0:batch_size]
    sample_imgs_l = train_lr_imgs[0:batch_size]
    sample_imgs_h = tl.prepro.threading_data(sample_imgs_h, fn=retain, is_random=False)
    print('sample HR sub-image:', sample_imgs_h.shape, sample_imgs_h.min(), sample_imgs_h.max())
    sample_imgs_l = tl.prepro.threading_data(sample_imgs_l, fn=retain, is_random=False)
    print('sample LR sub-image:', sample_imgs_l.shape, sample_imgs_l.min(), sample_imgs_l.max())
    tl.vis.save_images(sample_imgs_l, [ni, ni], save_dir_ginit + '/_train_sample_l.png')
    tl.vis.save_images(sample_imgs_h, [ni, ni], save_dir_ginit + '/_train_sample_h.png')
    tl.vis.save_images(sample_imgs_l, [ni, ni], save_dir_gan + '/_train_sample_l.png')
    tl.vis.save_images(sample_imgs_h, [ni, ni], save_dir_gan + '/_train_sample_h.png')

    # 初始化生成器
    # fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    print(" ** fixed learning rate: %f (for init G)" % lr_init)
    for epoch in range(0, n_epoch_init + 1):
        epoch_time = time.time()
        total_mse_loss, n_iter = 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            x_imgs = [1] * batch_size
            for i in range(0, batch_size):
                x_imgs[i] = np.concatenate([train_hr_imgs[idx + i], train_lr_imgs[idx + i]], axis=2)
            b_imgs = tl.prepro.threading_data(x_imgs, fn=retain, is_random=True)
            b_imgs_h = b_imgs[:, :, :, 0:3]
            b_imgs_l = b_imgs[:, :, :, 3:6]

            # update G
            errM, _ = sess.run([mse_loss, g_optim_init], {t_image: b_imgs_l, t_target_image: b_imgs_h})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, mse: %.8f " % (
            epoch, n_epoch_init, n_iter, time.time() - step_time, errM))
            total_mse_loss += errM
            n_iter += 1
        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, mse: %.8f\n" % (
        epoch, n_epoch_init, time.time() - epoch_time, total_mse_loss / n_iter)
        print(log)
        f = open('log_init.txt', 'a')
        f.write(log)
        f.close()

        # quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_l})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_ginit + '/train_%d.png' % epoch)

        # save model
        if (epoch != 0) and (epoch % 10 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_{}_init_softmax.npz'.format(tl.global_flag['mode']),
                              sess=sess)

    # 开始训练GAN网络
    for epoch in range(0, n_epoch + 1):
        # update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, total_mse_loss, total_vgg_loss, total_adv_loss, total_vgg_loss2, n_iter = 0, 0, 0, 0, 0, 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            x_imgs = [1] * batch_size
            for i in range(0, batch_size):
                x_imgs[i] = np.concatenate([train_hr_imgs[idx + i], train_lr_imgs[idx + i]], axis=2)
            b_imgs = tl.prepro.threading_data(x_imgs, fn=retain, is_random=True)
            b_imgs_h = b_imgs[:, :, :, 0:3]
            b_imgs_l = b_imgs[:, :, :, 3:6]

            # update G
            errG, errM, errV, errV2, _ = sess.run([g_loss, mse_loss, vgg_loss, softmax_loss,
                                                   g_optim], {t_image: b_imgs_l, t_target_image: b_imgs_h, phase_train_placeholder: False})
            print("Epoch [%2d/%2d] %4d time: %4.4fs, g_loss: %.8f (mse: %.6f vgg: %.6f facenet: %.6f)" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errG, errM, errV, errV2))
            total_g_loss += errG
            total_mse_loss += errM
            total_vgg_loss += errV
            total_vgg_loss2 += errV2
            n_iter += 1


        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f (mse: %.6f vgg: %.6f facenet: %.6f)\n" \
              % (epoch, n_epoch, time.time() - epoch_time, total_g_loss / n_iter,
                 total_mse_loss / n_iter, total_vgg_loss / n_iter, total_vgg_loss2 / n_iter)
        print(log)
        f = open('log.txt', 'a')
        f.write(log)
        f.close()

        # quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_l})
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        # save model
        if (epoch != 0) and (epoch % 50 == 0):
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/g_srgan_softmax%d.npz' % epoch, sess=sess)

    sess.close()
def evaluate():
    # create folders to save result images
    checkpoint_dir = "/home/fan/su/remove_face/checkpoint/facenet_pgd_joint_loss/"
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # 定义模型
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)

    for epoch_n in range(200, 250, 150):
        save_dir = "samples/evaluate/" + str(epoch_n)
        tl.files.exists_or_mkdir(save_dir)
        tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan_softmax%d.npz' % epoch_n, network=net_g)
        # 设置输入样本级数量
        for imid in range(1135):
            valid_lr_img = valid_lr_imgs[imid]
            print(valid_lr_img)
            valid_lr_img = (valid_lr_img / 127.5) - 1  # 归一化到［－1, 1]
            # 开始评估 123
            start_time = time.time()
            out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
            print("took: %4.4fs" % (time.time() - start_time))
            print("[*] save images")
            print(out)
            tl.vis.save_image(out[0], save_dir + '/%s' % (valid_lr_img_list[imid]))

def softmax(distance):
    threshold = 1.1
    # threshold = 0.99
    score = tf.where(
            distance > threshold,
            0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * distance / threshold)
    reverse_score = 1.0 - score
    softmax_output = tf.stack([reverse_score, score])
    return softmax_output
def prewhitenfacenet(imgs):
    image_size = (160, 160)
    # image, control = imgs.dequeue()
    images = []
    for img1 in tf.unstack(imgs):
        img2 = tf.image.per_image_standardization(img1)
        images.append(img2)
    return images


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='evaluate', help='srgan, evaluate')
    args = parser.parse_args()
    tl.global_flag['mode'] = args.mode
    if tl.global_flag['mode'] == 'srgan':
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
