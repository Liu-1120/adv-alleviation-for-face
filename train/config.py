#! /usr/bin/python
# -*- coding: utf8 -*-

from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

# Adam
config.TRAIN.batch_size = 16
config.TRAIN.lr_init = 1e-3
config.TRAIN.beta1 = 0.9

# initialize G 生成器初始化迭代次数是 n_epoch_init 次
config.TRAIN.n_epoch_init = 10
# config.TRAIN.lr_decay_init = 0.1
# config.TRAIN.decay_every_init = int(config.TRAIN.n_epoch_init / 2)

# adversarial learning
config.TRAIN.n_epoch = 1000
config.TRAIN.lr_decay = 0.1
config.TRAIN.decay_every = int(config.TRAIN.n_epoch / 2)

#非目标攻击
# 训练集路径
#干净样本文件夹
# config.TRAIN.hr_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/train/train_2/'
config.TRAIN.hr_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/train/all_train_clean/'
# 对抗样本文件夹
# config.TRAIN.lr_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/train/train_1/'
config.TRAIN.lr_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/train/all_train_adv_10.22/'
config.TRAIN.tongren_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/train/clean_reference_tongren/'
# 对抗样本文件夹
# config.TRAIN.lr_img_path = '/home/fan/su/generate_adv/fgsm_32/'
config.TRAIN.feitongren_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/train/clean_reference_feitongren/'
#facenet框架下需要参考样本
#参考样本路径
# config.TRAIN.reference_img_path = '/home/fan/facenet_adversarial_faces/train/all_train_reference_1/'
config.margin = 1.1
config.VALID = edict()
# 测试集路径
#dogging
# config.VALID.lr_img_path = '/home/fan/facenet_adversarial_faces/test/dogging_test_adv_clean_or/'
#impertation
# config.VALID.lr_img_path = '/home/fan/facenet_adversarial_faces/test/impersation_test_adv_clean_or/'
#legitimate
config.VALID.lr_img_path = '/home/fan/facenet_adversarial_faces/adv_generate/test/clean/test_dodging/'


def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
