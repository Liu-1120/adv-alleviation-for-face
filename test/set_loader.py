#!usr/bin/python
# -*- coding: utf-8 -*-
import lfw
import facenet
import tensorflow as tf
import numpy as np
import os

#用facenet做测试
# pairs_path = '/home/fan/vgg_face_dataset/test_vggface.txt'
# pairs_path = "C:\\Users\\hhq13\\Desktop\\facenet-test\\facenet_txt\\adv_dogging_3.txt"
pairs_path="F:\\3\\lfw\\impersation_true.txt"
# pairs_path="F:\\2\\faceV5\\facev5_txt\\true_same.txt"
# pairs_path = "/home/fan/facenet_adversarial_faces/test_txt/adv_impersation_3.txt"
# testset_path = "D:\CosFace-master\cos_pgd"
testset_path = "C:\\Users\\hhq13\\Desktop\\facenet-test\\test"
# testset_path="F:\\2\\faceV5\\FaceV5-160"
# testset_path = '/home/fan/vgg_face_dataset'
# 用fecenet做攻击
# pairs_path = "/home/fan/facenet_adversarial_faces/datasets/lfw/name_pair_different_same_test_random.txt"
# pairs_path = "/home/fan/facenet_adversarial_faces/datasets/lfw/name_pair_same_all.txt"
# pairs_path = "/home/fan/facenet_adversarial_faces/datasets/lfw/name_pair_different9.txt"
# testset_path = "/home/fan/facenet_adversarial_faces/datasets/lfw_align_mtcnnpy_160"
file_extension = 'png'
image_size = 160
txtfile = "/home/fan/facenet_adversarial_faces/label.txt"

def load_testset(size):
    # Load images paths and labels
    pairs = lfw.read_pairs(pairs_path)
    # print('pairs length is ')
    # print(len(pairs))
    paths, labels, pathname_list = lfw.get_paths(testset_path, pairs)
    # print('labels length is 6000')
    # print(len(labels))
    # print('paths length is ')
    # print(len(paths))

    # Random choice // permutation 排列 size个数
    permutation = np.random.choice(len(labels), size, replace=False)
    paths_batch_1 = []
    paths_batch_2 = []
    # print('the permutation is ')
    # print(permutation)
    # print(len(permutation))

    # 遍历
    for index in range(len(labels)):
    # for index in permutation:
        # paths_batch_1对应的是path0   paths_batch_2对应的是path1，有同人和不同人区分。
        paths_batch_1.append(paths[index * 2])
        paths_batch_2.append(paths[index * 2 + 1])

    # 取出的size个样本所对应的标签（同人或者非同人）
    # labels = np.asarray(labels)[permutation]

    # print(labels)

    # paths_batch_1 和 paths_batch_2是带有完整路径的图片
    paths_batch_1 = np.asarray(paths_batch_1)
    paths_batch_2 = np.asarray(paths_batch_2)

    # print(len(paths_batch_1))
    # print(paths_batch_1)
    # print(len(paths_batch_2))

    # filenames是只有 Agnelo_Queiroz_0001.png
    # filenames = []
    # for filepath in sorted(tf.gfile.Glob(paths_batch_1)):
    #     filenames.append(os.path.basename(filepath))
    # print(filenames)

    # Load images  img = flip(img, do_random_flip) ????
    faces1 = facenet.load_data(paths_batch_1, False, False, image_size)
    faces2 = facenet.load_data(paths_batch_2, False, False, image_size)

    # 像素归一化到0到1
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

    # 将labels中True和False值转换为 [1, 0], [0, 1] 这样的值   True=【1,0】 False=【0,1】
    # f = open(txtfile, 'a')
    onehot_labels = []
    label_index = []
    for index in range(len(labels)):
        if labels[index]:
            onehot_labels.append([1, 0])
            label_index.append(0)
        else:
            onehot_labels.append([0, 1])
            label_index.append(1)
    label_index = np.transpose(label_index)
    # # f.write(labels + '\n')
    # print(onehot_labels, file=f)
    # f.close()

    return faces1, faces2, np.array(onehot_labels), paths_batch_1, paths_batch_2, label_index



def load_construct_testset(size):
    # Load images paths and labels
    pairs = lfw.read_pairs(pairs_path)
    print('pairs length is ')
    # print(pairs)
    paths, labels = lfw.get_test_paths(testset_path, pairs)
    # print('labels length is 6000')
    # print(len(labels))
    # print('paths length is ')
    # print(paths)

    # Random choice // permutation 排列 size个数
    # permutation = np.random.choice(len(labels), size, replace=False)
    paths_batch_1 = []
    paths_batch_2 = []
    # print('the permutation is ')
    # print(permutation)
    # print(len(permutation))

    # 遍历
    for index in range(len(labels)):
    # for index in permutation:
        # paths_batch_1对应的是path0   paths_batch_2对应的是path1，有同人和不同人区分。
        paths_batch_1.append(paths[index * 2])
        paths_batch_2.append(paths[index * 2 + 1])

    # 取出的size个样本所对应的标签（同人或者非同人）
    # labels = np.asarray(labels)[permutation]

    # print(labels)

    # paths_batch_1 和 paths_batch_2是带有完整路径的图片
    paths_batch_1 = np.asarray(paths_batch_1)
    paths_batch_2 = np.asarray(paths_batch_2)

    # print(len(paths_batch_1))
    # print(paths_batch_1)
    # print(len(paths_batch_2))

    # filenames是只有 Agnelo_Queiroz_0001.png
    # filenames = []
    # for filepath in sorted(tf.gfile.Glob(paths_batch_1)):
    #     filenames.append(os.path.basename(filepath))
    # print(filenames)

    # Load images  img = flip(img, do_random_flip) ????
    faces1 = facenet.load_data(paths_batch_1, False, False, image_size)
    faces2 = facenet.load_data(paths_batch_2, False, False, image_size)
    # print(faces1)
    # 像素归一化到0到1
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)
    print('min_pixel: max_pixel:')
    print(min_pixel)
    print(max_pixel)
    # 将labels中True和False值转换为 [1, 0], [0, 1] 这样的值   True=【1,0】 False=【0,1】
    onehot_labels = []
    for index in range(len(labels)):
        if labels[index]:
            onehot_labels.append([1, 0])
        else:
            onehot_labels.append([0, 1])

    # print(onehot_labels)

    return faces1, faces2, np.array(onehot_labels), paths_batch_1, paths_batch_2
def load_construct_testset_1(size):
    # Load images paths and labels
    pairs = lfw.read_pairs(attack_pairs_path)
    # print('pairs length is ')
    # print(len(pairs))
    paths, labels = lfw.get_test_paths(testset_path, pairs)
    # print('labels length is 6000')
    # print(len(labels))
    # print('paths length is ')
    # print(len(paths))

    # Random choice // permutation 排列 size个数
    permutation = np.random.choice(len(labels), size, replace=False)
    paths_batch_1 = []
    paths_batch_2 = []
    # print('the permutation is ')
    # print(permutation)
    # print(len(permutation))

    # 遍历
    for index in range(len(labels)):
    # for index in permutation:
        # paths_batch_1对应的是path0   paths_batch_2对应的是path1，有同人和不同人区分。
        paths_batch_1.append(paths[index * 2])
        paths_batch_2.append(paths[index * 2 + 1])

    # 取出的size个样本所对应的标签（同人或者非同人）
    # labels = np.asarray(labels)[permutation]

    # print(labels)

    # paths_batch_1 和 paths_batch_2是带有完整路径的图片
    paths_batch_1 = np.asarray(paths_batch_1)
    paths_batch_2 = np.asarray(paths_batch_2)

    # print(len(paths_batch_1))
    # print(paths_batch_1)
    # print(len(paths_batch_2))

    # filenames是只有 Agnelo_Queiroz_0001.png
    # filenames = []
    # for filepath in sorted(tf.gfile.Glob(paths_batch_1)):
    #     filenames.append(os.path.basename(filepath))
    # print(filenames)

    # Load images  img = flip(img, do_random_flip) ????
    faces1 = facenet.load_data(paths_batch_1, False, False, image_size)
    faces2 = facenet.load_data(paths_batch_2, False, False, image_size)

    # 像素归一化到0到1
    min_pixel = min(np.min(faces1), np.min(faces2))
    max_pixel = max(np.max(faces1), np.max(faces2))
    faces1 = (faces1 - min_pixel) / (max_pixel - min_pixel)
    faces2 = (faces2 - min_pixel) / (max_pixel - min_pixel)

    # 将labels中True和False值转换为 [1, 0], [0, 1] 这样的值   True=【1,0】 False=【0,1】
    onehot_labels = []
    for index in range(len(labels)):
        if labels[index]:
            onehot_labels.append([1, 0])
        else:
            onehot_labels.append([0, 1])

    # print(onehot_labels)

    return faces1, faces2, np.array(onehot_labels), paths_batch_1, paths_batch_2
def softmax(distance):
    threshold = 1.1
    # threshold = 0.99
    score = tf.where(
            distance > threshold,
            0.5 + ((distance - threshold) * 0.5) / (4.0 - threshold),
            0.5 * distance / threshold)
    reverse_score = 1.0 - score
    softmax_output = tf.transpose(tf.stack([reverse_score, score]))
    return softmax_output

def prewhitenfacenet(imgs):
    image_size = (160, 160)
    # image, control = imgs.dequeue()
    images = []
    for img1 in tf.unstack(imgs):
        tf.image.per_image_standardization(img1)
        images.append(img1)
    return images