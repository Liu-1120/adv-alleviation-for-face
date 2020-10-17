#!usr/bin/python
# -*- coding: utf-8 -*-
"""Helper for evaluation on the Labeled Faces in the Wild dataset
"""

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

import os
import numpy as np
import facenet

def evaluate(embeddings, actual_issame, nrof_folds=10, distance_metric=0, subtract_mean=False):
    # Calculate evaluation metrics
    thresholds = np.arange(0, 4, 0.01)
    embeddings1 = embeddings[0::2]
    embeddings2 = embeddings[1::2]
    tpr, fpr, accuracy = facenet.calculate_roc(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    thresholds = np.arange(0, 4, 0.001)
    val, val_std, far = facenet.calculate_val(thresholds, embeddings1, embeddings2,
        np.asarray(actual_issame), 1e-3, nrof_folds=nrof_folds, distance_metric=distance_metric, subtract_mean=subtract_mean)
    return tpr, fpr, accuracy, val, val_std, far
def get_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    pathname_list = []
    # print('pairs is ')
    # print(pairs)
    for pair in pairs:

        # 同一个人的两张照片
        # if len(pair) == 2:
            # path0 和 path1 代表同一个人的两张不同图片
            # path0 = os.path.join(lfw_dir, pair[0])
            # path1 = os.path.join(lfw_dir, pair[1])
            # # path_name = pair[0]
            # # True 代表同人
            # issame = True
        # # 不同人的两张照片
        # elif len(pair) == 4:
        #     # path0 和 path1 代表不同人的两张不同图片
        #     path0 = os.path.join(lfw_dir, pair[2])
        #     path1 = os.path.join(lfw_dir, pair[3])
        #     path_name = pair[0]
        #     # False代表不同人
        #     issame = True
        if len(pair)==4:
            path0=os.path.join(lfw_dir,pair[0],pair[1])
            path1=os.path.join(lfw_dir,pair[0],pair[2])
            issame = True
        if len(pair)==5:
            path0 = os.path.join(lfw_dir, pair[0], pair[1])
            path1 = os.path.join(lfw_dir, pair[2], pair[3])
            issame = False
        # print(len(pair))
        # if len(pair) == 3:
        #     # path0 和 path1 代表同一个人的两张不同图片
        #     # path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
        #     # path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
        #     path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%d' % int(int(pair[1]))))
        #     path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%d' % int(int(pair[2]))))
        #     path_name = pair[0]
        #     # True 代表同人
        #     issame = True
        #     # 不同人的两张照片
        # elif len(pair) == 4:
        #     # path0 和 path1 代表不同人的两张不同图片
        #     path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%d' % int(int(pair[1]))))
        #     path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%d' % int(int(pair[3]))))
        #     path_name = pair[0]
        #     # False代表不同人
        #     issame = False

        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
            # pathname_list.append(path_name)

        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    print('path_list is ')
    print(path_list)
    print('path_list length is ')
    print(len(path_list))
    return path_list, issame_list, pathname_list


def get_paths_adv(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    pathname_list = []
    # print('pairs is ')
    # print(pairs)
    for pair in pairs:
        # print(pair)

        if len(pair) == 3:
            # path0 和 path1 代表同一个人的两张不同图片
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])))
            #
            # print(path_name)
            # True 代表同人
            issame = True
            # 不同人的两张照片
        elif len(pair) == 4:
            # path0 和 path1 代表不同人的两张不同图片
            path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
            path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
        
            # False代表不同人
            issame = False

        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)

        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    print('path_list is ')
    print(path_list)
    print('path_list length is ')
    print(len(path_list))
    return path_list, issame_list
  
def add_extension(path):
    if os.path.exists(path+'.jpg'):
        return path+'.jpg'
    elif os.path.exists(path+'.png'):
        return path+'.png'
    else:
        raise RuntimeError('No file "%s" with extension png or jpg.' % path)

def read_pairs(pairs_filename):
    pairs = []
    with open(pairs_filename, 'r') as f:
        for line in f.readlines()[1:]:
            pair = line.strip().split()
            pairs.append(pair)
    return np.array(pairs)

def get_test_paths(lfw_dir, pairs):
    nrof_skipped_pairs = 0
    path_list = []
    issame_list = []
    # print('pairs is ')
    # print(pairs)
    for pair in pairs:
        print(len(pair))
        # 同一个人的两张照片
            # path0 和 path1 代表同一个人的两张不同图片

        path0 = os.path.join(lfw_dir, pair[0], pair[2])
        path1 = os.path.join(lfw_dir, pair[1], pair[3])
        # True 代表同人
        issame = True
        # 不同人的两张照片
        # elif len(pair) == 4:
        #     # path0 和 path1 代表不同人的两张不同图片
        #     path0 = add_extension(os.path.join(lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])))
        #     path1 = add_extension(os.path.join(lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])))
        #     # False代表不同人
        #     issame = False
        if os.path.exists(path0) and os.path.exists(path1):
            path_list += (path0, path1)
            issame_list.append(issame)
        else:
            nrof_skipped_pairs += 1
    if nrof_skipped_pairs > 0:
        print('Skipped %d image pairs' % nrof_skipped_pairs)
    print('path_list is ')
    print(path_list)
    print('path_list length is ')
    print(len(path_list))
    return path_list, issame_list

