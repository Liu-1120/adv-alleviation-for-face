import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=96, hrg=96, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def retain(x, is_random=True):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    x = imresize(x, size=[160, 160], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x
def gaussian(x, is_random=True):
    batch_shape = x.shape()
    x = x + np.random.randn(batch_shape) * 0.5
    x = x / (255. / 2.)
    x = x - 1.
    return x

def retain_0_1(x, is_random=True):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    x = np.multiply(np.subtract(x, mean), 1 / std_adj)
    # print(x)
    x = crop(x, False, 160)
    x = flip(x, False)
    min_pixel = np.min(x)
    max_pixel = np.max(x)
    x = (x - min_pixel) / (max_pixel - min_pixel)
    return x


def crop(image, random_crop, image_size):
    if image.shape[1] > image_size:
        sz1 = int(image.shape[1] // 2)
        sz2 = int(image_size // 2)
        if random_crop:
            diff = sz1 - sz2
            (h, v) = (np.random.randint(-diff, diff + 1), np.random.randint(-diff, diff + 1))
        else:
            (h, v) = (0, 0)
        image = image[(sz1 - sz2 + v):(sz1 + sz2 + v), (sz1 - sz2 + h):(sz1 + sz2 + h), :]
    return image


def flip(image, random_flip):
    if random_flip and np.random.choice([True, False]):
        image = np.fliplr(image)
    return image