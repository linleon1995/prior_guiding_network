#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 14:09:14 2018

@author: acm528_02
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy.misc import imresize, imrotate
from scipy.ndimage.interpolation import rotate

def check_size(size):
    if type(size) == int:
        size = (size, size)
    if type(size) != tuple:
        raise TypeError('size is int or tuple')
    return size


def resize(image, size, type='img'):
    size = check_size(size)
    if type == 'img':
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_CUBIC)
    elif type == 'label':
        image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_NEAREST)
        image = np.uint8(image)
    return image


#def center_crop(image, crop_size):
#    crop_size = check_size(crop_size)
#    if len(image.shape) == 2:
#        h, w = image.shape
#    elif len(image.shape) == 3:
#        h, w, _ = image.shape
#    top = (h - crop_size[0]) // 2
#    left = (w - crop_size[1]) // 2
#    bottom = top + crop_size[0]
#    right = left + crop_size[1]
#    image = image[top:bottom, left:right]
#    return image

def random_crop(image, crop_size, top_left):
    crop_size = check_size(crop_size)
    bottom = top_left[0] + crop_size[0]
    right = top_left[1] + crop_size[1]
    image = image[top_left[0]:bottom, top_left[1]:right]
    return image

def horizontal_flip(image):
    image = image[:, ::-1]
    return image


def vertical_flip(image):
    image = image[::-1]
    return image


def scale_augmentation(image, scale_size, crop_size, top_left, type='img'):
    image = resize(image, (scale_size, scale_size), type)
    image = random_crop(image, crop_size, top_left)
    return image


def random_rotation(image, angle, type='img', dtype=np.float32):
    if len(image.shape) == 2:
        h, w = image.shape
    elif len(image.shape) == 3:
        h, w, _ = image.shape
    
    if type=='img':
        image = imrotate(image, angle, 'bilinear')
    elif type == 'label':
        image = imrotate(image, angle, 'nearest')
    # TODO: better way to selct cval
#    cval = image[0,0]
    
#    image = rotate(image, angle, cval=cval, output=dtype)
#    image = resize(image, (h, w), type)
    return image

if __name__ == '__main__':
    pass
