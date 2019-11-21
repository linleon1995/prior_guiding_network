#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:07:35 2018

@author: EE_ACM528_04
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import numpy as np
import nibabel as nib
import cv2
import glob
import pickle
import tensorflow as tf
#from tf_unet.image_util import BaseDataProvider
from data_augmentation import random_rotation, horizontal_flip, vertical_flip, scale_augmentation
from tf_tesis2 import stn, np_stn, prior_generate
import time

def _preprocess_zero_mean_unit_range(inputs, dtype=np.float32):
  """Map image values from [0, 255] to [-1, 1]."""
  preprocessed_inputs = (2.0 / 1.0) * dtype(inputs) - 1.0
  return dtype(preprocessed_inputs)


class BaseDataProvider(object):
    """
    Abstract base class for DataProvider implementation. Subclasses have to
    overwrite the `_next_data` method that load the next data and label array.
    This implementation automatically clips the data with the given min/max and
    normalizes the values to (0,1]. To change this behavoir the `_process_data`
    method can be overwritten. To enable some post processing such as data
    augmentation the `_post_process` method can be overwritten.

    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping

    """
    
    channels = 1
    n_class = 2
    
    
    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf
#        self.only_raw = only_raw
        
    def _load_data_and_label(self):
        data, label = self._next_data()
  
        train_data = self._process_data(data)
        labels = self._process_labels(label)

        train_data, labels = self._post_process(train_data, labels)
        
        nx = train_data.shape[1]
        ny = train_data.shape[0]

        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _load_data(self):
        data = self._next_data()
            
        train_data = self._process_data(data)

        nx = train_data.shape[1]
        ny = train_data.shape[0]
        return train_data.reshape(1, ny, nx, self.channels)
    
    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        
        return label
    
    def _process_data(self, data):
        # normalization
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data
    
    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        """
        return data, labels
    
    def __call__(self, n):
        
        if self.only_raw is not True:
            train_data, labels = self._load_data_and_label()

            nx = train_data.shape[1]
            ny = train_data.shape[2]
        
            X = np.zeros((n, nx, ny, self.channels))
            Y = np.zeros((n, nx, ny, self.n_class))
        
            X[0] = train_data
            Y[0] = labels
            for i in range(1, n):
                train_data, labels = self._load_data_and_label()
                X[i] = train_data
                Y[i] = labels
            
            return X, Y
        else:
            train_data = self._load_data()
            nx = train_data.shape[1]
            ny = train_data.shape[2]
        
            X = np.zeros((n, nx, ny, self.channels))
   
            X[0] = train_data

            for i in range(1, n):
                train_data = self._load_data()
                X[i] = train_data
       
            return X
        
    
class MedicalDataProvider(BaseDataProvider):
    # TODO: description, param
    """
    Generic data provider for medical images.
    Assumes that the data images and label images are stored in the different folder
    and that the labels have a different file suffix 
    e.g. 'train/fish_1.tif' and 'train/fish_1_mask.tif'

    Usage:
    data_provider = MedicalDataProvider("..fishes/train/*.tif")
        
    :param search_path: a glob search pattern to find all data and label images
    :param a_min: (optional) min value used for clipping
    :param a_max: (optional) max value used for clipping
    :param data_suffix: suffix pattern for the data images. Default '.tif'
    :param mask_suffix: suffix pattern for the label images. Default '_mask.tif'
    :param shuffle_data: if the order of the loaded file path should be randomized. Default 'True'
    :param channels: (optional) number of channels, default=1
    :param n_class: (optional) number of classes, default=2
    
    """
    
    def __init__(self, raw_path=None, mask_path=None, subject_list=None, imgs=None, imgs_mask=None, shuffle_data=True, 
                 class_list=np.arange(14), resize_ratio=0.5, only_raw=False, only_foreground=False, HU_window=[-180,250], data_aug=False, cubic=True, 
                 z_class=None, nx=256, ny=256, mode='affine', seq_length=None):
        self.raw_path = raw_path
        self.mask_path = mask_path
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.class_list = class_list
        self.n_class = len(self.class_list) + 1
        self.subject_list = subject_list
        self.HU_window = HU_window
        
        self.n_frames = [len(glob.glob(raw_path+'subject'+str(subject).zfill(4)+'/*.nii.gz')) for subject in subject_list]
        self.only_foreground = only_foreground
        with open('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/nonzero_subject_idx.pickle', 'rb') as file:
            nonzero_subject_idx = pickle.load(file)
        self.nonzero_subject_idx = [nonzero_subject_idx[idx] for idx in subject_list]
        self.seq_length = seq_length
        
        self.data_files = self._find_data_files()

        self.resize_ratio = resize_ratio  
        self.only_raw = only_raw
        self.data_aug = data_aug
        
        self.data_horizontal_flip=False
        self.data_vertical_flip=False
        self.data_crop_and_scale=False
        self.data_rotate=False
        self.mode=mode
        
        self.channels = 1
        self.cubic = cubic
        self.z_class = z_class
        self.nx = nx
        self.ny = ny
        self.ignore_label = 255
        self.onehot_label = True
        
        
        
        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)
        
        assert len(self.data_files) > 0, "No training files"
#        if only_raw is False:
#            assert imgs_mask is not None, 'imgs_mask do not exist'
        
        if self.seq_length is not None:
            assert self.seq_length%2 != 0
        print("Number of files used: %s" % len(self.data_files))
        
        
    def _load_data_and_label(self):
        data = self._next_data()      
        
        if self.only_raw:
            train_data = data[0]     
            train_data = self._process_data(train_data)
            self.nx = train_data.shape[1]
            self.ny = train_data.shape[0]
            train_data.reshape(1, self.ny, self.nx, self.channels)
            data[0] = train_data
        else:
            train_data = data[0]
            label = data[1]

            train_data = self._process_data(train_data)

            labels = self._process_labels(label)

            train_data, labels, angle, class_gt = self._post_process(train_data, labels)
            
#            nx = train_data.shape[1]
#            ny = train_data.shape[0]
            if self.seq_length is not None:
                train_data = [sample.reshape(1, self.ny, self.nx, self.channels) for sample in train_data]
                data[0] = train_data
                labels = [sample.reshape(1, self.ny, self.nx, self.n_class) for sample in labels]
                data[1] = labels
            else:
                data[0] = train_data.reshape(1, self.ny, self.nx, self.channels)
                data[1] = labels.reshape(1, self.ny, self.nx, self.n_class)

        return data, angle, class_gt

    
    def __call__(self, n):
        if self.only_raw is not True:
            if self.z_class is not None:
                if self.seq_length is not None:
                    X = np.zeros((n*self.seq_length, self.nx, self.ny, self.channels), dtype=np.float32)
                    Y = np.zeros((n*self.seq_length, self.nx, self.ny, self.n_class))
                    z_label = np.zeros((n*self.seq_length, self.n_class))
                    angle_label = np.zeros((n*self.seq_length, 2, 3), dtype=np.float32)
                else:
                    X = np.zeros((n, self.nx, self.ny, self.channels))
                    Y = np.zeros((n, self.nx, self.ny, self.n_class))
                    z_label = np.zeros((n, 1))
#                    z_label = np.zeros((n, self.n_class))
                    angle_label = np.zeros((n, 2, 3), dtype=np.float32)

                class_label = np.zeros((n, self.n_class), dtype=np.int32)
                
                for i in range(n):
                    data, angle, class_gt = self._load_data_and_label()
                    train_data = data[0]
                    labels = data[1]
                    z = data[2]
                    if self.seq_length is not None:
                        for s in range(self.seq_length): 
                            X[s*n+i] = train_data[s]
                            Y[s*n+i] = labels[s]
                            z_label[s*n+i] = z[s]
                            angle_label[s*n+i] = angle[s]
                    else:
                        X[i] = train_data
                        Y[i] = labels
                        z_label[i] = z
                        angle_label[i] = angle
                    class_label[i] = class_gt
                
#                time_f=time.time()
                return X, Y, z_label, angle_label, class_label
            else:
                data, angle = self._load_data_and_label()
                train_data = data[0]
                labels = data[1]
                
#                nx = train_data.shape[1]
#                ny = train_data.shape[2]
            
                X = np.zeros((n, self.nx, self.ny, self.channels))
                Y = np.zeros((n, self.nx, self.ny, self.n_class))
            
                X[0] = train_data
                Y[0] = labels
                for i in range(1, n):
                    data = self._load_data_and_label()
                    train_data = data[0]
                    labels = data[1]
                
                    X[i] = train_data
                    Y[i] = labels
                
                return X, Y
        else:
            train_data = self._load_data()
#            nx = train_data.shape[1]
#            ny = train_data.shape[2]
        
            X = np.zeros((n, self.nx, self.ny, self.channels))
   
            X[0] = train_data

            for i in range(1, n):
                train_data = self._load_data()
                X[i] = train_data
       
            return X
        
    def _loader(self, file_name, path, data_index, data_suffix='.nii.gz'):
        path = path + 'subject' + str(data_index[0]).zfill(4) + '/'
        nib_data = nib.load(path+file_name+str(data_index[0]).zfill(4)+'_'+str(data_index[1]).zfill(4)+data_suffix)
        return nib_data.get_data()
    
    def _forward(self, idx, length):
        """
        forward sequence generator
        Given idx (the output frame), this function can extend idx to sequence idx in forward direction
        e.g., input: idx=(3) length=4, output: seq_idx=(0, 1, 2, 3)
        """
        new_idx = np.arange(idx[1]-length+1, idx[1]+1)
        new_idx[new_idx<0]=-1
        return tuple(new_idx)

    def _backward(self, idx, num, length):
        """
        backward sequence generator
        Given idx (the output frame), this function can extend idx to sequence idx in backward direction
        e.g., input: idx=(7) length=3, output: seq_idx=(7, 8, 9)
        """
        new_idx = np.arange(idx[1], idx[1]+length)
        new_idx[new_idx>num-1]=-1
        return tuple(new_idx)
    
    def img_to_seq(self, img_idx):
        # TODO: description
        seq_idx = []
        for idx in img_idx:
            num = self.n_frames[idx[0]-self.subject_list[0]]
            offset = self.seq_length // 2
            new_idx = np.arange(idx[1]-offset, idx[1]+offset+1)
            new_idx[new_idx<0]=0
            new_idx[new_idx>num-1]=num-1
            seq_idx.append((idx[0],) + tuple(new_idx))
        return seq_idx
    
    def _find_data_files(self):
        index_img = []
        index_subject = []
        for i, subject in enumerate(self.n_frames):
            if self.only_foreground:
                min_start = 1e5
                max_end = 0
                for j in self.class_list:
                    foreground_start = self.nonzero_subject_idx[i][j][0]
                    foreground_end = self.nonzero_subject_idx[i][j][1]
                    if foreground_start is None or foreground_end is None:
                        continue
                    if min_start > foreground_start:
                        min_start = foreground_start
                    if max_end < foreground_end:
                        max_end = foreground_end
                n_frame_arr = np.arange(min_start, max_end)
            else:
                n_frame_arr = np.arange(subject)
                
            index_img = index_img + [n_frame_arr]
            index_subject = index_subject + [self.subject_list[i]*np.ones(len(n_frame_arr))]
            
        index_subject = np.concatenate(index_subject, axis=0)
        index_img = np.concatenate(index_img, axis=0)
        index_total = np.concatenate((index_subject[...,np.newaxis],index_img[...,np.newaxis]), axis=1)
        index_total = np.uint8(index_total)
        
        index_total = [tuple(i) for i in index_total]
        
        if self.seq_length is not None:
            index_total = self.img_to_seq(index_total)

#        print(index_total)        
        return index_total
    
    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
    
    def get_z_label(self, idx):
        # TODO: can start in nonzero, still cannot random choose
        z_level = self.n_frames[idx[0]-self.subject_list[0]] // self.z_class
        z_label = idx[1] // z_level
        if z_label >= self.z_class:
            z_label = self.z_class - 1
        return z_label
    
    def get_z_label_new(self, idx):
        z_label_list = np.zeros((1,self.n_class), dtype=np.int32)
        
        for i, c in enumerate([0]+self.class_list):
            start, end = self.nonzero_subject_idx[idx[0]-self.subject_list[0]][c]
            if start is None and end is None:
                continue
            z_level = (end-start+1) // self.z_class
            
            # check organ exist
            if idx[1]<start or idx[1]> end:
                # class 0 for zero value
                z_label = 0
            else:
                z_label = (idx[1]-start) // z_level
                if z_label >= self.z_class:
                    z_label = self.z_class - 1
                
                # exist organ class need to bigger than 0
                z_label += 1
                
            z_label_list[0,i] = z_label
        return z_label_list
    
#    def get_z_label_new2(self, idx):
#        z_label_list = np.zeros((1,1), dtype=np.int32)
#        z_level = self.n_frames[idx[0]-self.subject_list[0]] / self.z_class
#        if int(idx[1] / z_level) != int((idx[1]+1) / z_level):
#            z_label_list[0,0] = min(self.z_class-1, int((idx[1]+1) / z_level))
#        else:
#            z_label_list[0,0] = int((idx[1]) / z_level)
#        return z_label_list
    
    def get_z_label_new2(self, idx):
        """For"""
        z_label_list = np.zeros((1,1), dtype=np.float32)
        num_slice = self.n_frames[idx[0]-self.subject_list[0]]
        z_label_list[0,0] = idx[1] / num_slice
        return z_label_list
    
    def get_image_label(self, label):
        img_label = np.zeros((self.n_class), dtype=np.uint8)
        for k in range(self.n_class):
            if np.sum(np.where(label==k)) != 0:
                img_label[k] = 1
        return img_label
    
    
    def _next_data(self):
        self._cylce_file()   
        idx = self.data_files[self.file_idx]
        if self.seq_length is not None:
            img = [self._loader('imgs', self.raw_path, (idx[0], i)) for i in idx[1:]]       
            label = [self._loader('label', self.mask_path, (idx[0], i)) for i in idx[1:]]  
        else:
            img = self._loader('imgs', self.raw_path, idx)   
            
        if self.only_raw:
            data = img         
        else:
            if self.seq_length is None:
                label = self._loader('label', self.mask_path, idx)
            data = [img, label]
            
        if self.z_class is not None:
            if self.seq_length is not None:
                z_label = [self.get_z_label_new2((idx[0], i)) for i in idx[1:]]
            else:
                z_label = self.get_z_label_new2(idx)
            data.append(z_label)

        return data
        
    def _process_data(self, data):
        # HU window
        if self.seq_length is not None:
            for i in range(len(data)):
                sample = data[i]
                sample[sample>self.HU_window[1]] = self.HU_window[1]
                sample[sample<=self.HU_window[0]] = self.HU_window[0]
                sample = sample - np.amin(sample)
                sample = sample / np.amax(sample)
                data[i] = sample
                
            # raw_data resize
            if self.cubic:        
                data = [cv2.resize(sample, dsize=(self.nx, self.ny), interpolation=cv2.INTER_CUBIC) for sample in data]
            else:
                data = [cv2.resize(sample, dsize=(self.nx, self.ny), interpolation=cv2.INTER_LINEAR) for sample in data]
            data = np.float32(data)
        else:
            data[data>self.HU_window[1]] = self.HU_window[1]
            data[data<=self.HU_window[0]] = self.HU_window[0]

            data = data - np.amin(data)
            data = data / np.amax(data)

            if self.cubic:
                data = cv2.resize(data, dsize=(self.nx, self.ny), interpolation=cv2.INTER_CUBIC)
            else:
                data = cv2.resize(data, dsize=(self.nx, self.ny), interpolation=cv2.INTER_LINEAR)
            data = np.float32(data)

        return data
            
    def _process_labels(self, label):
        """
        one_hot_coding
        in: (row, col), out: (row, col, classes)
        """
        # label resize
        if self.seq_length is not None:
            for i in range(len(label)):
                sample = label[i]
                sample = cv2.resize(sample, dsize=(self.nx, self.ny), interpolation=cv2.INTER_NEAREST)
                label[i] = sample
        else:
            label = cv2.resize(label, dsize=(self.nx, self.ny), interpolation=cv2.INTER_NEAREST)

           
        return label

    def _post_process(self, data, labels):
        """
        Post processing hook that can be used for data augmentation
        
        :param data: the data array
        :param labels: the label array
        :param data_aug
        
        input: data, labels [H, W]
        """           
        # data augmentation  
        if self.data_rotate:
            if np.random.rand() < 0.5:
                angle = np.random.randint(*(-10,10))
                data = random_rotation(data, angle, type='img', dtype=np.float32)
                labels = random_rotation(labels, angle, type='label', dtype=np.uint8)
                
        if self.data_horizontal_flip:         
            if np.random.rand() < 0.5:
                data = horizontal_flip(data)
                labels = horizontal_flip(labels)
                
        if self.data_vertical_flip:     
            if np.random.rand() < 0.5:
                data = vertical_flip(data)
                labels = vertical_flip(labels)
                
        if self.data_crop_and_scale:     
            if np.random.rand() < 0.5:
                scale_size = np.random.randint(*(self.ny+1,300)) 
                crop_size = (self.nx, self.ny)
                top_left = (np.random.randint(0, scale_size-crop_size[0]), np.random.randint(0, scale_size-crop_size[1]))
                data = scale_augmentation(data, scale_size, crop_size, top_left, type='img')
                labels = scale_augmentation(labels, scale_size, crop_size, top_left, type='label')
        

        if self.mode == 'rotate':
            data = np.reshape(data, [1,self.nx,self.ny,1])
            labels = np.reshape(labels, [1,self.nx,self.ny,1])
    
            angle = np.random.randint(*(-10,10))
            
            rad = np.reshape(angle * 0.017453292519943295, [1,1,1])
            cos_rad = np.cos(rad)
            sin_rad = np.sin(rad)
            t1 = np.concatenate([cos_rad, -sin_rad, np.zeros_like(cos_rad)], 2)
            t2 = np.concatenate([sin_rad, cos_rad, np.zeros_like(cos_rad)], 2)
            theta = np.concatenate([t1, t2], 1)
            
            batch_grids = np_stn.affine_grid_generator_np(self.nx, self.ny, theta)
    
            x_s = batch_grids[:, 0, :, :]
            y_s = batch_grids[:, 1, :, :]
            
            data = np_stn.bilinear_sampler_np(data, x_s, y_s)
            labels = np.float32(labels)
            labels = np_stn.nearlist_sampler_np(labels, x_s, y_s)
            
            data = data[0,...,0]
            labels = labels[0,...,0]
            
        elif self.mode == 'affine':
            if self.seq_length is not None:
                data = [np.reshape(sample, [1,self.nx,self.ny,1]) for sample in data]
                labels = [np.reshape(sample, [1,self.nx,self.ny,1]) for sample in labels]
            else:
                data = np.reshape(data, [1,self.nx,self.ny,1])
                labels = np.reshape(labels, [1,self.nx,self.ny,1])
    
            angle = np.random.randint(*(-10,10))
            scale = np.random.uniform(0.8,1.2)
            shift_x = np.random.uniform(-0.4,0.4)
            shift_y = np.random.uniform(-0.4,0.4)
            skew_x = np.random.uniform(-0.2,0.2)
            skew_y = np.random.uniform(-0.2,0.2)
            
            rad = np.reshape(angle * 0.017453292519943295, [1,1,1])
            cos_rad = np.cos(rad)
            sin_rad = np.sin(rad)
#            t1 = np.concatenate([0.8*cos_rad, -sin_rad, np.zeros_like(cos_rad)], 2)
#            t2 = np.concatenate([sin_rad, 0.8*cos_rad, np.zeros_like(cos_rad)], 2)
#
#            theta = np.concatenate([t1, t2], 1)
            
#            angle = 0
            theta = np.array([[[scale*cos_rad, -sin_rad*skew_y, shift_x],
                               [scale*sin_rad*skew_x, cos_rad, shift_y]]])
    
#            theta = np.array([[[1, skew_y, 0],
#                               [skew_x, 1, 0]]])
            batch_grids = np_stn.affine_grid_generator_np(self.nx,self.ny, theta)
    
            x_s = batch_grids[:, 0, :, :]
            y_s = batch_grids[:, 1, :, :]
            
            if self.seq_length is not None:
                for i in range(len(data)):
                    sample = data[i]
                    sample = np_stn.bilinear_sampler_np(sample, x_s, y_s)
                    sample = sample[0,...,0]
                    sample = sample - np.amin(sample)
                    sample = sample / np.amax(sample)
                    data[i] = sample
                    
                    sample2 = labels[i]
                    sample2 = np.float32(sample2)
                    sample2 = np_stn.nearlist_sampler_np(sample2, x_s, y_s)
                    sample2 = sample2[0,...,0]
                    labels[i] = sample2
                angle=np.tile(theta,(self.seq_length,1,1))
            else:
                data = np_stn.bilinear_sampler_np(data, x_s, y_s)
                data = data[0,...,0]
                data = data - np.amin(data)
                data = data / np.amax(data)
                
                labels = np.float32(labels)
                labels = np_stn.nearlist_sampler_np(labels, x_s, y_s)
    
                labels = labels[0,...,0]
                angle=theta
            
        else:
            angle = np.array([[[1, 0, 0],
                                   [0, 1, 0]]])
            if self.seq_length is not None:
                angle=np.tile(angle,(self.seq_length,1,1))
                

        
        
        # class_label
        class_gt = self.get_image_label(labels)
        
        
        # normalization
#        data = _preprocess_zero_mean_unit_range(data)
#        data = data - np.amin(data)
#        data = data / np.amax(data)

        # one-hot coding
        if self.seq_length is not None:
            new_label =  []
            for l in labels:
                new_labels = np.zeros_like(l)
                
                for idx, c in enumerate(self.class_list):
                    new_labels += (idx+1)*np.uint8(l==c)
         
                one_hot_imgs = np.eye(self.n_class)[np.int32(new_labels)]
                
        #        one_hot_imgs = np.zeros((np.shape(labels)+(self.n_class,)), dtype=np.uint8)
        #        for i in range(self.n_class):
        #            one_hot_imgs[labels==i, i] = 1 
                
                one_hot_imgs = np.uint8(one_hot_imgs)
                new_label.append(one_hot_imgs)
            one_hot_imgs = new_label
        else:
            new_labels = np.zeros_like(labels)
            
            for idx, c in enumerate(self.class_list):
                new_labels += (idx+1)*np.uint8(labels==c)
            if self.onehot_label:
                new_labels = np.eye(self.n_class)[np.int32(new_labels)]
                new_labels = np.uint8(new_labels)
        
        
        return data, new_labels, angle, class_gt

  
