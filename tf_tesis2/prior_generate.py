#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 02:27:31 2019

@author: acm528_02
"""

import tensorflow as tf
import numpy as np
import glob
import nibabel as nib
import argparse
import matplotlib.pyplot as plt
import cv2
import os
# An iteratorable object to select necessary files, e.g., list, numpy array 
SUBJECT_LIST = np.arange(5)
#SUBJECT_LIST = None         
RAW_PATH='/home/acm528_02/Jing_Siang/data/Synpase_raw/raw/'  
MASK_PATH='/home/acm528_02/Jing_Siang/data/Synpase_raw/label/'
OUTPUT_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/prior/'


def _get_subject_as_prior(args):
    """
    Reason: Directly average all subject cause noisy prior
    Try to use a clean prior and apply network to affine for better fit
    """
    prior = load_nibabel_data(args.path)
    return prior

def display_prior(images, norm_layer, show_class):
    n=images.shape[0]
    step = n/norm_layer
    index = np.arange(0,n,step)
    index_d = np.int32(index)
    for i in index_d:
        print(i)
        end_idx = min(i+int(step), n)
        plt.imshow(np.sum(images[i:end_idx,...,show_class],axis=0))
        plt.show()
    return index_d

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="This program convert the .nii file from voxel form to slice form, \
                                     and store the same subject in assigned folder with demand format")
    parser.add_argument("--mask_subject_path", type=str, default=MASK_PATH,
                        help="raw input data directory")
    
    parser.add_argument("--prior-dir", type=str, default=OUTPUT_PATH,
                        help="output prior directory")
    return parser.parse_args()

def get_organ_data(data):
    """input the training data and get information of each organ
    Args: data: a list of training subject
    Returns: value to represent the organ size in each slice
    """
    subject_information = []
    for sample in data:
        organ_information = np.sum(np.sum(sample, axis=1), axis=1)
        subject_information.append(organ_information)
    return subject_information

def plot_organ_data(data, num_class, normalize=True):
    """input the training data and plot the total pixel value in each slice
    Args: data: a list of training subject
    """
    subject_information = get_organ_data(data)
    legend_list = []
    for i in range(1,num_class):
        print('class {}'.format(i))
        _, ax = plt.subplots()
        ax.set_xlabel('slice index')
        ax.set_ylabel('value')
        ax.set_title('class {} total pixel value in each slice'.format(i))
        for j in range(len(subject_information)):
            if normalize:
                s = subject_information[j].shape[0]
                x_axis=np.arange(0,1,1/s)
                # TODO: special case when slice number=98 xaxis size would be 99, fix it if have time
                if x_axis.shape[0] > s:
                    x_axis=x_axis[:s]
                ax.plot(x_axis, subject_information[j][...,i])
            else:
                ax.plot(subject_information[j][...,i])
            if i == 1:
                if j!=5:
                    legend_list.append('subject {}'.format(j+1))
                else:
                    legend_list.append('prior')
        ax.legend(legend_list)
        plt.show()  
        
def get_files_name(path, data_suffix='*.jpg'):
    subject = glob.glob(path + data_suffix)
    if not subject:
        raise IOError("No such file data suffix exist")
    subject.sort()
    return subject

def load_nibabel_data(path, num_of_class=None, processing_list=None, onehot_label=False):
    # get file list
    subject = get_files_name(path, data_suffix='*.nii.gz')
    
    # select processing subject by subject_list
    if processing_list is not None:
        subject = [subject[s] for s in processing_list]
    
    # preprocessing
    imgs = []
    for i in subject:
        sample = nib.load(i).get_data()
        sample = np.flip(np.swapaxes(sample, 0, -1), 1)
        if onehot_label:
            if num_of_class is None:
                raise ValueError('TODO!!')
            sample = np.eye(num_of_class)[sample]
            sample = np.uint8(sample)
        imgs.append(sample)
    return imgs

def get_index(num_slice, norm_slice):
    step = num_slice / norm_slice
    index = np.arange(0, num_slice, step)
    index_d = np.int32(index)
    return index_d

def _slice_normalize(subject, norm_slice, subsample='average'):
    """
    Args:
        subject: a list of subject
        norm_slice:
        subsample:
    Return: single array
    """
    # TODO: upsampling strategy
    num_of_slice = subject.shape[0]
    index_d = get_index(num_of_slice, norm_slice)
    num_slice_in_each_prior = []
    if subsample=='select':
        new_subject = subject[index_d]
    elif subsample=='average':
        new_subject = []
        for i in range(norm_slice):
            if i+1 == norm_slice:
                end_idx = num_of_slice
            else:
                end_idx = index_d[i+1]
            slices = subject[index_d[i]:end_idx]
            slices = np.sum(slices, axis=0)[np.newaxis]
            new_subject.append(slices)
            num_slice_in_each_prior.append(end_idx-index_d[i])
        new_subject = np.concatenate(new_subject, axis=0)
    return new_subject, num_slice_in_each_prior

def _get_prior(args,
               processing_list=None,
               norm_slice=100,
               num_class=14,
               num_shard=None,
               value_normalize=None,
               save_prior=True,
               subsample='average',
               ):
    """
    data_dir: a list of subject
    Return: single array
    """
    prior = 0
    # get annotation from training data
    if processing_list is not None:
        for i, processed_subject in enumerate(processing_list):
            subject = load_nibabel_data(args.mask_subject_path, num_of_class=num_class, processing_list=[processed_subject], onehot_label=True)

            # normalize each subject with num_of_slice
            new_subject, num_slice_in_each_prior = _slice_normalize(subject[0], norm_slice)
            
            # get summation of all the normalized subvjects
            prior += new_subject
            if value_normalize == 'frequency_normalize':
                if i==0:
                    total_slice_num = num_slice_in_each_prior
                else:
                    total_slice_num = [x+y for x, y in zip(total_slice_num, num_slice_in_each_prior)]
        if value_normalize is not None:
            if value_normalize == 'minmax_normalize':
                prior = prior / np.max(prior)
            elif value_normalize == 'frequency_normalize':
                prior = [p[np.newaxis]/s for s, p in zip(total_slice_num, prior)]
                prior = np.concatenate(prior, axis=0)
            else:
                raise ValueError('TODO!!')
    else:
        pass
    prior = np.float32(prior)           
    # save prior in npy format
    if save_prior:
        if num_shard is not None:
            num_per_shard = int(np.ceil(norm_slice / float(num_shard)))
            for shard_id in range(num_shard):
                start_idx = shard_id * num_per_shard
                end_idx = min((shard_id + 1) * num_per_shard, norm_slice)
                output_prior_path = os.path.join(args.prior_dir, 'prior_in_'+str(norm_slice)+'/')
                if not os.path.exists(output_prior_path):
                    os.mkdir(output_prior_path)
                np.save(output_prior_path+'num_shard_'+str(shard_id).zfill(4)+'.npy', prior[start_idx:end_idx])             
        else:
            np.save(prior)
    return prior


    

if __name__ == '__main__':  
    args = get_arguments()
    # TODO: args
    # TODO: norm_slice need to smaller than number of slice in each subject
    # TODO: MemoryError when generate prior in 80 slices
    # TODO: mkdir if old prior exist
    # RODO: specific folder name, e.g., normalization method
    prior = _get_prior(args,
                       processing_list=np.arange(25),
                       norm_slice=80,
                       num_class=14,
                       num_shard=5,
                       save_prior=True,
                       value_normalize='frequency_normalize')
    

