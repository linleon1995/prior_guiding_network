#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:27:56 2019

@author: EE_ACM528_04
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# from tf_tesis2  import unet_multi_task3
from tf_tesis2.eval_utils import compute_mean_dsc, compute_mean_iou, compute_accuracy, load_model, plot_confusion_matrix
# from tf_tesis2.dense_crf import crf_inference
#from tf_tesis2.network import unet, unet_multi_task_fine, unet_multi_task_fine_newz, unet_prior_guide, unet_prior_guide2, unet_prior_guide_encoder, unet_prior_guide_decoder
#from tf_tesis2.network import unet_prior_guide2, unet_prior_guide_encoder, unet_prior_guide_decoder
#from tf_unet_multi_task  import util_multi_task 
import numpy as np
import CT_scan_util_multi_task
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tf_unet_multi_task import unet_multi_task 
#from tf_unet import unet_from_lab_server
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import nibabel as nib
import glob
from tf_tesis2 import stn
PI_ON_180 = 0.017453292519943295
import argparse

N_CLASS = 14
RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_unet_multi_task/unet_mt_trained/unet32_sub25_labelmean/run_006/model.ckpt-55"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_008/model.ckpt-80"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_010/model.ckpt-40"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_022/model.ckpt-40"

#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_002/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_069/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_055/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_034/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_040/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_044/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_048/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_049/model.ckpt.best"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_051/model.ckpt.best"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_077/model.ckpt.best"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_083/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_087/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_091/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_094/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_003/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_008/model.ckpt.best"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_004/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_012/model.ckpt"
DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
#NUM_CLASSES = 14
NUM_STEPS = 1449 # Number of images in the validation set.
Z_CLASS = 3
SUBJECT_LIST = np.arange(25,30)
SHUFFLE = False
IMG_SHAPE = [1, 256, 256, 1]
LABEL_SHAPE = [1, 256, 256, N_CLASS]


#def display_attntion_on_image(img, attention, size=256):
    
    

def display_segmentation(x_test, y_test, pred, n_class):
    fig, (ax11, ax12, ax13) = plt.subplots(1,3)

    ax11.imshow(x_test[0,...,0], 'gray')
    ax11.set_axis_off()     

    ax12.imshow(np.argmax(y_test[0], -1), vmin=0, vmax=n_class-1)
    ax12.set_axis_off()

    ax13.imshow(pred[0], vmin=0, vmax=n_class-1)
    ax13.set_axis_off()
    plt.show()
    
    
#def display_segmentation(x_test, y_test, total_pred, n_class):
#    for i in range(len(x_test)):
#        print('sample: {}'.format(i))
#        fig, (ax11, ax12, ax13) = plt.subplots(1,3)
#
#        ax11.imshow(x_test[i][0,...,0], 'gray')
#        ax11.set_axis_off()     
#
#        ax12.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=n_class-1)
#        ax12.set_axis_off()
#
#        ax13.imshow(total_pred[i][0,...,0], vmin=0, vmax=n_class-1)
#        ax13.set_axis_off()
#        plt.show()
#        plt.close(fig)
        
        
def get_non_zero_idx(subject, n_class):
    organ_list = []
    for c in range(0, n_class):
        start = None
        end = None
        a=None
        for s in range(len(subject)):
            organ_sum = np.sum(subject[s][0,...,c])
            if organ_sum != 0 and start is None:
                start = s
            if organ_sum == 0 and start is not None and end is None:
                end = s-1
            if organ_sum != 0 and start is not None and end is not None:
                print('watch out', len(subject))
                a = s-1
        if end is None and start is not None:
            end = len(subject)-1
        o_idx = (start, end)
        if a is not None:
            o_idx = (start, end, a)
        organ_list.append(o_idx)
        
    return organ_list

def get_data(y, func):
    """
    given one numpy
    input subject
    """
    y_new = [func(np.reshape(sample, [-1, sample.shape[-1]]), axis=0)[np.newaxis] for sample in y]
    return np.concatenate(y_new,0)
 
def get_mean_data(y_test, func, n_frames):
    num = 0
    mean_data = []
    for n in n_frames:
        input_data = y_test[num:num+n]
        num += n
        output = get_data(input_data, func)
        mean_data.append(output)
    return mean_data
    
def plot_organs_data(data, n_subject, n_class, normalize=True):
    ll=[]
    plt.hold(True)
    for j in range(n_subject):
        for i in range(1, n_class):
            ll.append('class{}'.format(i))
            if normalize: 
                x_cord = np.linspace(0,1,data[j].shape[0])
            else:
                x_cord = np.arange(data[j].shape[0])
            plt.plot(x_cord, data[j][:,i])
            plt.legend(ll)
            
#        plt.savefig('subject'+str(j).zfill(3)+'_mean.jpg')
        print('subject: {}, class: {}'.format(j, i))
        _xlabel = 'slice number'
        if normalize: _xlabel = _xlabel + ' (normalized)'
        plt.xlabel(_xlabel)
        plt.ylabel('mean (organ size compare with image)')
        plt.show()

def plot_intersubject_data(data, n_subject, n_class, normalize=True):
    ll=[]
    plt.hold(True)
    for i in range(n_class):
        for j in range(n_subject): 
            ll.append('subject{}'.format(j))
            if normalize: 
                x_cord = np.linspace(0,1,data[j].shape[0])
            else:
                x_cord = np.arange(data[j].shape[0])
            plt.plot(x_cord, data[j][:,i])
            plt.legend(ll)
            
#        plt.savefig('subject'+str(j).zfill(3)+'_mean.jpg')
        print('subject: {}, class: {}'.format(j, i))
        _xlabel = 'slice number'
        if normalize: _xlabel = _xlabel + ' (normalized)'
        plt.xlabel(_xlabel)
        plt.ylabel('mean (organ size compare with image)')
        plt.show()
        
def plot_organ_exist(total_idx, organ_idx, i):
    total_slice = total_idx[1]-total_idx[0]
    start=organ_idx[0]/total_slice
    end=organ_idx[1]/total_slice
    
#    a=np.zeros(total_idx[1]-total_idx[0])
#    a[organ_idx[0]:organ_idx[1]]=i
    plt.plot(np.linspace(start,end,100), i*np.ones(100))

def plot_continuous_2(plt_func, subject, times):
    plt.hold(True)
    for i in range(times):
        plt_func(nonzero_subject_idx[subject][0], nonzero_subject_idx[subject][i], i)
    plt.xlabel('organ exist')
    plt.ylabel('organ')
    
def plot_continuous(plt_func, organ, times):
    plt.hold(True)
    for i in range(times):
        plt_func(nonzero_subject_idx[i][0], nonzero_subject_idx[i][organ], i)
    plt.xlabel('organ exist')
    plt.ylabel('organ')

def normalized_index(index_table, n_class):
    return [np.linspace(0,1,idx[n_class][1]-idx[n_class][0]+1) for idx in index_table]

def get_distance_vector(idx, index_table):
    pass

if __name__ == '__main__':
#    o=7
#    for i in range(100):
#        print(i)
#        fig, (ax11) = plt.subplots(1,1)
#        ss=yy3_spleen[i][0,...,o]*2+yy5_spleen[i][0,...,o]
#        print(np.sum(ss==3))
#        ax11.imshow(ss)
##        ax11.imshow()
#        plt.show()
        
    
#    cmap = plt.cm.jet  
#    o=7
#    for i in range(100,668):
#        print('step {}'.format(i))
#        z_label=z_test[i][0,o]
#        if z_label==1:
#            s=label_bot[0,...,o]
#        elif z_label==2:
#            s=label_mid[0,...,o]
#        elif z_label==3:
#            s=label_top[0,...,o]
#        elif z_label==0:
#            s=np.zeros((256,256))
#        s=(s-np.min(s))/(np.max(s)-np.min(s))
##        s=np.float32(s>0.5)
#        s=cmap(s)
#        s[...,-1]=0.4
#        fig, (ax11,ax12) = plt.subplots(1,2)
#        ax11.imshow(x_test[i][0,...,0], 'gray')
#        ax11.imshow(s)
#        ax12.imshow(y_test[i][0,...,o])
#        ax12.imshow(s)
#        plt.show()
        
#    cmap = plt.cm.jet  
#    o=4
#    for i in range(668):
#        print('step {}'.format(i))
#        z_label=z_test[i][0,0]
#        if z_label==0:
#            s=label_bot[0,...,o]
#        elif z_label==1:
#            s=label_mid[0,...,o]
#        elif z_label==2:
#            s=label_top[0,...,o]
#        s=(s-np.min(s))/(np.max(s)-np.min(s))
#        s=cmap(s)
#        s[...,-1]=0.4
#        fig, (ax11,ax12) = plt.subplots(1,2)
#        ax11.imshow(x_test[i][0,...,0], 'gray')
#        ax11.imshow(s)
#        ax12.imshow(y_test[i][0,...,o])
#        ax12.imshow(s)
#        plt.show()
        
    crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
    bisrgb = np.arange(50, 60, 10)
    bisxy = np.arange(1, 2, 1)

    idx=-1
    
    crf_table = np.zeros(bisrgb.shape + bisxy.shape)
    for i in bisrgb:
        idx += 1
        jdx = -1
        for j in bisxy:
            jdx += 1
            print(i, j)
            crf_config['bi_srgb'] = i
            crf_config['bi_sxy'] = j
#            label_mean, label_top, label_mid, label_bot, slices_nums = main(crf_config)
            x_test, y_test, z_test, prediction, z_pred, logits, total_mToU, total_acc, total_DSC, sum_cm, \
            label_mean, total_co, label_top, label_mid, label_bot, total_prior, \
            non_zero_subject_idx, total_hard_iou, total_soft_iou, z_acc = main(crf_config)
#            crf_table[idx,jdx] = total_DSC_crf 
            
    cmap = plt.cm.jet    
    for i in range(len(x_test)):
#        i=i+100
        print('sample: {}, zlabel: {}, z_prediction: {}'.format(i, int(z_test[i][0,0]), z_pred[i][0]))
        fig, (ax11, ax12, ax13) = plt.subplots(1,3)

        ax11.imshow(x_test[i][0,...,0], 'gray')
#        ax11.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
        ax11.set_axis_off()     

        ax12.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax12.imshow(prediction[i][0,...,0], vmin=0, vmax=13)
        ax12.set_axis_off()

#        ax13.imshow(y_test[i][0,...,7])
#        ax13.imshow(np.argmax(total_crf_out[i][0], -1), vmin=0, vmax=13)
        ax13.imshow(prediction[i][0,...,0], vmin=0, vmax=13)
        ax13.set_axis_off()
        plt.show()
        plt.close(fig)

    