#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 12:39:58 2018

@author: EE_ACM528_04
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import os
import click
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import CT_scan_util_multi_task
from tf_tesis2 import unet_multi_task3, prior_generate
get_prior = prior_generate._get_prior
load_nibabel_data = prior_generate.load_nibabel_data


RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
MASK_SUBJECT_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw/label/'
OUTPUT_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/'
PRIOR_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/prior/'
TRAIN_SUBJECT = np.arange(25)
VALID_SUBJECT = np.arange(25,27)
MODEL_FLAG = {'zlevel_classify': True, 'rotate_module': True, 'class_classify': True, 'crf': False}
CRF_CONFIG = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10, "resolution": (32,32)}
WIDTH = 256
HEIGHT = 256
HU_WINDOW = [-180, 250]
HU_WINDOW = [-125,275]
N_CLASS = 14
CLASS_LIST = np.arange(1, 14)
CLASS_LIST = [1,2,3,4,5,6,7]

parser = argparse.ArgumentParser()

parser.add_argument('--raw_path', type=str, default=RAW_PATH,
                    help='')

parser.add_argument('--mask_path', type=str, default=MASK_PATH,
                    help='')

parser.add_argument('--mask-subject-path', type=str, default=MASK_SUBJECT_PATH,
                    help='')

parser.add_argument('--output_path', type=str, default=OUTPUT_PATH,
                    help='')

parser.add_argument('--prior-dir', type=str, default=PRIOR_PATH,
                    help='')

parser.add_argument('--training_iters', type=int, default=None,
                    help='')

parser.add_argument('--epochs', type=int, default=100,
                    help='')

parser.add_argument('--restore', type=bool, default=False,
                    help='')

parser.add_argument('--layers', type=int, default=5,
                    help='')

parser.add_argument('--features_root', type=int, default=32,
                    help='')

parser.add_argument('--batch_size', type=int, default=18,
                    help='')

parser.add_argument('--data_augmentation', type=bool, default=False,
                    help='')

parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='')

parser.add_argument('--cubic', type=bool, default=False,
                    help='')

parser.add_argument('--z_class', type=int, default=60,
                    help='')

parser.add_argument('--power', type=float, default=0.9,
                    help='')

parser.add_argument('--lambda-z', type=float, default=1e-1,
                    help='')

parser.add_argument('--lambda-angle', type=float, default=1e-3,
                    help='')

parser.add_argument('--only_foreground', type=bool, default=False,
                    help='')

parser.add_argument('--shuffle-data', type=bool, default=True,
                    help='')

parser.add_argument('--seq-length', type=int, default=None,
                    help='')

parser.add_argument('--subject_for_prior', type=str, default='label0001.nii.gz',
                    help='')

def create_training_path(output_path):
    idx = 0
    path = os.path.join(output_path, "run_{:03d}".format(idx))
    while os.path.exists(path):
        idx += 1
        path = os.path.join(output_path, "run_{:03d}".format(idx))
    return path


def launch():
    print("Using raw data from: %s"%FLAGS.raw_path)
    print("Using label data from: %s"%FLAGS.mask_path)
    print("Using zaxis class from: %s"%FLAGS.z_class)

    data_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=FLAGS.raw_path,
                                      mask_path=FLAGS.mask_path,
                                      subject_list=TRAIN_SUBJECT,
                                      class_list = CLASS_LIST,
                                      resize_ratio=0.5,
                                      data_aug=FLAGS.data_augmentation,
                                      cubic=FLAGS.cubic,
                                      z_class=FLAGS.z_class,
                                      nx=HEIGHT,
                                      ny=WIDTH,
                                      HU_window=HU_WINDOW,
                                      only_foreground = FLAGS.only_foreground,
                                      shuffle_data=FLAGS.shuffle_data,
                                      seq_length=FLAGS.seq_length,
                                      mode=None,
                                      )

    valid_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=FLAGS.raw_path,
                                      mask_path=FLAGS.mask_path,
                                      subject_list=VALID_SUBJECT,
                                      class_list = CLASS_LIST,
                                      resize_ratio=0.5,
                                      data_aug=FLAGS.data_augmentation,
                                      cubic=FLAGS.cubic,
                                      z_class=FLAGS.z_class,
                                      nx=HEIGHT,
                                      ny=WIDTH,
                                      HU_window=HU_WINDOW,
                                      only_foreground = FLAGS.only_foreground,
                                      shuffle_data=FLAGS.shuffle_data,
                                      seq_length=FLAGS.seq_length,
                                      mode=None,
                                      )
    
    n_sample = len(data_provider._find_data_files())
    if FLAGS.training_iters is None:
        FLAGS.training_iters = n_sample // FLAGS.batch_size
        print('training itration:{}'.format(FLAGS.training_iters))

    prior_folder = FLAGS.prior_dir+'prior_in_'+str(FLAGS.z_class)+'/'
    # TODO: inspect npy file exist or not
    # TODO: logging

    ref_model = load_nibabel_data(FLAGS.mask_subject_path, processing_list=np.arange(1))[0]

    # if not os.path.isdir(prior_folder):
    #     print("Generating prior!!")
    #     ref_model = get_prior(FLAGS,
    #                    processing_list=TRAIN_SUBJECT,
    #                    norm_slice=data_provider.z_class,
    #                    num_class=data_provider.n_class,
    #                    num_shard=5,
    #                    save_prior=True,
    #                    value_normalize='frequency_normalize')
    # else:
    #     print("Prior has already exist in: {}".format(prior_folder))
    #     ref_model = np.concatenate([np.load(prior_folder+'num_shard_0000.npy'),
    #                                 np.load(prior_folder+'num_shard_0001.npy'),
    #                                 np.load(prior_folder+'num_shard_0002.npy'),
    #                                 np.load(prior_folder+'num_shard_0003.npy'),
    #                                 np.load(prior_folder+'num_shard_0004.npy')], 0)
    
    net = unet_multi_task3.Unet(model_flag=MODEL_FLAG,
                                nx=HEIGHT,
                                ny=WIDTH,
                                channels=data_provider.channels, 
                                n_class=data_provider.n_class, 
                                layers=FLAGS.layers, 
                                features_root=FLAGS.features_root,
            #                    cost="cross_entropy",
                                cost="mean_dice_coefficient", 
                                norm=True,
                                pretrained=None,
            #                    cost_kwargs={'regularizer': 1e-4}
                                summaries=True,
                                z_class = data_provider.z_class,
                                prior = ref_model,
                                batch_size = FLAGS.batch_size,
                                lambda_z = FLAGS.lambda_z,
                                lambda_angle = FLAGS.lambda_angle,
                                seq_length=FLAGS.seq_length,
#                                data_aug='resize_and_crop'
                                )
    kwargs={"learning_rate": FLAGS.learning_rate, "power": FLAGS.power, "epochs": FLAGS.epochs}
    path = FLAGS.output_path if FLAGS.restore else create_training_path(FLAGS.output_path)
    
    trainer = unet_multi_task3.Trainer(net, batch_size=FLAGS.batch_size, norm_grads=True, optimizer="adam", opt_kwargs=kwargs)
    path = trainer.train(data_provider, 
                         valid_provider,
                         path, 
                         training_iters=FLAGS.training_iters, 
#                         epochs=FLAGS.epochs, 
                         dropout=1, 
                         display_step=2, 
                         restore=FLAGS.restore)

if __name__ == '__main__':
    pass
    FLAGS, unparsed = parser.parse_known_args()
    launch()


    