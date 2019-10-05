#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:27:56 2019

@author: EE_ACM528_04
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tf_tesis2  import unet_multi_task3
from tf_tesis2.eval_utils import compute_mean_dsc, compute_mean_iou, compute_accuracy, load_model, plot_confusion_matrix
from tf_tesis2.dense_crf import crf_inference
from tf_tesis2.network import unet_prior_guide2, unet_prior_guide_encoder, unet_prior_guide_decoder
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
import pickle

MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_004/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_012/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_027/model.ckpt"

RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
REG_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw/reg_data/'
N_CLASS = 14
IGNORE_LABEL = 255
Z_CLASS = 3
SUBJECT_LIST = np.arange(24)
SHUFFLE = False
IMG_SIZE = 256
CRF_CONFIG = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
HU_WINDOW = [-180, 250]

#def get_new_z2(z_class):
#    # get slice number of each subject
#    
#    # get z-level value
#    z_value = 1 / z_class
#    
#    # z-level classify
    
    
def get_new_z():
    path_list = glob.glob(REG_PATH+'*.nii.gz')
    total_new_s_oh = 0
    path_list.sort()
    for s in SUBJECT_LIST:
        new_s = nib.load(path_list[s]).get_data().T
        new_s_oh = np.eye(N_CLASS)[new_s]
        total_new_s_oh += new_s_oh
    return total_new_s_oh        
#def shift_centroid(y_test, z_test, shift):
    
def get_centroid(label, width, height):
    """
    in: 
    out: centroid of on slice, shape: [n,2,n_class]
    """
    eps=1e-5
    n = label.get_shape().as_list()[0]
#    x = tf.linspace(-1.0, 1.0, width)
#    y = tf.linspace(-1.0, 1.0, height)
    x = tf.linspace(0.0, width-1, width)
    y = tf.linspace(0.0, height-1, height)
    x_t, y_t = tf.meshgrid(x, y)

    x_t = tf.reshape(x_t, [1, width, height, 1])
    y_t = tf.reshape(y_t, [1, width, height, 1])
    x_centroid = tf.reduce_sum(x_t*label, [1,2]) / (tf.reduce_sum(label, [1,2])+eps)
    y_centroid = tf.reduce_sum(y_t*label, [1,2]) / (tf.reduce_sum(label, [1,2])+eps)
    
    # round
    x_centroid = tf.cast(tf.round(x_centroid), tf.int32)
    y_centroid = tf.cast(tf.round(y_centroid), tf.int32)
    
    
    centroid = tf.concat([tf.expand_dims(x_centroid,1), tf.expand_dims(y_centroid,1)], axis=1)
    return centroid

def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="U-net multi-task")
    parser.add_argument("--data-dir", type=str, default=RAW_PATH,
                        help="...")
    parser.add_argument("--label-dir", type=str, default=MASK_PATH,
                        help="...")
    parser.add_argument("--model-dir", type=str, default=MODEL_PATH,
                        help="...")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--n-class", type=int, default=N_CLASS,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--z-class", type=int, default=Z_CLASS,
                        help="...")
    parser.add_argument("--shuffle", type=bool, default=SHUFFLE,
                        help="...")
    parser.add_argument("--crf_flag", type=bool, default=False,
                        help="...")
    return parser.parse_args()

    
def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    tf.reset_default_graph()


    # Load reader.
    data_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=args.data_dir,
                                      mask_path=args.label_dir,
                                      shuffle_data=args.shuffle,
                                      subject_list=SUBJECT_LIST,
                                      resize_ratio=0.5,
                                      data_aug=False,
                                      cubic=False,
                                      z_class=args.z_class,
                                      nx=IMG_SIZE,
                                      ny=IMG_SIZE,
                                      HU_window=HU_WINDOW,
                                      )

    # Create placeholder
    x = tf.placeholder("float", shape=[None, IMG_SIZE, IMG_SIZE, 1])
    y = tf.placeholder("float", shape=[None, IMG_SIZE, IMG_SIZE, args.n_class])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    z = tf.placeholder("int32", shape=[None, args.n_class], name='z_label')
    angle = tf.placeholder("float", shape=[None, 1], name='angle_label')
    class_label = tf.placeholder("float", shape=[None, args.n_class], name='class_label')

    label_centroid = get_centroid(y, IMG_SIZE, IMG_SIZE)
    

    # Set up tf session and initialize variables.  
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    
#    # Load weights.
#    loader = tf.train.Saver()
#    if args.model_dir is not None:
#        load_model(loader, sess, args.model_dir)

    
    num_steps = np.sum(data_provider.n_frames)
    print('num-steps: {}'.format(num_steps))
    
    all_x, all_y, all_z = [], [], []
    all_centroid, all_mean, all_shift =[], [], []
#    centroid_mean = np.zeros((args.z_class+1, 2, args.n_class))
#    for idx, n in enumerate(data_provider.n_frames):
#        total_centroid = []
#        x_test, y_test, z_test = [], [], []
#        for step in range(n):
#            if step%10:
#                print('subject: {}, slice {}/{}'.format(idx, step, n))
#            
#            image, label, z_label, angle_label, _class_label= data_provider(1)
#            x_test.append(image)
#            y_test.append(label)
#            z_test.append(z_label)
#            
#            feed_dict = {x: image, y: label, keep_prob: 1., is_training: False, z: z_label, angle: angle_label, 
#                         class_label: _class_label}
#            
#            centroid = sess.run(label_centroid, feed_dict) 
#            centroid_mean[z_label[0]] += centroid[0]
#            
#            total_centroid.append(centroid)
#    
#        centroids = np.concatenate(total_centroid,0)
##        centroid_mean[z_label[0]] += np.int32(np.round(np.mean(centroids, 0)))
##        centroid_shift = centroid_mean - centroids
#        
#        all_x.append(np.concatenate(x_test,0))
#        all_y.append(np.concatenate(y_test,0))
#        all_z.append(np.concatenate(z_test,0))
#        all_centroid.append(centroids)
#    all_mean.append(centroid_mean)
##        all_shift.append(centroid_shift)
        
#    final_output = [all_x, all_y, all_z, all_centroid, all_mean, all_shift]

    return final_output


if __name__ == '__main__':
    zz = get_new_z()
    zz = zz[:, ::-1]
#    crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
#    bisrgb = np.arange(50, 60, 10)
#    bisxy = np.arange(1, 2, 1)
#
#    idx=-1
#    
#    crf_table = np.zeros(bisrgb.shape + bisxy.shape)
#    for i in bisrgb:
#        idx += 1
#        jdx = -1
#        for j in bisxy:
#            jdx += 1
#            print(i, j)
#            crf_config['bi_srgb'] = i
#            crf_config['bi_sxy'] = j
#            x_test, y_test, z_test, all_centroid, all_mean, all_shift = main()

            

#    cmap = plt.cm.jet    
#    for i in range(len(x_test)):
##        i=i+100
##        print('sample: {}, zlabel: {}, z_prediction: {}'.format(i, int(z_test[i][0,0]), z_pred[i][0]))
#        print('sample: {}'.format(i))
#        fig, (ax11, ax12, ax13) = plt.subplots(1,3)
#
#        ax11.imshow(x_test[i][0,...,0], 'gray')
##        ax11.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax11.set_axis_off()     
#
#        ax12.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
##        ax12.imshow(total_pred[i][0,...,0], vmin=0, vmax=13)
#        ax12.set_axis_off()
#
##        ax13.imshow(y_test[i][0,...,7])
##        ax13.imshow(np.argmax(total_crf_out[i][0], -1), vmin=0, vmax=13)
#        ax13.imshow(total_pred[i][0,...,0], vmin=0, vmax=13)
#        ax13.set_axis_off()
#        plt.show()
#        plt.close(fig)

    