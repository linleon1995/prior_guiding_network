#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:27:56 2019

@author: EE_ACM528_04
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tf_tesis2  import unet_multi_task3, module
from tf_tesis2.eval_utils import compute_mean_dsc, compute_mean_iou, compute_accuracy, load_model, plot_confusion_matrix
from tf_tesis2.visualize_utils import display_segmentation
from tf_tesis2.dense_crf import crf_inference
#from tf_tesis2.network import unet, unet_multi_task_fine, unet_multi_task_fine_newz, unet_prior_guide, unet_prior_guide2, unet_prior_guide_encoder, unet_prior_guide_decoder
from tf_tesis2.network import (crn_encoder_sep, crn_decoder_sep, crn_atrous_encoder_sep, crn_atrous_decoder_sep, 
                               crn_encoder_sep_com, crn_decoder_sep_com, crn_encoder_sep_resnet50, crn_decoder_sep_resnet50,
                               crn_encoder_sep_new_aggregation, crn_decoder_sep_new_aggregation, crn_encoder, crn_decoder, crn_atrous_decoder_sep2)

import numpy as np
import CT_scan_util_multi_task
import matplotlib.pyplot as plt

from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import nibabel as nib
import glob
from tf_tesis2 import stn
PI_ON_180 = 0.017453292519943295
import argparse


MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_001/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_005/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_007/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_010/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_011/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_006/model.ckpt"

RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
IGNORE_LABEL = 255
Z_CLASS = 60
SUBJECT_LIST = np.arange(25,30)
SHUFFLE = False
IMG_SIZE = 256
CRF_CONFIG = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
HU_WINDOW = [-180, 250]
HU_WINDOW = [-125, 275]
CLASS_LIST = np.arange(1,14)
#CLASS_LIST = [6]
N_CLASS=len(CLASS_LIST)+1

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
    parser.add_argument("--display_flag", type=bool, default=False,
                        help="...")
    parser.add_argument("--get-info", type=bool, default=False,
                        help="...")
    parser.add_argument('--only_foreground', type=bool, default=False,
                    help='')
    parser.add_argument('--seq-length', type=int, default=None,
                    help='')
    return parser.parse_args()
  

def transform_global_prior(args, label_mean, z_pred):
    label_mean = tf.reshape(label_mean, [args.z_class, -1])
    label_transform = tf.matmul(z_pred, label_mean)
    label_transform = tf.reshape(label_transform, [1, 512,512,args.n_class])
#    label_transform = tf.expand_dims(label_mean[z_pred[0]], axis=0)      
    return label_transform    
 
    
    
def main(crf_config):
    """Create the model and start the evaluation process."""
    args = get_arguments()
    tf.reset_default_graph()


    # Load reader.
    data_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=args.data_dir,
                                      mask_path=args.label_dir,
                                      shuffle_data=args.shuffle,
                                      subject_list=SUBJECT_LIST,
                                      class_list=CLASS_LIST,
                                      resize_ratio=0.5,
                                      data_aug=False,
                                      cubic=False,
                                      z_class=args.z_class,
                                      nx=IMG_SIZE,
                                      ny=IMG_SIZE,
                                      HU_window=HU_WINDOW,
                                      mode=None,
                                      only_foreground = args.only_foreground,
                                      seq_length=args.seq_length
                                      )

    # Create placeholder
    x = tf.placeholder("float", shape=[None, IMG_SIZE, IMG_SIZE, 1])
    y = tf.placeholder("float", shape=[None, IMG_SIZE, IMG_SIZE, args.n_class])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    z = tf.placeholder("int32", shape=[None, 1], name='z_label')
#    z = tf.placeholder("int32", shape=[None, args.n_class], name='z_label')
#    z_a = tf.one_hot(indices=z,
#                    depth=int(args.z_class+1),
#                    on_value=1,
#                    off_value=0,
#                    axis=-1,
#                    )
    z_a = tf.one_hot(indices=z,
                    depth=int(args.z_class),
                    on_value=1,
                    off_value=0,
                    axis=-1,
                    )
    z_a = tf.cast(z_a, tf.float32)
    angle = tf.placeholder("float", shape=[None, 2, 3], name='angle_label')
    class_label = tf.placeholder("float", shape=[None, args.n_class], name='class_label')

    label_exist = tf.reshape(class_label, [1, 1, 1, args.n_class])
#    label_top = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_top_new_z.npy')
#    label_mid = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_middle_new_z.npy')
#    label_bot = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_bottom_new_z.npy')
#    zero_value = np.zeros_like(label_bot)
#    ref_model = np.concatenate([zero_value, label_bot, label_mid, label_top], 0)
    
    p='/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/prior/prior_in_60/'
#    p='/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/prior/prior_in_20/'
    ref_model = np.concatenate([np.load(p+'num_shard_0000.npy'),np.load(p+'num_shard_0001.npy'),np.load(p+'num_shard_0002.npy'),
                                np.load(p+'num_shard_0003.npy'),np.load(p+'num_shard_0004.npy')], 0)
    ref_model = np.float32(ref_model)
    
    label_mean = tf.convert_to_tensor(ref_model)
    
    ##
#    gg = [v for v in tf.global_variables() if 'conv_r2' in v.name]
#    for v in tf.global_variables():
#        print(v.name)
    g4=tf.image.resize_bilinear(y, [128,128])
    g3=tf.image.resize_bilinear(y, [64,64])
    g2=tf.image.resize_bilinear(y, [32,32])
    g1=tf.image.resize_bilinear(y, [16,16])
    gg=[g4,g3,g2,g1]
    
    # Create encoder
#    output, pooling = unet_prior_guide_encoder(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = unet_prior_guide_prof_encoder(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder_sep(x, args.n_class, args.z_class, 1, is_training=False )
    if args.seq_length is not None:
        batch_size = args.seq_length
    output, layer_dict = crn_atrous_encoder_sep(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder_sep_new_aggregation(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder_sep_resnet50(x, args.n_class, args.z_class, 1, is_training=False )    
#    output, layer_dict = crn_encoder_sep_com(x, args.n_class, args.z_class, 1, is_training=False )
    # Z output
#    z_output = z
    z_logits = output['z_output']
    z_output = tf.nn.softmax(z_logits)
    z_pred = tf.to_int32(tf.argmax(z_output, -1))
    
    # Anggle output
#    angle_output = output['angle_output']
    
    # global prior transform
#    zz = z + tf.cast(tf.equal(z, 0), tf.int32)
    label_transform = transform_global_prior(args, label_mean, z_output)
    label_transform = y

    if args.seq_length is not None:
        conv_output = tf.split(layer_dict['pool4'], args.seq_length, axis=0)
        with tf.variable_scope("RNN"):
            rnn_output = module.bidirectional_GRU(features=conv_output,
                                              batch_size=1,
                                              nx=256//8,
                                              ny=256//8,
                                              n_class=args.n_class,
                                              seq_length=args.seq_length,
                                              is_training=False)

#                self.logits = conv2d(rnn_output, [1,1,32,self.n_class], activate=None, scope="logits", bn_flag=False)
#                self.logits = rnn_output
        layer_dict['pool4'] = rnn_output
            
    # Create decoder    
#    output, layer_dict = unet_prior_guide_decoder( output, label_transform, 1, pooling, is_training=False )
#    output, layer_dict, info = unet_prior_guide_prof_decoder( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep_new_aggregation( output, label_transform, args.n_class, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder( output, label_transform, args.n_class, 1, layer_dict, is_training=False )
    output, layer_dict, info = crn_atrous_decoder_sep2( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep_resnet50( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep_com( output, label_transform, args.n_class, 1, layer_dict, is_training=False )
    logits = output['output_map']
    
#    layer_dict['guidance1'] = tf.nn.softmax(layer_dict['guidance1'])
#    layer_dict['guidance2'] = tf.nn.softmax(layer_dict['guidance2'])
#    layer_dict['guidance3'] = tf.nn.softmax(layer_dict['guidance3'])
#    layer_dict['guidance4'] = tf.nn.softmax(layer_dict['guidance4'])
    # Prediction       
    raw_output = tf.nn.softmax(logits)
    prediction = tf.argmax(raw_output, dimension=3)
    prediction = tf.expand_dims(prediction, dim=3) # Create 4-d tensor.
    
    p_for_show = colorize(tf.expand_dims(tf.argmax(y[0:1], 3), 3), cmap='viridis')
    
    # CRF
    if args.crf_flag:
        x_r = 255.0*tf.tile(x, [1,1,1,3])
        crf_output = build_crf(x_r, raw_output, args.n_class, crf_config)
        crf_output_b = tf.exp(crf_output)
        crf_output = tf.argmax(crf_output_b, dimension=3)
        crf_output = tf.expand_dims(crf_output, dim=3)

    # confusion matrix
    gt = tf.argmax(y, -1)
    gt = tf.reshape(gt, [-1,])
    z_gt = tf.reshape(z, [-1,])

    cm = tf.confusion_matrix(gt, tf.reshape(prediction, [-1,]), num_classes=args.n_class)
    cm_z = tf.confusion_matrix(z_gt, tf.reshape(z_pred, [-1,]), num_classes=args.z_class)
#    cm_z = tf.confusion_matrix(z_gt, tf.reshape(z_pred, [-1,]), num_classes=args.z_class+1)
    if args.crf_flag: cm_crf = tf.confusion_matrix(gt, tf.reshape(crf_output, [-1,]), num_classes=args.n_class)

    # angle error
#    angle_output = tf.reshape(angle_output, (1, 2, 3))
#    angle_square_err = tf.reduce_sum(tf.square(tf.subtract(angle, angle_output)))
#    angle_square_err = tf.squeeze(tf.square(tf.subtract(angle, angle_output)), axis=1)
    

    # Set up tf session and initialize variables.  
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    
    # Load weights.
    loader = tf.train.Saver()
    if args.model_dir is not None:
        load_model(loader, sess, args.model_dir)


    # Iterate over training steps.
    trainable_vars = {}
    for v in  tf.trainable_variables():
        trainable_vars[v.name] = v
        
    total_pred, total_pred_z, total_logits, x_test, y_test, z_test = [], [], [], [], [], []
    total_co = []
    total_prior = []
    slice_DSC = []
    total_crf_out = []
    tpp=[]
    total_guid = []
    total_info = []
    sum_cm = 0
    sum_cm_z = 0
    sum_cm_crf = 0
    total_angle_err = 0
    num_steps = len(data_provider._find_data_files())
    print('num-steps: {}'.format(num_steps))
    train_vars = 0
    if num_steps > 300:
        assert not args.get_info
    for step in range(num_steps):
        print('step {:d}'.format(step))
        
        image, label, z_label, angle_label, _class_label= data_provider(1)
        
        
        feed_dict = {x: image, y: label, keep_prob: 1., is_training: False, z: z_label, angle: angle_label, 
                     class_label: _class_label}
        
        if args.get_info:
            layer_dicts =  \
            sess.run(layer_dict, feed_dict)
            total_guid.append(layer_dicts)
            if step == 130 or step == 255:
                info_test = sess.run([info,gg], feed_dict)
                total_info.append(info_test)
            train_vars = sess.run(trainable_vars)
        _logits, _pred, _z_pred, np_cm, np_cm_z, prior, pp =  \
        sess.run([logits, prediction, z_output, cm, cm_z, label_transform, p_for_show], feed_dict) 

#        if args.get_info:
#            x_test.append(image)
#            y_test.append(label)
#            z_test.append(z_label)
#            total_logits.append(_logits)
#            total_pred.append(_pred)
#            total_pred_z.append(_z_pred)
#            tpp.append(pp)
#            total_prior.append(prior)
        sum_cm += np_cm
        sum_cm_z += np_cm_z
        
        slice_DSC.append(compute_mean_dsc(np_cm))

        if args.crf_flag:
            crf_out, np_cm_crf = sess.run([crf_output, cm_crf], feed_dict) 
            sum_cm_crf += np_cm_crf
            total_crf_out.append(crf_out)
          
            
    plot_confusion_matrix(sum_cm, classes=np.arange(args.n_class), normalize=True,
                          title='Confusion matrix, without normalization')
    plt.show()
    plot_confusion_matrix(sum_cm_z, classes=np.arange(args.z_class), normalize=True,
                          title='Confusion matrix, without normalization')
#    plot_confusion_matrix(sum_cm_z, classes=np.arange(args.z_class+1), normalize=True,
#                          title='Confusion matrix, without normalization')
    plt.show()

        
    total_mIoU = compute_mean_iou(sum_cm)
    total_DSC = compute_mean_dsc(sum_cm)          
    total_acc = compute_accuracy(sum_cm)
    
    z_acc = compute_accuracy(sum_cm_z)
#    total_angle_err = total_angle_err / num_steps
#    print('Angle MSE: {}'.format(total_angle_err))
    
    # final output
#    if args.get_info:
#        final_output = [x_test, y_test, z_test, total_pred, total_pred_z, total_logits, total_mIoU, total_acc, \
#                        total_DSC, sum_cm, label_mean, total_prior, z_acc, total_angle_err, tpp, total_guid, 
#                        total_info, sum_cm_z, train_vars]
#    else:
    final_output = [total_mIoU, total_acc, total_DSC, sum_cm, z_acc]
    
    if args.crf_flag: 
        plot_confusion_matrix(sum_cm_crf, classes=np.arange(args.n_class), normalize=True,
                              title='Confusion matrix, without normalization')
        plt.show()
        total_DSC_crf = compute_mean_dsc(sum_cm_crf) 
        final_output.append(total_DSC_crf)
        final_output.append(total_crf_out)
    
    return final_output

def plot_func(x):
    for i in range(32):
        print(i)
        fig, (ax1) = plt.subplots(1,1)
        ax1.imshow(x[...,i])
        ax1.set_axis_off()
        plt.show()

def display_guidance(total_guid, n_class):
    for i in range(n_class):
        print('class: {}'.format(i))
        fig, (ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(total_guid[130]['guidance_in'][0,...,i])
        ax2.imshow(total_guid[130]['guidance1'][0,...,i])
        ax3.imshow(total_guid[130]['guidance2'][0,...,i])
        plt.show()

def show_two_images(master, slave, cmap=plt.cm.jet, p=0.5):
    if np.max(slave)!=255 or np.min(slave)!=0:
        slave = (slave-np.min(slave)) / (np.max(slave)-np.min(slave))
    p_slave = cmap(slave)
    p_slave[:,:,3]=p
    fig, ax = plt.subplots()
    ax.imshow(master, 'gray')
    ax.imshow(p_slave)

def np_iou(a, b):
    intersection = np.sum(a*b)
    union = np.sum(a) + np.sum(b) - intersection
    if union == 0:
        return 1
    else:
        return intersection / union


if __name__ == '__main__':
    args = get_arguments()
    crf_config = CRF_CONFIG
    

    if args.crf_flag:
        bisrgb = np.arange(10, 50, 10)
        bisxy = np.arange(1e-5, 0.1, 0.02)
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
                
                eval = main(crf_config)
                x_test, y_test, z_test, total_pred, total_pred_z, total_logits, total_mIoU, total_acc, \
                total_DSC, sum_cm, label_mean, total_prior, z_acc, total_angle_err, total_DSC_crf, total_crf_out, sum_cm_z = eval
                crf_table[idx,jdx] = total_DSC_crf 
                
    else:
        eval = main(crf_config)
#        x_test, y_test, z_test, total_pred, total_pred_z, total_logits, total_mIoU, total_acc, \
#        total_DSC, sum_cm, label_mean, total_prior, z_acc, total_angle_err, tpp, \
#        total_guid, total_info, sum_cm_z, train_vars = eval
        total_mIoU, total_acc, total_DSC, sum_cm, z_acc = eval
                    
    if args.display_flag:         
        display_segmentation(x_test, y_test, total_pred, args.n_class)


    