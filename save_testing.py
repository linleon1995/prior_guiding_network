
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
from tf_tesis2.visualize_utils import display_segmentation
from tf_tesis2.dense_crf import crf_inference
#from tf_tesis2.network import unet, unet_multi_task_fine, unet_multi_task_fine_newz, unet_prior_guide, unet_prior_guide2, unet_prior_guide_encoder, unet_prior_guide_decoder
from tf_tesis2.network import (crn_encoder_sep, crn_decoder_sep, crn_atrous_encoder_sep, crn_atrous_decoder_sep, 
                               crn_encoder_sep_com, crn_decoder_sep_com, crn_encoder_sep_resnet50, crn_decoder_sep_resnet50,
                               crn_encoder_sep_new_aggregation, crn_decoder_sep_new_aggregation, crn_encoder, crn_decoder)
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


MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_004/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_012/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_027/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_003/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_014/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_022/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_009/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_006/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_011/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_013/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_015/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_023/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_018/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_033/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_030/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_031/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_020/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_019/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_021/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_043/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_022/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_033/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_028/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_024/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_032/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_025/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_029/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_034/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_015/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_000/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_006/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_009/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_004/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_005/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_008/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_014/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_018/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_020/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_015/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_019/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_020/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_026/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_029/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_027/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_037/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_002/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_004/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_005/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_012/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_006/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_002/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_001/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_007/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_011/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_045/model.ckpt"
#MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_046/model.ckpt"

RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/testing/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
IGNORE_LABEL = 255
Z_CLASS = 3
SUBJECT_LIST = np.arange(1,5)
SHUFFLE = False
IMG_SIZE = 256
CRF_CONFIG = {"g_sxy":3,"g_compat":3,"bi_sxy":1,"bi_srgb":110,"bi_compat":3,"iterations":10}
HU_WINDOW = [-180, 250]
HU_WINDOW = [-125, 275]
CLASS_LIST = np.arange(1,14)
#CLASS_LIST = [1,2,3,6,11]
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
  
def transform_global_prior(args, label_mean, angle_label, z_label):
    # select the z-level from z label
#    label_transform = [tf.expand_dims(label_mean[z_label[i,0]],0) for i in range(1)]
#    label_transform = tf.concat(label_transform, 0)

    label_transform=label_mean
#    b=[]
#    for i in range(1):
#        a = [tf.expand_dims(label_transform[z_label[i,j],...,j],-1) for j in range(14)]
#        a = tf.concat(a, -1)
#        b.append(tf.expand_dims(a,0))
#    b = tf.concat(b, 0)
#    label_transform = b
    
#    print(z_label, label_mean)
#    thresholding = False
#    if thresholding:
#        th = 0.8
#        z_label = tf.cast(tf.greater(z_label, th), tf.float32)
#    z_label = tf.to_int32(tf.argmax(z_label, -1))
#    z_label = tf.zeros_like(z_label)
#    label_transform = tf.concat([label_mean[...,0:1],label_mean[...,6:7],label_mean[...,7:8]], axis=3)
#    label_transform = tf.concat([label_mean[...,0:1],
#                                     label_mean[...,1:2],
#                                     label_mean[...,2:3],
#                                     label_mean[...,3:4],
#                                     label_mean[...,6:7],
#                                     label_mean[...,11:12]], axis=3)
    

#    print(z_label)
#    z_label=tf.expand_dims(tf.expand_dims(z_label, axis=1), axis=2)
#    label_transform = tf.multiply(z_label, label_mean)
#    label_transform = tf.reduce_sum(label_transform, axis=-1)
    z_label = tf.to_int32(tf.argmax(z_label, -1))
    b=[]
    for i in range(1):
        a = [tf.expand_dims(label_transform[z_label[i,j],...,j],-1) for j in range(args.n_class)]
        a = tf.concat(a, -1)
        b.append(tf.expand_dims(a,0))
    b = tf.concat(b, 0)
    label_transform = b

#    # select class from class label
#    label_transform = tf.multiply(label_transform, label_exist)
        
#    rad = tf.expand_dims(angle_label * PI_ON_180, 2)
#    cos_rad = tf.cos(rad)
#    sin_rad = tf.sin(rad)
#    t1 = tf.concat([cos_rad, -sin_rad, tf.zeros_like(cos_rad)], 2)
#    t2 = tf.concat([sin_rad, cos_rad, tf.zeros_like(cos_rad)], 2)
#    theta = tf.concat([t1, t2], 1)
    
#    theta = angle_label
#    batch_grids = stn.affine_grid_generator(256, 256, theta)
#    x_s = batch_grids[:, 0, :, :]
#    y_s = batch_grids[:, 1, :, :]
#
#    background = stn.bilinear_sampler(1-label_transform[...,0:1], x_s, y_s)
#    background = 1 - background
#    foreground = stn.bilinear_sampler(label_transform[...,1:], x_s, y_s)
##    label_transform = tf.concat([foreground, background], -1)
#    label_transform = tf.concat([background, foreground], -1)
##    label_transform=    stn.bilinear_sampler(label_transform, x_s, y_s)
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
                                      z_class=None,
                                      nx=IMG_SIZE,
                                      ny=IMG_SIZE,
                                      HU_window=HU_WINDOW,
                                      mode=None,
                                      only_foreground = args.only_foreground,
                                      seq_length=args.seq_length,
                                      only_raw = False,
                                      )

    # Create placeholder
    x = tf.placeholder("float", shape=[None, IMG_SIZE, IMG_SIZE, 1])
    angle = tf.placeholder("float", shape=[None, 2, 3], name='angle_label')

    ##
#    label_top = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_top_new_z.npy')[...,np.newaxis]
#    label_mid = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_middle_new_z.npy')[...,np.newaxis]
#    label_bot = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_bottom_new_z.npy')[...,np.newaxis]
#    zero_value = np.zeros_like(label_bot)
#    ref_model = np.concatenate([zero_value, label_bot, label_mid, label_top], -1)

    label_top = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_top_new_z.npy')
    label_mid = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_middle_new_z.npy')
    label_bot = np.load('/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/label_bottom_new_z.npy')
    zero_value = np.zeros_like(label_bot)
    ref_model = np.concatenate([zero_value, label_bot, label_mid, label_top], 0)
    
    label_mean = tf.convert_to_tensor(ref_model)
    
    

    # Create encoder
#    output, pooling = unet_prior_guide_encoder(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = unet_prior_guide_prof_encoder(x, args.n_class, args.z_class, 1, is_training=False )
    output, layer_dict = crn_encoder_sep(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_atrous_encoder_sep(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder_sep_new_aggregation(x, args.n_class, args.z_class, 1, is_training=False )
#    output, layer_dict = crn_encoder_sep_resnet50(x, args.n_class, args.z_class, 1, is_training=False )    
#    output, layer_dict = crn_encoder_sep_com(x, args.n_class, args.z_class, 1, is_training=False )
    # Z output
#    z_output = z
    z_logits = output['z_output']
    z_output_a = tf.nn.softmax(tf.cast(z_logits, tf.float32))
#    z_output_a = tf.zeros_like(z_output_a)


#    z_prediction = tf.expand_dims(z_output, dim=1) # Create 4-d tensor.

    # Anggle output
#    angle_output = output['angle_output']
    
    # global prior transform
#    zz = z + tf.cast(tf.equal(z, 0), tf.int32)
    label_transform = transform_global_prior(args, label_mean, angle, z_output_a)
#    label_transform = tf.zeros_like(y)
#    label_transform = tf.ones_like(y)
#    label_transform = y
#    label_transform = label_mean
    
#    foreground = label_transform[...,1:]
#    background = label_transform[...,0:1]
    
    # Create decoder    
#    output, layer_dict = unet_prior_guide_decoder( output, label_transform, 1, pooling, is_training=False )
#    output, layer_dict, info = unet_prior_guide_prof_decoder( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep_new_aggregation( output, label_transform, args.n_class, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder( output, label_transform, args.n_class, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_atrous_decoder_sep( output, label_transform, 1, layer_dict, is_training=False )
    output, layer_dict, info = crn_decoder_sep( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep_resnet50( output, label_transform, 1, layer_dict, is_training=False )
#    output, layer_dict, info = crn_decoder_sep_com( output, label_transform, args.n_class, 1, layer_dict, is_training=False )
    logits = output['output_map']

    # Prediction       
    raw_output = tf.nn.softmax(logits)
    prediction = tf.argmax(raw_output, dimension=3)
    prediction = tf.expand_dims(prediction, dim=3) # Create 4-d tensor.
#    prediction = tf.image.resize_nearest_neighbor(prediction, [512,512], name='nearest_neighbor')
#    prediction = tf.image.resize_bilinear(prediction, [512,512], name='bilinear')        

    
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
        
    total_pred, x_test = [], []
    num_steps = data_provider.n_frames
    print('num-steps: {}'.format(num_steps))
    for subject_step in num_steps:
        subject_pred = []
        for step in range(subject_step):
            if step%20 == 0:
                print('step {}/{}'.format(step, subject_step))
            
            image = data_provider(1)
            x_test.append(image)
    
            
            feed_dict = {x: image}
            
            _pred = sess.run(prediction, feed_dict) 
            subject_pred.append(_pred[...,0])
            subject = np.concatenate(subject_pred, axis=0)
        total_pred.append(subject)  
            
    
    
    return total_pred

    
if __name__ == '__main__':
    args = get_arguments()
    crf_config = CRF_CONFIG
    total_pred = main(crf_config)
    new_pred = []
    for pred in total_pred:
        pred = pred.T[:,::-1]
        new_pred.append(pred)
        
#    # save subject for online evaluate
#    j=60
#    path = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/testing/'
#    for pred in new_pred:
#        img = nib.Nifti1Image(pred, np.eye(4))
#        nib.save(img, path+'label'+str(j).zfill(4)+'.nii.gz')
#        j+=1
        
        

    
