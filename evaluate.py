#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 16:27:56 2019

@author: EE_ACM528_04
"""

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from tf_tesis2  import unet_multi_task3, module, prior_generate
from tf_tesis2.eval_utils import (compute_mean_dsc, compute_mean_iou, compute_accuracy, load_model, 
                                plot_confusion_matrix, save_evaluation, plot_box_diagram, plot_histogram)
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

load_nibabel_data = prior_generate.load_nibabel_data


MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_017/model.ckpt"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/tesis_trained/run_022/model.ckpt"

RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
MODEL_FLAG = {'zlevel_classify': True, 'rotate_module': True, 'class_classify': True, 'crf': False}
MASK_SUBJECT_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw/label/'
OUTPUT_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/eval/'
IGNORE_LABEL = 255
HEIGHT = 256
WIDTH = 256
Z_CLASS = 60
SUBJECT_LIST = np.arange(25,30)
SHUFFLE = False
IMG_SIZE = 256
HU_WINDOW = [-125, 275]
CLASS_LIST = np.arange(1,14)
#CLASS_LIST = [6]
N_CLASS=len(CLASS_LIST)+1
N_OBSERVE_SUBJECT = 5
OBSERVE_SUBJECT_LIST = [25,26]



parser = argparse.ArgumentParser()

parser.add_argument('--raw-path', type=str, default=RAW_PATH,
                    help='')

parser.add_argument('--mask-path', type=str, default=MASK_PATH,
                    help='')

parser.add_argument('--mask-subject-path', type=str, default=MASK_SUBJECT_PATH,
                    help='')

parser.add_argument('--output_path', type=str, default=OUTPUT_PATH,
                    help='')

parser.add_argument('--lambda-z', type=float, default=1e-1,
                    help='')

parser.add_argument('--lambda-guidance', type=float, default=1e-1,
                    help='')

parser.add_argument('--subject_for_prior', type=str, default='label0001.nii.gz',
                    help='')

parser.add_argument("--model-dir", type=str, default=MODEL_PATH,
                    help="...")

parser.add_argument("--n-class", type=int, default=N_CLASS,
                    help="Number of classes to predict (including background).")

parser.add_argument("--z-class", type=int, default=Z_CLASS,
                    help="...")

parser.add_argument("--shuffle", type=bool, default=SHUFFLE,
                    help="...")

parser.add_argument("--display_flag", type=bool, default=False,
                    help="...")

parser.add_argument('--only_foreground', type=bool, default=False,
                help='')

parser.add_argument('--seq-length', type=int, default=None,
                help='')

parser.add_argument("--save-eval", type=str, default=True,
                    help="...")

parser.add_argument("--max-observe-subject", type=int, default=5,
                    help="...")               


def transform_global_prior(args, prior, z_pred):
    prior = tf.reshape(prior, [args.z_class, -1])
    label_transform = tf.matmul(z_pred, prior)
    label_transform = tf.reshape(label_transform, [1, 512,512,args.n_class])
#    label_transform = tf.expand_dims(prior[z_pred[0]], axis=0)      
    return label_transform    
 

def main():
    """Create the model and start the evaluation process."""
    tf.reset_default_graph()
    # Check correctness of observe_subject_list

    # Create data provider.
    data_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=FLAGS.raw_path,
                                      mask_path=FLAGS.mask_path,
                                      shuffle_data=FLAGS.shuffle,
                                      subject_list=SUBJECT_LIST,
                                      class_list=CLASS_LIST,
                                      resize_ratio=0.5,
                                      data_aug=False,
                                      cubic=False,
                                      z_class=FLAGS.z_class,
                                      nx=WIDTH,
                                      ny=HEIGHT,
                                      HU_window=HU_WINDOW,
                                      mode=None,
                                      only_foreground = FLAGS.only_foreground,
                                      seq_length=FLAGS.seq_length
                                      )

    # Restore model from checkpoint
    ref_model = load_nibabel_data(FLAGS.mask_subject_path, processing_list=np.arange(1))[0]
    net = unet_multi_task3.Unet(model_flag=MODEL_FLAG,
                                nx=HEIGHT,
                                ny=WIDTH,
                                channels=data_provider.channels, 
                                n_class=data_provider.n_class, 
                                cost="mean_dice_coefficient", 
                                norm=True,
                                pretrained=None,
                                summaries=False,
                                z_class = data_provider.z_class,
                                prior = ref_model,
                                batch_size = 1,
                                lambda_z = FLAGS.lambda_z,
                                lambda_guidance = FLAGS.lambda_guidance,
                                seq_length=FLAGS.seq_length,
#                                data_aug='resize_and_crop'
                                )
    
    # output
    output = net.output
    logits = output['output_map']
    raw_output = tf.nn.softmax(logits)
    prediction = tf.argmax(raw_output, dimension=3)
    z_pred = output['z_output']

    
    z_label = tf.one_hot(indices=net.z_label,
                         depth=int(FLAGS.z_class),
                         on_value=1,
                         off_value=0,
                         axis=-1,
                         )
    z_label = tf.cast(z_label, tf.float32)

    # TODO: 
#    if FLAGS.seq_length is not None:
#        conv_output = tf.split(layer_dict['pool4'], FLAGS.seq_length, axis=0)
#        with tf.variable_scope("RNN"):
#            rnn_output = module.bidirectional_GRU(features=conv_output,
#                                              batch_size=1,
#                                              nx=WIDTH//8,
#                                              ny=HEIGHT//8,
#                                              n_class=FLAGS.n_class,
#                                              seq_length=FLAGS.seq_length,
#                                              is_training=False)
#
##                self.logits = conv2d(rnn_output, [1,1,32,self.n_class], activate=None, scope="logits", bn_flag=False)
##                self.logits = rnn_output
#        layer_dict['pool4'] = rnn_output
            
    # confusion matrix for segmentation result
    label = tf.argmax(net.y, -1)
    cm = tf.confusion_matrix(tf.reshape(label, [-1,]), tf.reshape(prediction, [-1,]), num_classes=FLAGS.n_class)

    # Set up tf session and initialize variables.  
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    
    # Load weights.
    loader = tf.train.Saver()
    if FLAGS.model_dir is not None:
        load_model(loader, sess, FLAGS.model_dir)
    else:
        raise ValueError("model checkpoint not exist")

    # TODO: 
    # Iterate over training steps.
    trainable_vars = {}
    for v in  tf.trainable_variables():
        trainable_vars[v.name] = v

    cm_total = 0
    DSC_slice = []
    total_image, total_label, total_pred = {}, {}, {}
    for slice_idx in data_provider.data_files:
        print("subject: {}, slice: {}".format(*slice_idx))
        image, label, z_label, _, _= data_provider(1)
        feed_dict = {net.x: image, net.y: label, net.keep_prob: 1., net.is_training: False, net.z_label: z_label}
        slice_key = "s" + str(slice_idx[0]) + "f" + str(slice_idx[1])

        # DSC for each slice
        cm_slice, pred = sess.run([cm, prediction], feed_dict)
        cm_total += cm_slice
        DSC_slice.append(compute_mean_dsc(cm_slice))

        # visualization
        if FLAGS.display_flag:
            display_segmentation(image, label, pred, FLAGS.n_class)

        # save image
        if FLAGS.save_eval and len(OBSERVE_SUBJECT_LIST) < FLAGS.max_observe_subject:
            save_evaluation()

        # packing list for return segmentation   specific subject
        if len(OBSERVE_SUBJECT_LIST) < N_OBSERVE_SUBJECT:
            total_image[slice_key] = image
            total_label[slice_key] = label
            if slice_idx[0] in OBSERVE_SUBJECT_LIST:
                total_pred[slice_key] = pred

    # plot confusion_matrix        
    plot_confusion_matrix(cm_total, classes=np.arange(FLAGS.n_class), normalize=True,
                          title='Confusion matrix, without normalization')
    plt.show()
    
        
    mIoU_total = compute_mean_iou(cm_total)
    DSC_total = compute_mean_dsc(cm_total)          
    Acc_total = compute_accuracy(cm_total)

    if FLAGS.save_eval:
        # plot histogram of DSC
        plot_histogram(path=FLAGS.output_path)    

        # plot box digram
        plot_box_diagram(path=FLAGS.output_path)      
    
    if len(OBSERVE_SUBJECT_LIST) < FLAGS.max_observe_subject:
        return DSC_slice, mIoU_total, DSC_total, Acc_total, total_pred, total_image, total_label
    else:
        return DSC_slice, mIoU_total, DSC_total, Acc_total

# def plot_func(x):
#     for i in range(32):
#         print(i)
#         fig, (ax1) = plt.subplots(1,1)
#         ax1.imshow(x[...,i])
#         ax1.set_axis_off()
#         plt.show()

# def display_guidance(total_guid, n_class):
#     for i in range(n_class):
#         print('class: {}'.format(i))
#         fig, (ax1,ax2,ax3) = plt.subplots(1,3)
#         ax1.imshow(total_guid[130]['guidance_in'][0,...,i])
#         ax2.imshow(total_guid[130]['guidance1'][0,...,i])
#         ax3.imshow(total_guid[130]['guidance2'][0,...,i])
#         plt.show()

# def show_two_images(master, slave, cmap=plt.cm.jet, p=0.5):
#     if np.max(slave)!=255 or np.min(slave)!=0:
#         slave = (slave-np.min(slave)) / (np.max(slave)-np.min(slave))
#     p_slave = cmap(slave)
#     p_slave[:,:,3]=p
#     fig, ax = plt.subplots()
#     ax.imshow(master, 'gray')
#     ax.imshow(p_slave)

# def np_iou(a, b):
#     intersection = np.sum(a*b)
#     union = np.sum(a) + np.sum(b) - intersection
#     if union == 0:
#         return 1
#     else:
#         return intersection / union


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    eval = main()
    # total_mIoU, total_acc, total_DSC, sum_cm, z_acc = eval
                    
    # if FLAGS.display_flag:         
    #     display_segmentation(x_test, y_test, total_pred, args.n_class)


    