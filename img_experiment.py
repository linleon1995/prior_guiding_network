#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 21:55:28 2019

@author: acm528_02
"""


import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tf_unet_multi_task  import unet_multi_task2
#from tf_unet_multi_task  import util_multi_task 
import numpy as np
import CT_scan_util_multi_task
import matplotlib.pyplot as plt
#import tensorflow as tf
#from tf_unet_multi_task import unet_multi_task 
#from tf_unet import unet_from_lab_server
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from data_augmentation import random_rotation, horizontal_flip, vertical_flip, scale_augmentation
import nibabel as nib
import glob

import argparse

N_CLASS = 14
RAW_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/raw/'
MASK_PATH = '/home/acm528_02/Jing_Siang/data/Synpase_raw_frames/label/'
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_unet/unet_trained_CT_scan/run_new/unet32_pre_sub25_bilinear/run_115/model.ckpt-35"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_unet_multi_task/unet_mt_trained/run_001/model.ckpt-35"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_unet_multi_task/unet_mt_trained/unet32_sub25_gmp/run_000/model.ckpt-35"
MODEL_PATH = "/home/acm528_02/Jing_Siang/project/Tensorflow/tf_unet_multi_task/unet_mt_trained/unet32_sub25_z_attention/run_004/model.ckpt-85"
DATA_LIST_PATH = './dataset/val.txt'
IGNORE_LABEL = 255
#NUM_CLASSES = 14
NUM_STEPS = 1449 # Number of images in the validation set.
Z_CLASS = 3
SUBJECT_LIST = np.arange(3)
SHUFFLE = False
IMG_SHAPE = [1, 256, 256, 1]
LABEL_SHAPE = [1, 256, 256, N_CLASS]





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

    return parser.parse_args()
    

#def model_prepare(x, args):
#    logits, z_logits, variables = unet_multi_task.create_conv_net_upsample_multi_task(x, 
#                                                                                      keep_prob=1, 
#                                                                                      is_training=False, 
#                                                                                      channels=1, 
#                                                                                      n_class=args.n_class, 
#                                                                                      layers=5, 
#                                                                                      features_root=32,  
#                                                                                      summaries=False, 
#                                                                                      z_class=args.z_class
#                                                                                      )
#    return logits, z_logits


def model_prepare(x, z, args):
    logits, z_logits, _, classmap = unet_multi_task2.create_conv_net_upsample_multi_task_GMP(x, 
                                                                                      z_label=z,                       
                                                                                      keep_prob=1, 
                                                                                      is_training=False, 
                                                                                      channels=1, 
                                                                                      n_class=args.n_class, 
                                                                                      layers=5, 
                                                                                      features_root=32,  
                                                                                      summaries=False, 
                                                                                      z_class=args.z_class
                                                                                      )
    return logits, z_logits, classmap


def compute_mean_dsc(total_cm):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = np.sum(total_cm, axis=0).astype(float)
      sum_over_col = np.sum(total_cm, axis=1).astype(float)
      cm_diag = np.diagonal(total_cm).astype(float)
      denominator = sum_over_row + sum_over_col
    
      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = np.sum((denominator != 0).astype(float))
    
      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = np.where(
          denominator > 0,
          denominator,
          np.ones_like(denominator))
    
      dscs = 2*cm_diag / denominator
    
      print('Dice Score Simililarity for each class:')
      for i, dsc in enumerate(dscs):
        print('    class {}: {:.4f}'.format(i, dsc))
    
      # If the number of valid entries is 0 (no classes) we return 0.
      m_dsc = np.where(
          num_valid_entries > 0,
          np.sum(dscs) / num_valid_entries,
          0)
      m_dsc = float(m_dsc)
      print('mean Dice Score Simililarity: {:.4f}'.format(float(m_dsc)))
      return m_dsc
  
    
def compute_mean_iou(total_cm):
      """Compute the mean intersection-over-union via the confusion matrix."""
      sum_over_row = np.sum(total_cm, axis=0).astype(float)
      sum_over_col = np.sum(total_cm, axis=1).astype(float)
      cm_diag = np.diagonal(total_cm).astype(float)
      denominator = sum_over_row + sum_over_col - cm_diag
    
      # The mean is only computed over classes that appear in the
      # label or prediction tensor. If the denominator is 0, we need to
      # ignore the class.
      num_valid_entries = np.sum((denominator != 0).astype(float))
    
      # If the value of the denominator is 0, set it to 1 to avoid
      # zero division.
      denominator = np.where(
          denominator > 0,
          denominator,
          np.ones_like(denominator))
    
      ious = cm_diag / denominator
    
      print('Intersection over Union for each class:')
      for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))
    
      # If the number of valid entries is 0 (no classes) we return 0.
      m_iou = np.where(
          num_valid_entries > 0,
          np.sum(ious) / num_valid_entries,
          0)
      m_iou = float(m_iou)
      print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
      return m_iou


def compute_accuracy(total_cm):
      """Compute the accuracy via the confusion matrix."""
      denominator = total_cm.sum().astype(float)
      cm_diag_sum = np.diagonal(total_cm).sum().astype(float)

      # If the number of valid entries is 0 (no classes) we return 0.
      accuracy = np.where(
          denominator > 0,
          cm_diag_sum / denominator,
          0)
      accuracy = float(accuracy)
      print('Pixel Accuracy: {:.4f}'.format(float(accuracy)))
      return accuracy
    
    
def load_model(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()    
    
    
#def get_classmap(label, conv6, gmp_w):
#        conv6_resized = tf.image.resize_bilinear( conv6, [256, 256] )
##        with tf.variable_scope("GMP", reuse=True):
#        label_w = tf.gather(tf.transpose(gmp_w), label)
#        label_w = tf.reshape( label_w, [-1, 512, 1] ) # [batch_size, 1024, 1]
#
#        conv6_resized = tf.reshape(conv6_resized, [-1, 256*256, 512]) # [batch_size, 224*224, 1024]
#
#        classmap = tf.matmul( conv6_resized, label_w )
#        classmap = tf.reshape( classmap, [-1, 256, 256] )
#        return classmap    

    
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
                                      cubic=True,
                                      z_class=args.z_class,
                                      )
#    data_provider.frames_number()
#    image, label, z_label = data_provider(32)
#    x, y = data_provider(frames_number[idx])
    
    # Create network.

    # Prediction
    x = tf.placeholder("float", shape=[None, 256, 256, 1])
    y = tf.placeholder("float", shape=[None, 256, 256, args.n_class])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool)
    z = tf.placeholder("int32", shape=[None, 1], name='z_label')
    
    logits, z_logits, classmap = model_prepare(x, z, args)
    
    
    # Prediction       
    raw_output = tf.nn.softmax(logits)
    raw_output = tf.argmax(raw_output, dimension=3)
    prediction = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    z_output = tf.nn.softmax(z_logits)
    z_output = tf.argmax(z_output, dimension=1)
    z_prediction = tf.expand_dims(z_output, dim=1) # Create 4-d tensor.


    # mIoU
    pred = tf.reshape(prediction, [-1,])
    z_pred = tf.reshape(z_prediction, [-1,])
    gt = tf.argmax(y, -1)
    gt = tf.reshape(gt, [-1,])
#    z_gt = tf.argmax(z, -1)
    z_gt = tf.reshape(z, [-1,])
    
    cm = tf.confusion_matrix(gt, pred, num_classes=args.n_class)
    cm_z = tf.confusion_matrix(z_gt, z_pred, num_classes=args.z_class)

    # class_map
#    classmap = get_classmap(z_pred, conv_final, gmp_w)
    
    
    # Set up tf session and initialize variables.  
    sess = tf.Session()
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    
#    # Load weights.
    loader = tf.train.Saver()
    if args.model_dir is not None:
        load_model(loader, sess, args.model_dir)


    # Iterate over training steps.
    total_mIoU, total_acc, total_pred, total_pred_z, total_logits, x_test, y_test, z_test = [], [], [], [], [], [], [], []
    slice_DSC = []
    total_gmp = []
    sum_cm = 0
    sum_cm_z = 0
    num_steps = np.sum(data_provider.n_frames)
    print('num-steps: {}'.format(num_steps))
    
    for step in range(num_steps):
        image, label, z_label = data_provider(1)
        x_test.append(image)
        y_test.append(label)
        z_test.append(z_label)
        for angle in range(0, 360, 10):
            image = random_rotation(image, angle, type='img', dtype=np.float32)
            label = random_rotation(label, angle, type='label', dtype=np.uint8)
            
            feed_dict = {x: image, y: label, keep_prob: 1., is_training: False, z: z_label}
            _logits, _pred, _z_pred, np_cm, np_cm_z, _classmap = sess.run([logits, prediction, z_pred, cm, cm_z, classmap], feed_dict) 
    
            sum_cm += np_cm
            sum_cm_z += np_cm_z
#        total_pred.append(_pred)
#        total_logits.append(_logits)
#        total_pred_z.append(_z_pred)  
#        total_gmp.append(_classmap)
#
#            print('step {:d}'.format(step))
#            slice_DSC.append(compute_mean_iou(np_cm))
    plot_confusion_matrix(sum_cm, classes=np.arange(args.n_class), normalize=True,
                          title='Confusion matrix, without normalization')
    plt.show()
    total_mIoU = compute_mean_iou(sum_cm)
    total_DSC = compute_mean_dsc(sum_cm)          
    total_acc = compute_accuracy(sum_cm)
    
    plot_confusion_matrix(sum_cm_z, classes=np.arange(args.z_class), normalize=True,
                          title='Confusion matrix, without normalization')
    plt.show()
    z_acc = compute_accuracy(sum_cm_z)
    
    return x_test, y_test, z_test, total_pred, total_pred_z, total_logits, total_mIoU, total_acc, total_DSC, sum_cm, slice_DSC, sum_cm_z, z_acc, total_gmp


if __name__ == '__main__':
    x_test, y_test, z_test, prediction, z_pred, logits, total_mToU, total_acc, total_DSC, sum_cm, slice_DSC, sum_cm_z, z_acc, total_gmp = main()

#    cmap = plt.cm.jet    
#    for i in range(len(x_test)):
#        print('sample: {}, zlabel: {}, z_prediction: {}'.format(i, int(z_test[i][0,0]), z_pred[i][0]))
#        fig, (ax11, ax12, ax13) = plt.subplots(1,3)
#        cam = total_gmp[i][0,...,0]
#        cam = np.uint8(255*((cam-np.min(cam))/(np.max(cam)-np.min(cam))))
#        cam = cmap(cam)
#        
#        ax11.imshow(x_test[i][0,...,0], 'gray')
#        cam_tranpency = np.copy(cam)
#        cam_tranpency[...,-1] = 0.4
#        ax11.imshow(cam_tranpency)
#        ax11.set_axis_off()     
#
##        ax12.imshow(cam)
##        ax12.set_axis_off()
#        ax12.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax12.set_axis_off()
#
#        ax13.imshow(prediction[i][0,...,0], vmin=0, vmax=13)
#        ax13.set_axis_off()
#        plt.show()
        
#        ax13.imshow(np.argmax(y_test[i][0], -1), vmin=0, vmax=13)
#        ax13.set_axis_off()
#        plt.show()
          