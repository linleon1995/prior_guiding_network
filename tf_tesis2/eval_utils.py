#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 12:05:16 2019

@author: acm528_02
"""
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

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