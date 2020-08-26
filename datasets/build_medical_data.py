#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:03 2020

@author: EE_ACM528_04
"""

import collections
import six
import tensorflow as tf
import nibabel as nib
import numpy as np
import argparse
import SimpleITK as sitk

# TODO: tensorflow 1.4 API doesn't support tf.app.flags.DEFINE_enume, apply this after update tensorflow version
# FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_enum('image_format', 'nii.gz', ['nii.gz', 'dcm'],
#                          'Image format.')

# tf.app.flags.DEFINE_enum('label_format', 'nii.gz', ['nii.gz', 'dcm'],
#                          'Segmentation label format.')

parser = argparse.ArgumentParser()

parser.add_argument('--image_format', type=str, default='nii.gz',
                    help='Image format.')

parser.add_argument('--label_format', type=str, default='nii.gz',
                    help='Segmentation label format.')                    

FLAGS, unparsed = parser.parse_known_args()

# A map from image format to expected data format.
_IMAGE_FORMAT_MAP = {
    'nii.gz': 'nii.gz',
    'dcm': 'dcm'
}

# TODO: Return one reader for reading all 3D data under given path
class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='nii.gz', channels=1):
    """Class constructor.
    Args:
      image_format: Image format. Only 'jpeg', 'jpg', or 'png' are supported.
      channels: Image channels.
    """
    self._image_format = image_format


  def decode_image(self, image_path):
      """
      """
      # Read the medical image (.nii.gz .dcm) containing the volume with SimpleITK
      image = sitk.ReadImage(image_path)
      
      # Access the numpy array:
      image_arr = sitk.GetArrayFromImage(image)
      
      self._decode = np.int32(image_arr)
      
      # if self._image_format in ('.nii.gz', 'nii.gz', '.nii', 'nii'):
          
      #     
          
      # elif self._image_format in ('dcm', '.dcm'):
      #     import matplotlib.pyplot as plt
      #     pass
      #     image_arr = sitk.GetArrayFromImage(image)
      #     self._decode = np.int32(image_arr)
      #     plt.imshow(self._decode[0])
      #     plt.show()
      return self._decode


  def read_image_dims(self, image):
      """
      """
      if len(image.shape) == 3:
          depth, height, width = image.shape
      elif len(image.shape) == 2:
          height, width = image.shape
          depth = 1
      return (height, width, depth)


def _int64_list_feature(values):
  """Returns a TF-Feature of int64_list.
  Args:
    values: A scalar or list of values.
  Returns:
    A TF-Feature.
  """
  if not isinstance(values, collections.Iterable):
    values = [values]

  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _bytes_list_feature(values):
  """Returns a TF-Feature of bytes.
  Args:
    values: A string.
  Returns:
    A TF-Feature.
  """
  def norm2bytes(value):
    return value.encode() if isinstance(value, str) and six.PY3 else value

  return tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[norm2bytes(values)]))


def image_seg_to_tfexample(image_data, filename, height, width, depth, num_slices, seg_data=None, organ_label=None):
  """Converts one image/segmentation pair to tf example.
  Args:
    image_data: string of image data.
    filename: image filename.
    height: image height.
    width: image width.
    seg_data: string of semantic segmentation data.
  Returns:
    tf example of one image/segmentation pair.
  """
  feature={
      'image/encoded': _bytes_list_feature(image_data),
      'image/filename': _bytes_list_feature(filename),
      'image/format': _bytes_list_feature(_IMAGE_FORMAT_MAP[FLAGS.image_format]),
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/depth': _int64_list_feature(depth),
      'image/num_slices': _int64_list_feature(num_slices),
      # 'image/segmentation/class/format': _bytes_list_feature(FLAGS.label_format),
  }
  if seg_data is not None:
    feature['segmentation/encoded'] = _bytes_list_feature(seg_data)
    
  if organ_label is not None:
    feature['image/segmentation/class/organ_label'] = _bytes_list_feature(organ_label)
  
  return tf.train.Example(features=tf.train.Features(feature=feature))
  
# def priors_to_tfexample(priors, parse_name, num_slices, prior_id):
#   """Converts one image/segmentation pair to tf example.
#   Args:
#     image_data: string of image data.
#     filename: image filename.
#     height: image height.
#     width: image width.
#     seg_data: string of semantic segmentation data.
#   Returns:
#     tf example of one image/segmentation pair.
#   """
#   return tf.train.Example(features=tf.train.Features(feature={
#       'prior/num_slices': _int64_list_feature(num_slices),
#       'prior/prior_id': _int64_list_feature(prior_id),
#       'prior/'+parse_name: _bytes_list_feature(priors)
#   }))