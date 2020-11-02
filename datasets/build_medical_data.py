#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image loader variants.
Code branched out from https://github.com/tensorflow/models/tree/master/research/deeplab
, and please refer to it for more details.
"""

import collections
import six
import tensorflow as tf
import nibabel as nib
import numpy as np
import SimpleITK as sitk


class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self, image_format='nii.gz', channels=1):
    """Class constructor.
    Args:
      image_format: Image format.
      channels: Image channels.
    """
    self._image_format = image_format

  def decode_image(self, image_path):
      # Read the medical image (.nii.gz .dcm) containing the volume with SimpleITK
      image = sitk.ReadImage(image_path)

      # Access the numpy array:
      image_arr = sitk.GetArrayFromImage(image)

      self._decode = np.int32(image_arr)
      return self._decode

  def read_image_dims(self, image):
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


def image_seg_to_tfexample(image_data, height, width, depth, num_slices, image_format=None, seg_data=None):
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
      'image/height': _int64_list_feature(height),
      'image/width': _int64_list_feature(width),
      'image/depth': _int64_list_feature(depth),
      'image/num_slices': _int64_list_feature(num_slices),
  }
  if seg_data is not None:
    feature['segmentation/encoded'] = _bytes_list_feature(seg_data)

  if image_format is not None:
    feature['image/format'] = _bytes_list_feature(image_format)

  return tf.train.Example(features=tf.train.Features(feature=feature))
