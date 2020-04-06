#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020

@author: EE_ACM528_04
"""

import glob
import math
import os.path
import re
import sys
import argparse
import numpy as np
import tensorflow as tf

import build_medical_data
 
# TODO: tensorflow 1.4 API doesn't support tf.app.flags.DEFINE_enume, apply this after update tensorflow version
# FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('miccai_2013',
#                            '/home/acm528_02/Jing_Siang/data/Synpase_raw/',
#                            'MICCAI 2013 dataset root folder.')

# tf.app.flags.DEFINE_string(
#     'output_dir',
#     '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord',
#     'Path to save converted SSTable of TensorFlow examples.')

parser = argparse.ArgumentParser()

parser.add_argument('--miccai-2013', type=str, default='/home/acm528_02/Jing_Siang/data/Synpase_raw/',
                    help='MICCAI 2013 dataset root folder.')

parser.add_argument('--output-dir', type=str, default='/home/acm528_02/Jing_Siang/data/Synpase_raw/vm_data/tfrecord',
                    help='Path to save converted SSTable of TensorFlow examples.')                    

parser.add_argument('--prior_id', type=int, default=0,
                    help='')  

# TODO: maybe save the whole voxel and sample in the data_generator code
parser.add_argument('--num_samples', type=int, default=None,
                    help='')  
                    
_NUM_SHARDS = 25
_NUM_SLICES = 3779
_NUM_VOXELS = 30
_DATA_TYPE = "2D"
PRIOR_IMGS = 'prior_imgs'
PRIOR_SEGS = 'prior_segs'

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'raw',
    'label': 'label',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    'image': 'img',
    'label': 'label',
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'nii.gz',
    'label': 'nii.gz',
}

# TODO: describe
_DATA_SPLIT = 25/30

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])


def _get_files(data, dataset_split):
  """Gets files for the specified data type and dataset split.
  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train', 'val', 'test')
  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """
  if data == 'label' and dataset_split == 'test':
    return None
  pattern = '%s*.%s' % (_POSTFIX_MAP[data], _DATA_FORMAT_MAP[data])
  # TODO: separate in image
  # TODO: description
  # TODO: dataset converting and prior converting should be separate, otherwise prior converting will be executed twice
  search_files = os.path.join(
      FLAGS.miccai_2013, _FOLDERS_MAP[data], pattern)
  filenames = glob.glob(search_files)
  filenames.sort()
  split_idx = int(len(filenames)*_DATA_SPLIT)
  if dataset_split == 'train':
      filenames = filenames[0:3]
  elif dataset_split == 'val':
      filenames = filenames[3:4]
  return filenames


def _convert_dataset(dataset_split):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', dataset_split)
  label_files = _get_files('label', dataset_split)


  num_images = len(image_files)
  num_shard = num_images

  image_reader = build_medical_data.ImageReader('nii.gz', channels=1)
  label_reader = build_medical_data.ImageReader('nii.gz', channels=1)

  for shard_id in range(num_shard):
    shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, _NUM_SHARDS)
    output_filename = os.path.join(FLAGS.output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
            shard_id+1, num_images, shard_id+1))
      # sys.stdout.flush()
      # Read the image.
      image_data = image_reader.decode_image(image_files[shard_id])
      height, width, num_slices = image_reader.read_image_dims(image_data)
      # Read the semantic segmentation annotation.
      seg_data = label_reader.decode_image(label_files[shard_id])
      seg_height, seg_width, _ = label_reader.read_image_dims(seg_data)
      if height != seg_height or width != seg_width:
        raise RuntimeError('Shape mismatched between image and label.')
      # Convert to tf example.
      # TODO: re_match?
      re_match = _IMAGE_FILENAME_RE.search(image_files[shard_id])
      if re_match is None:
        raise RuntimeError('Invalid image filename: ' + image_files[shard_id])
      filename = os.path.basename(re_match.group(1))

      if _DATA_TYPE == "2D":
        for i in range(0, num_slices, 5):
          label_list = list(range(0, num_slices, 5))
          label_list.pop(label_list.index(i))
          for j in label_list:
            
            image_slice = image_data[i].tostring()
            label_slice = seg_data[j].tostring()
            example = build_medical_data.image_seg_to_tfexample(
                image_slice, label_slice, filename, height, width, depth=i, num_slices=num_slices)
            tfrecord_writer.write(example.SerializeToString())
      elif _DATA_TYPE == "3D":
        image_slice = image_data.tostring()
        seg_slice = seg_data.tostring()
        example = build_medical_data.image_seg_to_tfexample(
            image_slice, seg_slice, filename, height, width, depth=None, num_slices=num_slices)
        tfrecord_writer.write(example.SerializeToString())
      
      
      
  sys.stdout.write('\n')
  sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  for dataset_split in ['train', 'val']:
    _convert_dataset(dataset_split)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)