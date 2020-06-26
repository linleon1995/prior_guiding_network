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

import build_medical_data, file_utils

 
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

parser.add_argument('--output-dir', type=str, default='/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord',
                    help='Path to save converted SSTable of TensorFlow examples.')                    

parser.add_argument('--prior_id', type=int, default=0,
                    help='')  

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
  # TODO: description
  # TODO: dataset converting and prior converting should be separate, otherwise prior converting will be executed twice
 
  if data == 'label' and dataset_split == 'test':
    return None
  
  filenames = file_utils.get_file_list(FLAGS.miccai_2013+_FOLDERS_MAP[data]+"/", fileExt=["nii.gz"], 
                                       sort_files=True)
  if dataset_split[1] > len(filenames):
    raise ValueError("Unknown split name")
  filenames = filenames[dataset_split[0]:dataset_split[1]]
  
  # if dataset_split in ('train', 'val'):
  #   split_idx = int(len(filenames)*_DATA_SPLIT)
  #   if dataset_split == 'train':
  #       filenames = filenames[:split_idx]
  #   elif dataset_split == 'val':
  #       filenames = filenames[split_idx:]
  # elif dataset_split == 'test':
  #   pass
  # else:
  #   raise ValueError("Unknown split name")
  return filenames


def _convert_dataset(dataset_split, split_indices):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', split_indices)
  label_files = _get_files('label', split_indices)

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
      image_data = image_data[:,::-1]
      height, width, num_slices = image_reader.read_image_dims(image_data)
      # Read the semantic segmentation annotation.
      seg_data = label_reader.decode_image(label_files[shard_id])
      seg_data = seg_data[:,::-1]
      seg_height, seg_width, _ = label_reader.read_image_dims(seg_data)
      if height != seg_height or width != seg_width:
        raise RuntimeError('Shape mismatched between image and label.')
      # Convert to tf example.
      # TODO: re_match?
      re_match = _IMAGE_FILENAME_RE.search(image_files[shard_id])
      if re_match is None:
        raise RuntimeError('Invalid image filename: ' + image_files[shard_id])
      filename = os.path.basename(re_match.group(1))

      # TODO: 14, organ label
      if _DATA_TYPE == "2D":
        seg_onehot = np.eye(14)[seg_data]
        organ_labels = np.sum(np.sum(seg_onehot, 1), 1)
        organ_labels = np.int32(organ_labels>0)
        for i in range(num_slices):
          image_slice = image_data[i].tostring()
          seg_slice = seg_data[i].tostring()
          organ_label = organ_labels[i].tostring()
          example = build_medical_data.image_seg_to_tfexample(image_slice, seg_slice, 
                                                              filename, height, width, depth=i, 
                                                              num_slices=num_slices, organ_label=organ_label)
          tfrecord_writer.write(example.SerializeToString())
      elif _DATA_TYPE == "3D":
        pass
        # image_slice = image_data.tostring()
        # seg_slice = seg_data.tostring()
        # organ_label = np.int32(np.nonzero(np.sum(seg_slice)))
        # example = build_medical_data.image_seg_to_tfexample(
        #     image_slice, seg_slice, filename, height, width, depth=None, num_slices=num_slices, organ_label=organ_label)
        # tfrecord_writer.write(example.SerializeToString())
      
      
    # if FLAGS.prior_id is not None:  
    #   if shard_id == FLAGS.prior_id:
    #     prior_imgs = image_data
    #     prior_segs = seg_data
        
    #     # Assign one training data (voxel) as a prior
    #     if prior_imgs is not None:
    #       prior_filename = '%s-%05d.tfrecord' % (
    #           PRIOR_IMGS, FLAGS.prior_id)
    #       output_filename = os.path.join(FLAGS.output_dir, prior_filename)
    #       with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    #         sys.stdout.write('\r>> Converting image prior (prior_id %d)' % (FLAGS.prior_id))
    #         sys.stdout.flush()
            
    #         if FLAGS.num_samples is not None:
    #           indices = np.arange(0, num_slices, num_slices/FLAGS.num_samples)
    #           indices = np.int32(indices)
    #           prior_imgs = np.take(prior_imgs, indices, axis=0)
    #           new_num_slices = FLAGS.num_samples
    #         else:
    #           new_num_slices =  num_slices
              
    #         example = build_medical_data.priors_to_tfexample(
    #           prior_imgs.tostring(), PRIOR_IMGS, num_slices=new_num_slices, prior_id=FLAGS.prior_id)
    #         tfrecord_writer.write(example.SerializeToString())
            
    #     if prior_segs is not None:
    #       prior_filename = '%s-%05d.tfrecord' % (
    #           PRIOR_SEGS, FLAGS.prior_id)
    #       output_filename = os.path.join(FLAGS.output_dir, prior_filename)
    #       with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    #         sys.stdout.write('\r>> Converting segmentatin prior (prior_id %d)' % (FLAGS.prior_id))
    #         sys.stdout.flush()
            
    #         if FLAGS.num_samples is not None:
    #           indices = np.arange(0, num_slices, num_slices/FLAGS.num_samples)
    #           indices = np.int32(indices)
    #           prior_segs = np.take(prior_segs, indices, axis=0)
    #           new_num_slices = FLAGS.num_samples
    #         else:
    #           new_num_slices =  num_slices
              
    #         example = build_medical_data.priors_to_tfexample(
    #           prior_segs.tostring(), PRIOR_SEGS, num_slices=new_num_slices, prior_id=FLAGS.prior_id)
    #         tfrecord_writer.write(example.SerializeToString())
      
  sys.stdout.write('\n')
  sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  # for dataset_split in ['train', 'val']:
  dataset_split = {"train": [0,15],
                   "val": [15,20]}
  for split in dataset_split.items():
    _convert_dataset(*split)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)