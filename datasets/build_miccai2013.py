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


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/home/acm528_02/Jing_Siang/data/Synpase_raw/',
                    help='MICCAI 2013 dataset root folder.')

parser.add_argument('--output_dir', type=str, default='/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord',
                    help='Path to save converted SSTable of TensorFlow examples.')


parser.add_argument('--dataset_split', type=str, default=None,
                    help='')

# parser.add_argument('--num_shard', type=int, default=None,
#                     help='')

# parser.add_argument('--num_samples', type=int, default=None,
#                     help='')

# TODO: manage multiple integers
parser.add_argument('--split_indices', type=int, default=None,
                    help='')

parser.add_argument('--extract_fg_exist_slice', type=bool, default=False,
                    help='')


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
# _DATA_SPLIT = 25/30

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])


def _get_files(data, data_dir, dataset_split, split_indices=None):
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

  filenames = file_utils.get_file_list(
    data_dir+_FOLDERS_MAP[data]+"/", fileStr=[dataset_split], fileExt=["nii.gz"], sort_files=True)

  if split_indices is not None:
    # TODO: do it correctly
    if split_indices[1] > len(filenames):
      raise ValueError("Out of Range")

    filenames = filenames[split_indices[0]:split_indices[1]]

  # if data == 'label' and dataset_split == 'test':
  #   return None

  return filenames


def _convert_dataset(dataset_split, data_dir, output_dir, extract_fg_exist_slice, split_indices=None):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files('image', data_dir, _POSTFIX_MAP["image"], split_indices)
  if dataset_split in ("train", "val"):
    label_files = _get_files('label', data_dir, _POSTFIX_MAP["label"], split_indices)

  num_images = len(image_files)

  image_reader = build_medical_data.ImageReader('nii.gz', channels=1)
  if dataset_split in ("train", "val"):
    label_reader = build_medical_data.ImageReader('nii.gz', channels=1)

  for shard_id in range(num_images):
    if extract_fg_exist_slice:
      shard_filename = '%s-%s-%05d-of-%05d.tfrecord' % (
        dataset_split, "fg", shard_id, num_images)
    else:
      shard_filename = '%s-%05d-of-%05d.tfrecord' % (
        dataset_split, shard_id, num_images)
    output_filename = os.path.join(output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:

      # Read the image.
      image_data = image_reader.decode_image(image_files[shard_id])
      image_data = image_data[:,::-1]
      height, width, num_slices = image_reader.read_image_dims(image_data)

      if extract_fg_exist_slice:
        image_type = "image (foreground)"
      else:
        image_type = "image"

      sys.stdout.write('\n>> [{}] Converting {} {}/{} shard {} in num_frame {} and size[{},{}]'.format(
        dataset_split, image_type, shard_id+1, num_images, shard_id+1, len(image_files), height, width))

      # sys.stdout.flush()
      if dataset_split in ("train", "val"):
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

      # TODO: organ label
      # if dataset_split in ("train", "val"):
      #   seg_onehot = np.eye(NUM_CLASS)[seg_data]
      #   organ_labels = np.sum(np.sum(seg_onehot, 1), 1)
      #   organ_labels = np.int32(organ_labels>0)
      for i in range(num_slices):
        if extract_fg_exist_slice:
          cond = np.sum(seg_data[i])
        else:
          cond = True
        if cond:
          image_slice = image_data[i].tostring()
          if dataset_split in ("train", "val"):
            seg_slice = seg_data[i].tostring()
            # organ_label = organ_labels[i].tostring()
            example = build_medical_data.image_seg_to_tfexample(image_slice, filename, height, width, depth=i,
                                                                num_slices=num_slices, seg_data=seg_slice)
          elif dataset_split == "test":
            example = build_medical_data.image_seg_to_tfexample(image_slice, filename, height, width, depth=i,
                                                                num_slices=num_slices)

          tfrecord_writer.write(example.SerializeToString())




  sys.stdout.write('\n')
  sys.stdout.flush()


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  # for dataset_split in ['train', 'val']:
  # data_dir = "/home/user/DISK/data/Jing/data/Training/"
  # output_dir = "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/Trian_Sets/tfrecord/"
  # dataset_split = {
  #                  "train": [0,24],
  #                  "val": [24,30],
  #                  "test": None
  #                  }
  # for extract_fg_exist_slice in [True, False]:
  #   for split, indices in dataset_split.items():
  #     _convert_dataset(split, data_dir, output_dir, extract_fg_exist_slice, indices)

  data_dir = "/home/user/DISK/data/Jing/data/Testing/"
  output_dir = "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/tfrecord/img/Test_Sets/"
  _convert_dataset("test", data_dir, output_dir, False, None)

if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)