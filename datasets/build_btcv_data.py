#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020

@author: Jing-Siang, Lin
"""


import os
import re
import sys
import argparse
import numpy as np
import tensorflow as tf
import build_medical_data
import file_utils
str2bool = file_utils.str2bool

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='2015 MICCAI BTCV dataset root folder.')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to save converted TensorFlow examples.')

parser.add_argument('--split_indices', nargs='+', type=int,
                    help="Indices to for the training set and validation set splitting")

parser.add_argument('--extract_fg_exist_slice', type=str2bool,
                    help='Extract the slice including foreground')

parser.add_argument('--seq_length', type=int, required=True,
                    help='The slice number in single sequence.')

# A map from data split to folder name that saves the data.
_SPLIT_MAP = {
    'train': 'Train_Sets',
    'val': 'Train_Sets',
    'test': 'Test_Sets',
}

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


def _get_files(data, data_dir, dataset_split, train_split_indices=None):
  """Gets files for the specified data type and dataset split.
  """
  filenames = file_utils.get_file_list(
    os.path.join(data_dir, _SPLIT_MAP[dataset_split], _FOLDERS_MAP[data]), 
    fileStr=[_POSTFIX_MAP[data]], fileExt=["nii.gz"], sort_files=True)
  
  if train_split_indices is not None:
    if max(train_split_indices) > len(filenames):
      raise ValueError("Out of Range")
    
    if dataset_split == "train":
      split_indices = train_split_indices
    elif dataset_split == "val":
      split_indices = list(set(range(len(filenames)))-set(train_split_indices))
    filenames = [filenames[idx] for idx in split_indices]
  return filenames


def _convert_dataset(dataset_split, data_dir, seq_length, output_dir, extract_fg_exist_slice, split_indices=None):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """
  image_files = _get_files("image", data_dir, dataset_split, split_indices)
  image_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["image"], channels=1)
  
  if dataset_split in ("train", "val"):
    label_files = _get_files("label", data_dir, dataset_split, split_indices)
    label_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["label"], channels=1)
  
  if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
  if extract_fg_exist_slice:
    image_type = "image (foreground)"
  else:
    image_type = "image"
    
  # Convert each subject to single tfrecord example  
  num_images = len(image_files)
  for shard_id in range(num_images):
    if seq_length > 1:
        data_tag = "seq"
    else:
        data_tag = "img"
    if extract_fg_exist_slice:
      shard_filename = '%s-%s-%s-%05d-of-%05d.tfrecord' % (dataset_split, data_tag, "fg", shard_id, num_images)
    else:
      shard_filename = '%s-%s-%05d-of-%05d.tfrecord' % (dataset_split, data_tag, shard_id, num_images)
    output_filename = os.path.join(output_dir, shard_filename)
    with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
      # Read the image.
      image_data = image_reader.decode_image(image_files[shard_id])
      image_data = image_data[:,::-1]
      height, width, num_slices = image_reader.read_image_dims(image_data)
      
      # Read the semantic segmentation annotation.
      if dataset_split in ("train", "val"):
        seg_data = label_reader.decode_image(label_files[shard_id])
        seg_data = seg_data[:,::-1]
        seg_height, seg_width, _ = label_reader.read_image_dims(seg_data)
        if height != seg_height or width != seg_width:
          raise RuntimeError('Shape mismatched between image and label.')
       
      for i in range(num_slices):
        if extract_fg_exist_slice:
          process = np.sum(seg_data[i])
        else:
          process = True
          
        if process:
          if seq_length > 1:
            img_format = _DATA_FORMAT_MAP["image"].encode('ascii')
            if dataset_split in ("train", "val"):
              sequence = img_seg_to_sequence(
                image_data, height, width, num_slices, i, seq_length, img_format, seg_data)
            elif dataset_split == "test":
              sequence = img_seg_to_sequence(
                image_data, height, width, num_slices, i, seq_length, img_format)
            tfrecord_writer.write(sequence.SerializeToString())
          else:
            image_slice = image_data[i].tostring()
            if dataset_split in ("train", "val"):
              seg_slice = seg_data[i].tostring()
            elif dataset_split == "test":
              seg_slice = None
            example = build_medical_data.image_seg_to_tfexample(
              image_slice, height, width, depth=i, num_slices=num_slices, seg_data=seg_slice)
            tfrecord_writer.write(example.SerializeToString())
          
      sys.stdout.write('\n>> [{}] Converting {} {}/{} shard {} in num_frame {} sequence length {} and size[{},{}]'.format(
        dataset_split, image_type, shard_id+1, num_images, shard_id+1, num_slices, seq_length, height, width))

  sys.stdout.write('\n')
  sys.stdout.flush()


def img_seg_to_sequence(
  image_data, height, width, num_slices, slice_idx, seq_length=None, img_format=None, seg_data=None):
    sequence = tf.train.SequenceExample()
    context = sequence.context.feature
    features = sequence.feature_lists.feature_list
    
    # Aceess slices for each sequence sample
    start = slice_idx - seq_length // 2
    for j in range(start, start+seq_length):
        if j < 0:
            sequence_idx = 0
        elif j > num_slices-1:
            sequence_idx = num_slices-1
        else:
            sequence_idx = j
        image_slice = image_data[sequence_idx].tostring()
        image_encoded = features['image/encoded'].feature.add()
        image_encoded.bytes_list.value.append(image_slice)
        if seg_data is not None:
            seg_slice = seg_data[sequence_idx].tostring()
            segmentation_encoded = features['segmentation/encoded'].feature.add()
            segmentation_encoded.bytes_list.value.append(seg_slice)
        depth_encoded = features['image/depth'].feature.add()
        depth_encoded.int64_list.value.append(sequence_idx)
        
    context['image/height'].int64_list.value.append(height)
    context['image/width'].int64_list.value.append(width)
    context['image/num_slices'].int64_list.value.append(num_slices)
    if img_format is not None:
        context['image/format'].bytes_list.value.append(img_format)
    return sequence
    
    
def main(unused_argv):
  assert FLAGS.split_indices[1] > FLAGS.split_indices[0]
  dataset_split = {"train": list(range(*FLAGS.split_indices)),
                   "val": list(range(*FLAGS.split_indices)),
                   "test": None}
  
  for split, indices in dataset_split.items():
    out_dir = os.path.join(FLAGS.data_dir, FLAGS.output_dir, "img", _SPLIT_MAP[split])
    _convert_dataset(
      split, FLAGS.data_dir, FLAGS.seq_length, out_dir, FLAGS.extract_fg_exist_slice, indices)
      
  
if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)