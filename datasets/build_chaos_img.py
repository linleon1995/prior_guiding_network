#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020
@author: Jing-Siang, Lin
"""

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import build_medical_data
import file_utils


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='2019 ISBI CHAOS dataset root folder.')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to save converted SSTable of TensorFlow examples.')

parser.add_argument('--split_indices', nargs='+', type=int,
                    help="")


MR_LABEL_CONVERT = {63: 1, 126: 2, 189: 3, 252: 4}
# A map from data split to folder name that saves the data.
_SPLIT_MAP = {
    'train': 'Train_Sets',
    'val': 'Train_Sets',
    'test': 'Test_Sets',
}

# A map from data modality to folder name that saves the data.
_MODALITY_MAP = {
    'CT': ['CT'],
    'MR_T1_In': ["MR", "T1DUAL", "InPhase"],
    'MR_T1_Out': ["MR", "T1DUAL", "OutPhase"],
    'MR_T2': ["MR", "T2SPIR"],
}

# A map from data type to folder name that saves the data.
_FOLDERS_MAP = {
    'image': 'DICOM_anon',
    'label': 'Ground',
}

# A map from data type to filename postfix.
_POSTFIX_MAP = {
    "CT": {'image': ['IMG', "i"],'label': ['liver']},
    "MR_T1_In": {'image': ['IMG'],'label': ['IMG']},
    "MR_T1_Out": {'image': ['IMG'],'label': ['IMG']},
    "MR_T2": {'image': ['IMG'],'label': ['IMG']}
}

# A map from data type to data format.
_DATA_FORMAT_MAP = {
    'image': 'dcm',
    'label': 'png',
}


def _get_files(data_path, modality, img_or_label):
  """Gets files for the specified data type and dataset split.
  Args:
    data: String, desired data ('image' or 'label').
    dataset_split: String, dataset split ('train', 'val', 'test')
  Returns:
    A list of sorted file names or None when getting label for
      test set.
  """
  if "CT" in modality:
    subject_path = os.path.join(data_path, _FOLDERS_MAP[img_or_label])
  elif "MR" in modality:
    subject_path = os.path.join(data_path, _MODALITY_MAP[modality][1], _FOLDERS_MAP[img_or_label])
    if "MR_T1" in modality and  _FOLDERS_MAP[img_or_label]==_FOLDERS_MAP["image"]:
      subject_path = os.path.join(subject_path, _MODALITY_MAP[modality][2])
  else:
    raise ValueError("Unknown data modality")

  filenames = file_utils.get_file_list(subject_path,
                                       fileStr=_POSTFIX_MAP[modality][img_or_label],
                                       fileExt=[_DATA_FORMAT_MAP[img_or_label]],
                                       sort_files=True)
  return filenames


def _convert_single_subject(output_filename, modality, image_files, label_files=None):
  """write one subject in one tfrecord sample
  """
  with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    if label_files is not None:
      assert len(image_files) == len(label_files)
      label_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["image"], channels=1)

    image_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["label"], channels=1)
    for i in range(len(image_files)):
      # Read the image.
      image_data = image_reader.decode_image(image_files[i])
      image_data = image_data[0]

      image_slice = image_data.tostring()
      height, width = np.shape(image_data)

      num_slices = len(image_files)

      # Read the semantic segmentation annotation.
      example_kwargs = {}
      if label_files is not None:
        seg_data = label_reader.decode_image(label_files[i])

        if "MR" in modality:
          seg_data = file_utils.convert_label_value(seg_data, MR_LABEL_CONVERT)
        elif "CT" in modality:
          seg_data = file_utils.convert_label_value(seg_data, {255: 1})

        seg_slice = seg_data.tostring()
        example_kwargs = {"seg_data": seg_slice}

      filename = image_files[i]

      example = build_medical_data.image_seg_to_tfexample(
          image_slice, filename, height, width, depth=i, num_slices=num_slices, **example_kwargs)
      tfrecord_writer.write(example.SerializeToString())

  return height, width


def _convert_dataset(data_dir, out_dir, dataset_split, modality, train_split_indices=None):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """

  data_path = os.path.join(data_dir, _SPLIT_MAP[dataset_split], _MODALITY_MAP[modality][0])

  folder_for_each_subject = os.listdir(data_path)
  folder_for_each_subject.sort()
  if train_split_indices is not None:
    if max(train_split_indices) > len(folder_for_each_subject):
      raise ValueError("Out of Range")
   
    if dataset_split == "train":
      split_indices = train_split_indices
    elif dataset_split == "val":
      split_indices = list(set(range(len(folder_for_each_subject)))-set(train_split_indices))
    folder_for_each_subject = [folder_for_each_subject[idx] for idx in split_indices]

  num_shard = len(folder_for_each_subject)

  # make dir if not exist
  if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

  total_slices = 0
  for shard_id, sub_folder in enumerate(folder_for_each_subject):
    path = os.path.join(data_path, sub_folder)

    image_files = _get_files(path, modality, img_or_label="image")
    kwargs={"image_files": image_files}
    if dataset_split in ("train", "val"):
      label_files = _get_files(path, modality, img_or_label="label")
      kwargs["label_files"] = label_files

    total_slices += len(image_files)

    shard_filename = '%s-%s-%05d-of-%05d.tfrecord' % (
          dataset_split, modality, shard_id, num_shard)

    output_filename = os.path.join(out_dir, shard_filename)
    height, width = _convert_single_subject(output_filename, modality, **kwargs)

    sys.stdout.write('\n>> [{}:{}] Converting image {}/{} shard {} in num_frame {} and size[{},{}]'.format(
        dataset_split, modality, shard_id+1, num_shard, shard_id+1, len(image_files), height, width))

  sys.stdout.write('\n' + 60*"-")
  sys.stdout.write('\n total_slices: {}'.format(total_slices))
  sys.stdout.write('\n')
  sys.stdout.flush()


def main(unused_argv):
  assert FLAGS.split_indices[1] > FLAGS.split_indices[0]
  dataset_split = {
                  "train": list(range(*FLAGS.split_indices)),
                  "val": list(range(*FLAGS.split_indices)),
                  "test": None
                  }
  for m in ["CT", "MR_T2", "MR_T1_In", "MR_T1_Out"]:
    for split in dataset_split:
      modality_for_output = m
      if "MR_T1" in modality_for_output:
        modality_for_output = "MR_T1"

      out_dir = os.path.join(FLAGS.data_dir, FLAGS.output_dir, "img", _SPLIT_MAP[split], modality_for_output)
      _convert_dataset(FLAGS.data_dir, out_dir, split, m, dataset_split[split])


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)