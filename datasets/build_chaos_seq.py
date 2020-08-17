#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020

@author: EE_ACM528_04
"""

import glob
import math
import os
import re
import sys
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import build_medical_data, file_utils
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MR_LABEL_CONVERT = {63: 1, 126: 2, 189: 3, 252: 4}
# TODO: tensorflow 1.4 API doesn't support tf.app.flags.DEFINE_enume, apply this after update tensorflow version
# TODO: neccary condition for some varaibles
# TODO: build folder
# FLAGS = tf.app.flags.FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-name', type=str, default="2019_ISBI_CHAOS",
                    help='')

parser.add_argument('--data-dir', type=str, default='/home/acm528_02/Jing_Siang/data/2019_ISBI_CHAOS/',
                    help='MICCAI 2013 dataset root folder.')

parser.add_argument('--output-dir', type=str, default='/home/acm528_02/Jing_Siang/data/2019_ISBI_CHAOS/tfrecord/',
                    help='Path to save converted SSTable of TensorFlow examples.')

parser.add_argument('--dataset-split', type=str, default="train",
                    help='')

parser.add_argument('--dataset-modality', type=str, default="CT",
                    help='')

parser.add_argument('--num-shard', type=int, default=None,
                    help='')

parser.add_argument('--num_samples', type=int, default=None,
                    help='')

# TODO: should manage multiple integers
parser.add_argument('--split-indices', type=int, default=None,
                    help='')

parser.add_argument('--seq_length', type=int, default=3,
                    help='')

# TODO:
_NUM_SLICES = 3779
_NUM_VOXELS = 30
_DATA_TYPE = "2D"

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

# Image file pattern.
# TODO:
# _IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP["CT"]['image'])


def _get_files(data_path, modality, img_or_label):
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


def _convert_single_subject_to_seq(dataset_name, output_filename, modality, seq_length, image_files, label_files=None):
  """write one subject in one tfrecord sample
  """
  with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
    if label_files is not None:
      assert len(image_files) == len(label_files)
      # TODO: if ITK could accept all format, just remove the reader format option
      label_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["image"], channels=1)

    image_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["label"], channels=1)
    for i in range(len(image_files)):
        sequence = tf.train.SequenceExample()
        context = sequence.context.feature
        features = sequence.feature_lists.feature_list
        
        num_slices = len(image_files)
        start = i - seq_length // 2
        for j in range(start, start+seq_length):
            if j < 0:
                slice_idx = 0
            elif j > num_slices-1:
                slice_idx = num_slices-1
            else:
                slice_idx = j

            # Read the image.
            image_data = image_reader.decode_image(image_files[slice_idx])
            image_data = image_data[0]
            image_slice = image_data.tostring()
            # print(np.shape(image_data))
            height, width = np.shape(image_data)
            
            # Read the semantic segmentation annotation.
            if label_files is not None:
                seg_data = label_reader.decode_image(label_files[slice_idx])
                if "MR" in modality:
                    seg_data = file_utils.convert_label_value(seg_data, MR_LABEL_CONVERT)
                elif "CT" in modality:
                    seg_data = file_utils.convert_label_value(seg_data, {255: 1})
                    # seg_data = seg_data // 255
                    
                seg_slice = seg_data.tostring()
            filename = image_files[slice_idx]

                
            image_encoded = features['image/encoded'].feature.add()
            image_encoded.bytes_list.value.append(image_slice)
            if label_files is not None:
                segmentation_encoded = features['segmentation/encoded'].feature.add()
                segmentation_encoded.bytes_list.value.append(seg_slice)
            depth_encoded = features['image/depth'].feature.add()
            depth_encoded.int64_list.value.append(slice_idx)
                
            
        context['dataset/name'].bytes_list.value.append(dataset_name.encode('ascii'))
        context['dataset/num_frames'].int64_list.value.append(num_slices)
        context['image/format'].bytes_list.value.append(_DATA_FORMAT_MAP["image"].encode('ascii'))
        context['image/channels'].int64_list.value.append(1)
        context['image/height'].int64_list.value.append(height)
        context['image/width'].int64_list.value.append(width)
    
        tfrecord_writer.write(sequence.SerializeToString())

    #   # Read the semantic segmentation annotation.
    #   example_kwargs = {}
    #   if label_files is not None:
    #     seg_data = label_reader.decode_image(label_files[i])
        
    #     if "MR" in modality:
    #       seg_data = file_utils.convert_label_value(seg_data, MR_LABEL_CONVERT)
    #       # if i %10 == 0:
    #       #   print(np.shape(seg_data))
    #       #   plt.imshow(seg_data)
    #       #   plt.show()
    #     elif "CT" in modality:
    #       seg_data = file_utils.convert_label_value(seg_data, {255: 1})
    #     # seg_data = seg_data // 255
        
    #     seg_slice = seg_data.tostring()
        
    #     # seg_onehot = np.eye(N_CLASS)[seg_data]
    #     # organ_labels = np.sum(np.sum(seg_onehot, 1), 1)
    #     example_kwargs = {"seg_data": seg_slice,
    #                       # "organ_label": organ_label
    #                       }
        
    #     # print(np.min(seg_data), np.max(seg_data))
    #   # # TODO: re_match?
    #   # re_match = _IMAGE_FILENAME_RE.search(image_files[i])
    #   # if re_match is None:
    #   #   raise RuntimeError('Invalid image filename: ' + image_files[i])
    #   # filename = os.path.basename(re_match.group(1))
    #   filename = image_files[i]

    #   example = build_medical_data.image_seg_to_tfexample(
    #       image_slice, filename, height, width, depth=i, num_slices=num_slices, **example_kwargs)
    #   tfrecord_writer.write(example.SerializeToString())

  return height, width

def _convert_dataset(dataset_name, out_dir, dataset_split, modality, seq_length, split_indices=None):
  """Converts the specified dataset split to TFRecord format.
  Args:
    dataset_split: The dataset split (e.g., train, val).
  Raises:
    RuntimeError: If loaded image and label have different shape, or if the
      image file with specified postfix could not be found.
  """

  data_path = os.path.join(FLAGS.data_dir, _SPLIT_MAP[dataset_split], _MODALITY_MAP[modality][0])

  folder_for_each_subject = os.listdir(data_path)
  folder_for_each_subject.sort()
  if split_indices is not None:
    if split_indices[1] > len(folder_for_each_subject):
      raise ValueError("Out of Range")
    folder_for_each_subject = folder_for_each_subject[split_indices[0]:split_indices[1]]

  if FLAGS.num_shard is not None:
    num_shard = FLAGS.num_shard
  else:
    num_shard = len(folder_for_each_subject)

  # TODO: make dir automatically
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

    # for shard_id in range(num_shard):
    #   sys.stdout.write('\n>> Converting image %d/%d shard %d in num_frame %d' % (
    #     shard_id+1, num_shard, shard_id+1, len(image_files)))

    total_slices += len(image_files)
    
    shard_filename = '%s-%s-%05d-of-%05d.tfrecord' % (
          dataset_split, modality, shard_id, num_shard)

    output_filename = os.path.join(out_dir, shard_filename)
    height, width = _convert_single_subject_to_seq(dataset_name, output_filename, modality, seq_length, **kwargs)
    
    sys.stdout.write('\n>> [{}:{}] Converting image {}/{} shard {} in num_frame {} and size[{},{}]'.format(
        dataset_split, modality, shard_id+1, num_shard, shard_id+1, len(image_files), height, width))
  
  sys.stdout.write('\n' + 60*"-")
  sys.stdout.write('\n total_slices: {}'.format(total_slices))
  sys.stdout.write('\n')
  sys.stdout.flush()


 
def unit_test_get_files():
  total_files = {}
  for img_or_label in ['image', 'label']:
    for m in ["CT", "MR_T2", "MR_T1_In", "MR_T1_Out"]:
      path = os.path.join(FLAGS.data_dir, _SPLIT_MAP["train"], _MODALITY_MAP[m][0], str(1))
      files = _get_files(path, m, img_or_label)
      total_files[img_or_label+"--"+m] = files
      print(total_files.items())
      print(60*"o")


def main(unused_argv):
  # Only support converting 'train' and 'val' sets for now.
  # for dataset_split in ['train', 'val']:
  # unit_test_get_files()
  
  dataset_split = {
                #   "train": [0,16],
                #   "val": [16,20],
                  "test": None
                  }
  for m in ["CT", "MR_T2", "MR_T1_In", "MR_T1_Out"]:
#   for m in ["MR_T2"]:
    for split in dataset_split:
      modality_for_output = m
      if "MR_T1" in modality_for_output:
        modality_for_output = "MR_T1"
        
      out_dir = os.path.join(FLAGS.output_dir, "seq"+str(FLAGS.seq_length), _SPLIT_MAP[split], modality_for_output)
      _convert_dataset(FLAGS.dataset_name, out_dir, split, m, FLAGS.seq_length, dataset_split[split])


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)