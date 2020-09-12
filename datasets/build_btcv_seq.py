#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:18:33 2020

@author: Jing-Siang, Lin
"""

import glob
import math
import os.path
import re
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import build_medical_data
import file_utils
import build_btcv_img
_get_files = build_btcv_img._get_files
 
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True,
                    help='2015 MICCAI BTCV dataset root folder.')

parser.add_argument('--output_dir', type=str, required=True,
                    help='Path to save converted TensorFlow sequence examples.')                  

parser.add_argument('--split_indices', nargs='+', type=int,
                    help="Indices to for the training set and validation set splitting")

# parser.add_argument('--extract_fg_exist_slice', type=bool, default=False,
#                     help='Extract the slice including foreground')

parser.add_argument('--seq_length', type=int, default=3,
                    help='')  


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

# Image file pattern.
_IMAGE_FILENAME_RE = re.compile('(.+)' + _POSTFIX_MAP['image'])



def _convert_dataset(dataset_split, data_dir, seq_length, output_dir, split_indices=None):
    """Converts the specified dataset split to TFRecord format.
    Args:
        dataset_split: The dataset split (e.g., train, val).
    Raises:
        RuntimeError: If loaded image and label have different shape, or if the
        image file with specified postfix could not be found.
    """
    image_files = _get_files("image", data_dir, dataset_split, split_indices)
    if dataset_split in ("train", "val"):
        label_files = _get_files("label", data_dir, dataset_split, split_indices)

    image_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["image"], channels=1)
    if dataset_split in ("train", "val"):
        label_reader = build_medical_data.ImageReader(_DATA_FORMAT_MAP["label"], channels=1)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert each subject to single tfrecord sequence example     
    num_images = len(image_files)
    for shard_id in range(num_images):
        shard_filename = '%s-%s-%05d-of-%05d.tfrecord' % (
            dataset_split, "seq", shard_id, num_images)
        output_filename = os.path.join(output_dir, shard_filename)
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            # Read the image.
            image_data = image_reader.decode_image(image_files[shard_id])
            image_data = image_data[:,::-1]
            height, width, num_slices = image_reader.read_image_dims(image_data)
            
            image_type = "image"
            sys.stdout.write('\n>> [{}] Converting {} {}/{} shard {} in num_frame {} sequence length {} and size[{},{}]'.format(
                dataset_split, image_type, shard_id+1, num_images, shard_id+1, num_slices, seq_length, height, width))
      
      
            # Read the semantic segmentation annotation.
            if dataset_split in ("train", "val"):
                seg_data = label_reader.decode_image(label_files[shard_id])
                seg_data = seg_data[:,::-1]
                seg_height, seg_width, _ = label_reader.read_image_dims(seg_data)
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                
            re_match = _IMAGE_FILENAME_RE.search(image_files[shard_id])
            if re_match is None:
                raise RuntimeError('Invalid image filename: ' + image_files[shard_id])
            filename = os.path.basename(re_match.group(1))

            for i in range(num_slices):
                sequence = tf.train.SequenceExample()
                context = sequence.context.feature
                features = sequence.feature_lists.feature_list
                
                # Aceess slices for each sequence sample
                start = i - seq_length // 2
                for j in range(start, start+seq_length):
                    if j < 0:
                        slice_idx = 0
                    elif j > num_slices-1:
                        slice_idx = num_slices-1
                    else:
                        slice_idx = j
                    image_slice = image_data[slice_idx].tostring()
                    if dataset_split in ("train", "val"):
                        seg_slice = seg_data[slice_idx].tostring()

                    image_encoded = features['image/encoded'].feature.add()
                    image_encoded.bytes_list.value.append(image_slice)
                    if dataset_split in ("train", "val"):
                        segmentation_encoded = features['segmentation/encoded'].feature.add()
                        segmentation_encoded.bytes_list.value.append(seg_slice)
                    depth_encoded = features['image/depth'].feature.add()
                    depth_encoded.int64_list.value.append(slice_idx)
                    
                context['image/height'].int64_list.value.append(height)
                context['image/width'].int64_list.value.append(width)
                context['image/num_slices'].int64_list.value.append(num_slices)
                context['image/format'].bytes_list.value.append(_DATA_FORMAT_MAP["image"].encode('ascii'))
                
                tfrecord_writer.write(sequence.SerializeToString())
    sys.stdout.write('\n')
    sys.stdout.flush()

    
def main(unused_argv):
  assert FLAGS.split_indices[1] > FLAGS.split_indices[0]
  dataset_split = {"train": list(range(*FLAGS.split_indices)),
                   "val": list(range(*FLAGS.split_indices)),
                   "test": None}
  
  for split, indices in dataset_split.items():
    out_dir = os.path.join(FLAGS.output_dir, "seq"+str(FLAGS.seq_length), _SPLIT_MAP[split])
    _convert_dataset(split, FLAGS.data_dir, FLAGS.seq_length, out_dir, split_indices=indices)


if __name__ == '__main__':
  FLAGS, unparsed = parser.parse_known_args()
  main(unparsed)