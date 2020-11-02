#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:39:35 2019

@author: Jing-Siang, Lin
"""

import numpy as np

import collections
import os
import tensorflow as tf
import common
import input_preprocess
from core import preprocess_utils
from datasets import file_utils, dataset_infos
from utils import train_utils

_DATASETS_INFORMATION = {
    '2015_MICCAI_Abdominal': dataset_infos._MICCAI_ABDOMINAL_INFORMATION,
    '2019_ISBI_CHAOS_CT': dataset_infos._ISBI_CHAOS_INFORMATION_CT,
    '2019_ISBI_CHAOS_MR_T1': dataset_infos._ISBI_CHAOS_INFORMATION_MR_T1,
    '2019_ISBI_CHAOS_MR_T2': dataset_infos._ISBI_CHAOS_INFORMATION_MR_T2,
}

BASE_DATA_DIR = common.BASE_DATA_DIR
# TODO: shape check
# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'




def get_data_dir(dataset_infos, name, split, seq_length=None):
    if seq_length is not None:
        if seq_length > 1:
            img_or_seq = "seq" + str(seq_length)
        else:
            img_or_seq = "img"
    else:
        img_or_seq = "img"
        
    if name == "2015_MICCAI_Abdominal":
        return os.path.join(
            BASE_DATA_DIR, "2015_MICCAI_BTCV", "tfrecord", img_or_seq, 
            dataset_infos.split_folder_map[split])
    elif "2019_ISBI_CHAOS" in name:
        modality = name.split("CHAOS_")[1]
        return os.path.join(
            BASE_DATA_DIR, "2019_ISBI_CHAOS", "tfrecord", img_or_seq, 
            dataset_infos.split_folder_map[split], modality)
    else:
        raise ValueError("Unknown Dataset Name")


def get_z_label(mt_output_node, num_slices, depth, z_class=None):
    if mt_output_node > 1:
        if z_class is not None:
            z_class = tf.cast(z_class, tf.float32)
            depth = tf.cast(depth, tf.float32)
            num_slices = tf.cast(num_slices, tf.float32)
            return tf.cast(tf.divide(depth, tf.divide(num_slices, z_class)), tf.int32)
        else:
            raise ValueError("Unknown z class")
    elif mt_output_node == 1:
        return depth / num_slices
    else:
        raise ValueError("Unknown multi-task model output node")


class Dataset(object):
    """"Represents input data for prior-guided based network"""
    def __init__(self,
                 dataset_name,
                 split_name,
                 batch_size,
                 crop_size,
                 pre_crop_flag=None,
                 guidance_type=None,
                 min_resize_value=None,
                 max_resize_value=None,
                 resize_factor=None,
                 min_scale_factor=1.,
                 max_scale_factor=1.,
                 scale_factor_step_size=0,
                 model_variant=None,
                 num_readers=1,
                 is_training=False,
                 shuffle_data=False,
                 repeat_data=False,
                 mt_class=None,
                 mt_label_method=None,
                 mt_label_type=None,
                 prior_num_slice=None,
                 prior_num_subject=None,
                 seq_length=None,
                 seq_type=None,
                 label_for_each_frame=None,
                 z_loss_name=None,
                 ):
        """Initializes the dataset."""
        self.dataset_infos = _DATASETS_INFORMATION[dataset_name]
        self.dataset_dir = [get_data_dir(self.dataset_infos, dataset_name, split, seq_length) for split in split_name]
        self.splits_to_sizes = self.dataset_infos.splits_to_sizes
        
        self.dataset_name = dataset_name
        
        if model_variant is None:
            tf.logging.warning('Please specify a model_variant.')

        self.split_name = split_name
        self.batch_size = batch_size
        self.crop_size = crop_size
        if pre_crop_flag:
            self.pre_crop_size = self.dataset_infos.train["pre_crop_size"]
        else:
            self.pre_crop_size = [None, None]
        self.min_resize_value = min_resize_value
        self.max_resize_value = max_resize_value
        self.resize_factor = resize_factor
        self.min_scale_factor = min_scale_factor
        self.max_scale_factor = max_scale_factor
        self.scale_factor_step_size = scale_factor_step_size
        self.model_variant = model_variant
        self.num_readers = num_readers
        self.is_training = is_training
        self.shuffle_data = shuffle_data
        self.repeat_data = repeat_data
        self.mt_class = mt_class
        self.mt_label_method = mt_label_method
        self.mt_label_type = mt_label_type
        self.prior_num_slice = prior_num_slice
        self.prior_num_subject = prior_num_subject
        self.guidance_type = guidance_type
        self.num_of_classes = self.dataset_infos.num_classes
        self.ignore_label = self.dataset_infos.ignore_label
        self.seq_length = seq_length
        self.seq_type = seq_type
        self.label_for_each_frame = label_for_each_frame
        self.z_loss_name = z_loss_name

    def _parse_image(self, example_proto):
        """Function to parse the example"""
        features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/filename':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/height':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/width':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/depth':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        'image/num_slices':
            tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        if "train" in self.split_name or "val" in self.split_name:
            features['segmentation/encoded'] =  tf.FixedLenFeature((), tf.string, default_value='')
            
        parsed_features = tf.parse_single_example(example_proto, features)

        image = tf.decode_raw(parsed_features['image/encoded'], tf.int32)
        if "train" in self.split_name or "val" in self.split_name:
            label = tf.decode_raw(parsed_features['segmentation/encoded'], tf.int32)
        elif "test" in self.split_name:
            label = None

        sample = {
            common.IMAGE: image,
            common.HEIGHT: parsed_features['image/height'],
            common.WIDTH: parsed_features['image/width'],
            common.DEPTH: parsed_features['image/depth'],
            common.NUM_SLICES: parsed_features['image/num_slices'],
        }

        if label is not None:
          sample[common.LABEL] = label
        return sample


    def _parse_sequence(self, serialized_example):
        """Function to parse the example"""
        keys_to_context_features = {
            'image/format': tf.FixedLenFeature((), tf.string, default_value=''),
            'image/height': tf.FixedLenFeature((), tf.int64, default_value=0),
            'image/width': tf.FixedLenFeature((), tf.int64, default_value=0),
            'video_id': tf.FixedLenFeature((), tf.string, default_value=''),
            'dataset/num_frames': tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        keys_to_sequence_features = {
            'image/encoded': tf.FixedLenSequenceFeature((), dtype=tf.string),
            'image/depth':
                tf.FixedLenSequenceFeature((), tf.int64),
        }
        if "train" in self.split_name or "val" in self.split_name:
         keys_to_sequence_features['segmentation/encoded'] = tf.FixedLenSequenceFeature((), tf.string)

        context, feature_list = tf.parse_single_sequence_example(
            serialized_example, keys_to_context_features,
            keys_to_sequence_features)

        image = tf.decode_raw(feature_list["image/encoded"], tf.int32)
        if "train" in self.split_name or "val" in self.split_name:
            label = tf.decode_raw(feature_list["segmentation/encoded"], tf.int32)
        elif "test" in self.split_name:
            label = None

        depth = feature_list["image/depth"]
        num_slices = context["dataset/num_frames"]
        num_slices = tf.cast(num_slices, tf.int32)

        sample = {
            common.HEIGHT: context['image/height'],
            common.WIDTH: context['image/width'],
            common.IMAGE: image,
            common.NUM_SLICES: num_slices,
            common.DEPTH: depth,
        }
        if label is not None:
          sample[common.LABEL] = label

        # get multi-task label
        if self.mt_label_method in ("reg", "cls") and self.mt_label_type in ("class_label", "z_label"):
            if self.mt_label_method == "reg" and self.mt_label_type == "class_label":
                raise ValueError("Class label only accept classification method")

            if self.mt_label_type == "z_label":
                mt_label = get_z_label(self.mt_label_method, num_slices, depth, z_class=self.mt_class)
            elif self.mt_label_type == "class_label":
                mt_label = context["image/class_label"]

            sample[common.Z_LABEL] = mt_label
        return sample


    def load_prior_from_dir(self, height, width):
        """Load pre-defined prior in shape [Height, Width, Class, Z] where 
        Class is the segmentation categories, Z is the splitting number in longitudinal axis
        Return:
            prior_list: Contains all the prior from assign dataset.
        """
        def get_prior_dir(sub_data_dir):
            if self.dataset_name == "2019_ISBI_CHAOS":
                if "CT" in self.dataset_name:
                    modality = "CT"
                elif "MR_T1" in self.dataset_name:
                    modality = "MR_T1"
                elif "MR_T2" in self.dataset_name:
                    modality = "MR_T2"    
                prior_dir = os.path.join(sub_data_dir.split("tfrecord")[0], "priors", modality)
            elif self.dataset_name == "2015_MICCAI_Abdominal":
                prior_dir = os.path.join(sub_data_dir.split("tfrecord")[0], "priors")
            else:
                raise ValueError("Unknown Dataset Name") 
            return prior_dir
                
        prior_list = []
        for data_dir in self.dataset_dir:
            prior_dir = get_prior_dir(data_dir)
            prior = np.load(
                os.path.join(prior_dir, "train-slice%03d-subject%03d.npy" %(self.prior_num_slice, self.prior_num_subject)))
            # prior in shape [H,W,K,Z]
            prior = tf.convert_to_tensor(prior)
            # prior in shape [Z,H,W,K] for image resize
            prior = tf.transpose(tf.expand_dims(prior,axis=0), [4,1,2,3,0])[...,0]
            prior = tf.image.resize_bilinear(prior, [height,width])
            prior_list.append(prior)
        prior_segs = tf.concat(prior_list, axis=0)
        prior_segs = tf.reshape(prior_segs, [height, width, self.prior_num_slice*self.num_of_classes])
        return prior_segs


    def _preprocessing(self, sample):
        """
        image: [num_frame, height, width, channel]
        label: [num_frame, height, width, 1]
        prior_segs: [num_frame, height, width, class]
        """
        height = sample[common.HEIGHT]
        width = sample[common.WIDTH]
        image = tf.reshape(sample[common.IMAGE], [self.seq_length, height, width])
        image = tf.transpose(image, [1,2,0])
        if common.LABEL in sample:
          label = tf.reshape(sample[common.LABEL], [self.seq_length, height, width])
          label = tf.transpose(label, [1,2,0])
        else:
          label = None
        depth = sample[common.DEPTH]
        num_slices = sample[common.NUM_SLICES]

        # get prior
        # TODO: prior for pgn-v1
        if self.guidance_type == "training_data_fusion":
            # print("Input Prior Infomrmation: Slice=%d, Subject=%d" % (
            #     self.prior_num_slice, self.prior_num_subject))
            prior_segs = self.load_prior_from_dir(height, width)
            [_, _, prior_channel] = preprocess_utils.resolve_shape(prior_segs, rank=3)
        elif self.guidance_type == "ground_truth":
            prior_segs = label
        elif self.guidance_type == "zeros":
            prior_segs = tf.zeros_like(label)
        else:
            prior_segs = None

        # Preprocessing for images, label and z_label
        original_image, image, label, _, prior_segs = input_preprocess.preprocess_image_and_label_seq(
            image=image,
            label=label,
            prior_segs=prior_segs,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            channel=self.dataset_infos.channel,
            seq_length=self.seq_length,
            label_for_each_frame=self.label_for_each_frame,
            pre_crop_height=self.pre_crop_size[0],
            pre_crop_width=self.pre_crop_size[1],
            num_class=self.num_of_classes,
            HU_window=self.dataset_infos.HU_window,
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            ignore_label=self.ignore_label,
            is_training=self.is_training,
            model_variant=self.model_variant)

        if self.seq_length > 1:
            image = tf.expand_dims(tf.transpose(image, [2, 0, 1]), axis=3)
            if label is not None:
                label = tf.expand_dims(tf.transpose(label, [2, 0, 1]), axis=3)
                
        sample[common.IMAGE] = image
        if not self.is_training:
            # Original image is only used during visualization.
            sample[common.ORIGINAL_IMAGE] = original_image

        if label is not None:
            sample[common.LABEL] = label

        if prior_segs is not None:
            sample[common.PRIOR_SEGS] = tf.reshape(
                prior_segs, [self.crop_size[0], self.crop_size[1], prior_channel, 1])

        # get multi-task label
        if self.z_loss_name is not None:
            mt_label = get_z_label(self.z_loss_name, num_slices, depth, z_class=self.mt_class)
            sample[common.Z_LABEL] = mt_label
        return sample

    def get_dataset(self):
        """Gets an iterator that iterates across the dataset once.
        Returns:
          An iterator of type tf.data.Iterator.
        """
        files = []
        for sub_data_dir in self.dataset_dir:
            files.extend(file_utils.get_file_list(
                sub_data_dir, fileStr=self.split_name, fileExt=["tfrecord"], sort_files=True))

        self.files = files
        if self.seq_length == 1:
            dataset = (
                tf.data.TFRecordDataset(files)
                .map(self._parse_image, num_parallel_calls=self.num_readers)
                .map(self._preprocessing, num_parallel_calls=self.num_readers)
                )
        elif self.seq_length > 1:
            dataset = (
                tf.data.TFRecordDataset(files)
                .map(self._parse_sequence, num_parallel_calls=self.num_readers)
                .map(self._preprocessing, num_parallel_calls=self.num_readers)
                )

        if self.shuffle_data:
          dataset = dataset.shuffle(buffer_size=100)

        if self.repeat_data:
          dataset = dataset.repeat()  # Repeat forever for training.
        else:
          dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset



