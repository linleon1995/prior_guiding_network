#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 21:39:35 2019

@author: acm528_02
"""

import numpy as np

import collections
import os
import tensorflow as tf
import common
import input_preprocess
from core import preprocess_utils
from datasets import build_prior, file_utils, dataset_infos
from utils import train_utils

_DATASETS_INFORMATION = {
    '2013_MICCAI_Abdominal': dataset_infos._MICCAI_ABDOMINAL_INFORMATION,
    '2019_ISBI_CHAOS_CT': dataset_infos._ISBI_CHAOS_INFORMATION_CT,
    '2019_ISBI_CHAOS_MR_T1': dataset_infos._ISBI_CHAOS_INFORMATION_MR_T1,
    '2019_ISBI_CHAOS_MR_T2': dataset_infos._ISBI_CHAOS_INFORMATION_MR_T2,
}

# TODO: MR_T1 in and out
#TODO: test set
# /home/user/DISK/data/Jing/data/Training/tfrecord/
_DATASETS_STORING_PATH_MAP = {
    # '2013_MICCAI_Abdominal': {"train": "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/Train_Sets/tfrecord/",
    #                           "val":  "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/Train_Sets/tfrecord/",
    #                           "test": None},
    '2013_MICCAI_Abdominal': {"train": "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/Train_Sets/tfrecord_seq/",
                              "val":  "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/Train_Sets/tfrecord_seq/",
                              "test": "/home/user/DISK/data/Jing/data/2013_MICCAI_BTCV/tfrecord/seq3/Test_Sets/"},
                              
    '2019_ISBI_CHAOS_CT': {"train": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Train_Sets/CT/",
                           "val":  "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Train_Sets/CT/",
                           "test":  None},
    '2019_ISBI_CHAOS_MR_T1': {"train": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Train_Sets/MR_T1/",
                              "val": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Train_Sets/MR_T1/",
                              "test": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Test_Sets/MR_T1/"},
    '2019_ISBI_CHAOS_MR_T2': {"train": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Train_Sets/MR_T2/",
                              "val": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Train_Sets/MR_T2/",
                              "test": "/home/user/DISK/data/Jing/data/2019_ISBI_CHAOS/tfrecord/seq3/Test_Sets/MR_T2/"},
    }

# BASE_DATA_DIR = dataset_infos.BASE_DATA_DIR
SPLIT_FOLDER_MAP = {"train": "Train_Sets",
                    "val": "Train_Sets",
                    "test": "Test_Sets"}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'




# def get_data_dir(name, split, seq_length):
#     if seq_length is not None:
#         if seq_length > 1:
#             img_or_seq = "seq" + str(seq_length)
#         else:
#             img_or_seq = "img"
#     else:
#         img_or_seq = "img"
#     if name == "2013_MICCAI_Abdominal":
#         return os.path.join(BASE_DATA_DIR, "tfrecord", img_or_seq, SPLIT_FOLDER_MAP[split])
#     elif "2019_ISBI_CHAOS" in name:
#         modality = name.split("CHAOS_")[1]
#         return os.path.join(BASE_DATA_DIR, "tfrecord", img_or_seq, SPLIT_FOLDER_MAP[split], modality)
#     else:
#         raise ValueError("Unknown Dataset Name")


def get_z_label(z_label_method, num_slices, depth, z_class=None):
    if z_label_method == "cls":
        if z_class is not None:
            z_class = tf.cast(z_class, tf.float32)
            depth = tf.cast(depth, tf.float32)
            num_slices = tf.cast(num_slices, tf.float32)
            return tf.cast(tf.divide(depth, tf.divide(num_slices, z_class)), tf.int32)
        else:
            raise ValueError("Unknown z class")
    elif z_label_method == "reg":
        return depth / num_slices
    else:
        raise ValueError("Unknown z label method")


class Dataset(object):
    """"Represents input data for prior-guided based network"""
    def __init__(self,
                 dataset_name,
                 split_name,
                 batch_size,
                 crop_size,
                #  HU_window,
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
                 ):
        """Initializes the dataset."""
        # TODO: make sure num_class in all datasets are same

        self.dataset_dir = {}
        self.splits_to_sizes = {}
        for sub_dataset in dataset_name:
            if sub_dataset not in _DATASETS_INFORMATION:
                raise ValueError('The specified dataset is not supported.')
            # TODO: Only consider the first split dir --> split_name[0]
            self.dataset_dir[sub_dataset] = _DATASETS_STORING_PATH_MAP[sub_dataset][split_name[0]]
            # self.dataset_dir[sub_dataset] = get_data_dir(sub_dataset, split_name, seq_length)
            self.splits_to_sizes[sub_dataset] = _DATASETS_INFORMATION[sub_dataset].splits_to_sizes
        self.dataset_name = dataset_name
        # TODO: dataset information for each dataset
        self.dataset_infos = _DATASETS_INFORMATION[dataset_name[0]]
        
        for split in split_name:
            for sub_dataset_split in self.splits_to_sizes.values():
                if split not in sub_dataset_split:
                    raise ValueError('data split name %s not recognized' % split)

        if model_variant is None:
            tf.logging.warning('Please specify a model_variant.')

        # if not isinstance(seq_length, int):
        #     raise ValueError('Please specify the length of sequence')

        self.split_name = split_name
        # self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.crop_size = crop_size
        if pre_crop_flag:
            self.pre_crop_size = _DATASETS_INFORMATION[sub_dataset].train["pre_crop_size"]
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
        self.num_of_classes = _DATASETS_INFORMATION[self.dataset_name[0]].num_classes
        self.ignore_label = _DATASETS_INFORMATION[self.dataset_name[0]].ignore_label
        self.seq_length = seq_length
        self.seq_type = seq_type
        self.label_for_each_frame = label_for_each_frame


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
            # features['image/segmentation/class/organ_label'] = tf.FixedLenFeature((), tf.string, default_value='')

        parsed_features = tf.parse_single_example(example_proto, features)

        image = tf.decode_raw(parsed_features['image/encoded'], tf.int32)
        if "train" in self.split_name or "val" in self.split_name:
            # label = tf.decode_raw(parsed_features['image/segmentation/class/encoded'], tf.int32)
            label = tf.decode_raw(parsed_features['segmentation/encoded'], tf.int32)
            # organ_label = tf.decode_raw(parsed_features["image/segmentation/class/organ_label"], tf.int32)
        elif "test" in self.split_name:
            label = None

        sample = {
            common.IMAGE: image,
            common.HEIGHT: parsed_features['image/height'],
            common.WIDTH: parsed_features['image/width'],
            common.DEPTH: parsed_features['image/depth'],
            common.NUM_SLICES: parsed_features['image/num_slices'],
            # "organ_label": organ_label,
            # "split": self.split_name
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
            # 'segmentation/format': tf.FixedLenFeature(
            #     (), tf.string, default_value='png'),
            'video_id': tf.FixedLenFeature((), tf.string, default_value=''),
            'dataset/num_frames': tf.FixedLenFeature((), tf.int64, default_value=0),
        }

        # TODO
        # label_name = 'class' if dataset_name == 'davis_2016' else 'object'
        # label_name = 'miccai_2013'
        keys_to_sequence_features = {
            'image/encoded': tf.FixedLenSequenceFeature((), dtype=tf.string),
            # 'segmentation/encoded':
            #     tf.FixedLenSequenceFeature((), tf.string),
            'image/depth':
                tf.FixedLenSequenceFeature((), tf.int64),
        }
        if "train" in self.split_name or "val" in self.split_name:
          # keys_to_context_features['segmentation/format'] = tf.FixedLenFeature((), tf.string, default_value='png')
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
        """
        1. load prior
        2. check spatial dimension among all priors
        3. if different raise error
        4 if all prior's spatial dimension are same concat all prior example: CHAOS task1 + MICCAI --> [512,512,16]
        """
        if self.guidance_type in ("training_data_fusion", "training_data_fusion_h"):
            print("Input Prior Infomrmation: Slice=%d, Subject=%d" % (
                self.prior_num_slice, self.prior_num_subject))
            # # TODO: get prior name properly
            def load_prior(prior_dir):
                prior_name = "train-slice%03d-subject%03d" %(self.prior_num_slice, self.prior_num_subject)
                prior = np.load(os.path.join(prior_dir, prior_name+".npy"))
                return prior

            prior_list = []
            for sub_dataset in self.dataset_name:
                prior = load_prior(_DATASETS_INFORMATION[sub_dataset].prior_dir)
                prior = tf.convert_to_tensor(prior)
                # Consider prior in shape [H,W,K,1]
                prior = tf.image.resize_bilinear(tf.expand_dims(prior,axis=0)[...,0], [height,width])
                prior_list.append(prior)
            # TODO: only consider first dataset prior
            prior_segs = tf.concat(prior_list, axis=3)[0]
        else:
            prior_segs = None
        return prior_segs


    def _preprocessing_seq(self, sample):
        """
        image: [num_frame, height, width, channel]
        label: [num_frame, height, width, 1]
        prior_segs: [num_frame, height, width, class]
        """
        height = sample[common.HEIGHT]
        width = sample[common.WIDTH]

        # image = tf.reshape(sample[common.IMAGE], [height, width, self.seq_length])
        image = tf.reshape(sample[common.IMAGE], [self.seq_length, height, width])
        image = tf.transpose(image, [1,2,0])
        if common.LABEL in sample:
        #   label = tf.reshape(sample[common.LABEL], [height, width, self.seq_length])
          label = tf.reshape(sample[common.LABEL], [self.seq_length, height, width])
          label = tf.transpose(label, [1,2,0])
        else:
          label = None

        depth = sample[common.DEPTH]
        num_slices = sample[common.NUM_SLICES]

        # get prior
        if None not in (self.prior_num_slice, self.prior_num_subject):
            prior_segs = self.load_prior_from_dir(height, width)
            [_, _, prior_channel] = preprocess_utils.resolve_shape(prior_segs, rank=3)
        else:
            prior_segs = None

        # TODO: input prior shape should be [NHWC] but [HWKC]
        # TODO: prior_segs and prior_seg_3d

        # Preprocessing for images, label and z_label
        original_image, image, label, original_label, pp, prior_segs = input_preprocess.preprocess_image_and_label_seq(
            image=image,
            label=label,
            prior_segs=prior_segs,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            channel=1,
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


        # TODO: image channel from 1 --> data_infos.channel
        # if self.seq_length != 1:
        #     image = tf.reshape(image, [self.seq_length, self.crop_size[0], self.crop_size[1], 1])
        #     if label is not None:
        #         label = tf.reshape(label, [self.seq_length, self.crop_size[0], self.crop_size[1], 1])
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
        if self.mt_label_method in ("reg", "cls") and self.mt_label_type in ("class_label", "z_label"):
            if self.mt_label_method == "reg" and self.mt_label_type == "class_label":
                raise ValueError("Class label only accept classification method")

            if self.mt_label_type == "z_label":
                mt_label = get_z_label(self.mt_label_method, num_slices, depth, z_class=self.mt_class)
            # elif self.mt_label_type == "class_label":
            #     mt_label = context["image/class_label"]

            sample[common.Z_LABEL] = mt_label
        return sample

    def get_one_shot_iterator(self):
        """Gets an iterator that iterates across the dataset once.
        Returns:
          An iterator of type tf.data.Iterator.
        """
        # TODO: string case
        files = []
        for sub_data_dir in self.dataset_dir.values():
            files.extend(file_utils.get_file_list(sub_data_dir, fileStr=self.split_name,
                                                fileExt=["tfrecord"], sort_files=True))
   
        if "2019_ISBI_CHAOS_MR_T1" in self.dataset_name:
            new_files = [f for f in files if "MR_T1_Out" not in f]
            files = new_files
            # new_files = []
            # mr_t1_in, mr_t1_out, remain = [], [], []
            
            # for ele in files:
            #     if "MR_T1_In" in ele:
            #         mr_t1_in.append(ele)
            #     elif "MR_T1_Out" in ele:
            #         mr_t1_out.append(ele)
            #     else:
            #         remain.append(ele)
            # assert len(mr_t1_in) == len(mr_t1_out)
            
            # for idx, ele in enumerate(mr_t1_in):
            #     new_files.append(ele)
            #     new_files.append(mr_t1_out[idx])    
            # new_files.extend(remain)
            
        self.files = files
        if self.seq_length == 1:
            dataset = (
                tf.data.TFRecordDataset(files)
                .map(self._parse_image, num_parallel_calls=self.num_readers)
                .map(self._preprocessing_seq, num_parallel_calls=self.num_readers)
                )
        elif self.seq_length > 1:
            dataset = (
                tf.data.TFRecordDataset(files)
                .map(self._parse_sequence, num_parallel_calls=self.num_readers)
                .map(self._preprocessing_seq, num_parallel_calls=self.num_readers)
                )

        if self.shuffle_data:
          dataset = dataset.shuffle(buffer_size=100)

        if self.repeat_data:
          dataset = dataset.repeat()  # Repeat forever for training.
        else:
          dataset = dataset.repeat(1)

        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset



