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
from utils import train_utils
DatasetDescriptor = collections.namedtuple(
    'DatasetDescriptor',
    [
        'splits_to_sizes',  # Splits of the dataset into training, val and test.
        'num_classes',  # Number of semantic classes, including the
                        # background class (if exists). For example, there
                        # are 20 foreground classes + 1 background class in
                        # the PASCAL VOC 2012 dataset. Thus, we set
                        # num_classes=21.
        'ignore_label',  # Ignore label value.
    ])


_MICCAI_ABDOMINAL_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 3111,
        'val': 668,
    },
    num_classes=14,
    ignore_label=255,
) 


_DATASETS_INFORMATION = {
    '2013_MICCAI_Abdominal': _MICCAI_ABDOMINAL_INFORMATION,
}

# Default file pattern of TFRecord of TensorFlow Example.
_FILE_PATTERN = '%s-*'

class Dataset(object):
    """"Represents input data for prior-guided based network"""
    def __init__(self,
                 dataset_name,
                 split_name,
                 dataset_dir,
                 model_options, 
                 batch_size,
                 crop_size,
                 HU_window,
                #  only_foreground,
                 z_label_method,
                 seq_length=None,
                 z_class=None,
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
                 num_prior_samples=None,
                 ):
        """Initializes the dataset."""
        if dataset_name not in _DATASETS_INFORMATION:
            raise ValueError('The specified dataset is not supported yet.')
        self.dataset_name = dataset_name
        
        splits_to_sizes = _DATASETS_INFORMATION[dataset_name].splits_to_sizes

        if split_name not in splits_to_sizes:
          raise ValueError('data split name %s not recognized' % split_name)
      
        if model_variant is None:
            tf.logging.warning('Please specify a model_variant.')
      
      
        self.split_name = split_name
        self.dataset_dir = dataset_dir
        self.model_options=model_options
        self.batch_size = batch_size
        self.crop_size = crop_size
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
        self.z_class = z_class
        self.z_label_method = z_label_method
        self.HU_window = HU_window
        self.num_prior_samples = num_prior_samples
        
        self.num_of_classes = _DATASETS_INFORMATION[self.dataset_name].num_classes
        self.ignore_label = _DATASETS_INFORMATION[self.dataset_name].ignore_label

    def _parse_prior(self, example_proto, parse_name):
        """Function to parse the prior example"""
        features = {
          'prior/num_slices': 
              tf.FixedLenFeature((), tf.int64, default_value=0),
          'prior/'+parse_name: 
              tf.FixedLenFeature((), tf.string, default_value=''),}
        
        parsed_features = tf.parse_single_example(example_proto, features)
        prior = tf.decode_raw(parsed_features['prior/'+parse_name], tf.int32)
        prior = tf.reshape(prior , [1,512,512,20])
        # prior = tf.image.resize_nearest_neighbor(prior, [256,256])
        prior = tf.squeeze(prior, axis=0)
        num_slices = parsed_features['prior/num_slices']
        
        sample = {parse_name: prior,
                  'num_slices': num_slices}
        return sample
      
    def _parse_function(self, example_proto):
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
        'image/segmentation/class/encoded':
            tf.FixedLenFeature((), tf.string, default_value=''),
        'image/segmentation/class/format':
            tf.FixedLenFeature((), tf.string, default_value=''),
        }
        
        parsed_features = tf.parse_single_example(example_proto, features)
        
        image = tf.decode_raw(parsed_features['image/encoded'], tf.int32)
        image = tf.reshape(image, [512,512,1])
        
        label = tf.decode_raw(parsed_features['image/segmentation/class/encoded'], tf.int32)
        label = tf.reshape(label, [512,512,1])
        # label = tf.reshape(label, [parsed_features['image/height'], parsed_features['image/width']])
        
        # import prior
        # TODO: paramarize subject selection
        # TODO: 'priors' --> common.PRIORS
        
        sample = {
            common.IMAGE: image,
            common.HEIGHT: parsed_features['image/height'],
            common.WIDTH: parsed_features['image/width'],
            common.DEPTH: parsed_features['image/depth'],
            common.NUM_SLICES: parsed_features['image/num_slices'],
        }
        

        if label is not None:
          if label.get_shape().ndims == 2:
            label = tf.expand_dims(label, 2)
          elif label.get_shape().ndims == 3 and label.shape.dims[2] == 1:
            pass
          else:
            raise ValueError('Input label shape must be [height, width], or '
                             '[height, width, 1].')
    
          label.set_shape([None, None, 1])
    
          sample[common.LABELS_CLASS] = label
      
        return sample
     
    def _preprocessing(self, sample):
        image = sample[common.IMAGE]
        label = sample[common.LABELS_CLASS]
        depth = sample[common.DEPTH]
        num_slices = sample[common.NUM_SLICES]
        
        # TODO: clear sample problem
        path = self.dataset_dir.split('tfrecord')[0]
        
        if self.model_options.affine_transform and self.model_options.deformable_transform:
          prior_imgs = train_utils.load_nibabel_data(
            path+'raw/', processing_list=np.arange(1))[0]
          prior_imgs = np.swapaxes(np.swapaxes(prior_imgs, 0, 2), 0, 1)
          prior_imgs_slices = prior_imgs.shape[2]
          sample['prior_slices'] = prior_imgs_slices
          
          if self.num_prior_samples is not None:
            sample_rate = prior_imgs_slices / self.num_prior_samples
            indices = np.arange(0, prior_imgs_slices, sample_rate)
            indices = np.int32(indices)
            prior_imgs = np.take(prior_imgs, indices, axis=2)
            sample['prior_slices'] = self.num_prior_samples
            
          prior_imgs = tf.convert_to_tensor(prior_imgs)
          prior_imgs = tf.cast(prior_imgs, tf.int32)
          sample[common.PRIOR_IMGS] = prior_imgs
        else:
          prior_imgs = None
          
        if self.model_options.affine_transform:                                  
          prior_segs = train_utils.load_nibabel_data(
            path+'label/', processing_list=np.arange(1))[0]
          prior_segs = np.swapaxes(np.swapaxes(prior_segs, 0, 2), 0, 1)
          prior_segs_slices = prior_segs.shape[2]
          sample['prior_slices'] = prior_segs_slices
            
          if self.num_prior_samples is not None:
            sample_rate = prior_segs_slices / self.num_prior_samples
            indices = np.arange(0, prior_segs_slices, sample_rate)
            indices = np.int32(indices)
            prior_segs = np.take(prior_segs, indices, axis=2)
            sample['prior_slices'] = self.num_prior_samples

          prior_segs = tf.convert_to_tensor(prior_segs)
          prior_segs = tf.cast(prior_segs, tf.int32)
          sample[common.PRIOR_SEGS] = prior_segs
        else:
          prior_segs = None 
        
        if self.model_options.affine_transform and self.model_options.deformable_transform:
          assert prior_imgs_slices == prior_segs_slices 
            
        # Preprocessing for images, label and z_label
        original_image, image, label, original_label, z_label, prior_imgs, prior_segs = input_preprocess.preprocess_image_and_label(
            image=image,
            label=label,
            depth=depth,
            prior_imgs=prior_imgs,
            prior_segs=prior_segs,
            num_slices=num_slices,
            crop_height=self.crop_size[0],
            crop_width=self.crop_size[1],
            z_label_method=self.z_label_method,
            z_class=self.z_class,
            HU_window=self.HU_window,
            min_resize_value=self.min_resize_value,
            max_resize_value=self.max_resize_value,
            resize_factor=self.resize_factor,
            min_scale_factor=self.min_scale_factor,
            max_scale_factor=self.max_scale_factor,
            scale_factor_step_size=self.scale_factor_step_size,
            ignore_label=self.ignore_label,
            is_training=self.is_training,
            model_variant=self.model_variant,
            num_prior_samples=self.num_prior_samples)

        # Preprocessing for segmentation prior and image prior
        # input_preprocess.preprocess_prior()
        
        
        sample[common.IMAGE] = image
    
        if not self.is_training:
          # Original image is only used during visualization.
          sample[common.ORIGINAL_IMAGE] = original_image
    
        if label is not None:
          sample[common.LABEL] = label
          
        # Remove common.LABEL_CLASS key in the sample since it is only used to
        # derive label and not used in training and evaluation.
        sample.pop(common.LABELS_CLASS, None)

        if z_label is not None:
          sample[common.Z_LABEL] = z_label

        # Remove common.DEPTH key and NUM_SLICES key in the sample since they are only used to
        # derive z_label and not used in training and evaluation.
        # TODO: [16,16] links to output_strides
        if prior_imgs is not None:
            prior_imgs = tf.image.resize_bilinear(tf.expand_dims(prior_imgs, axis=0), [16,16])
            prior_imgs = tf.squeeze(prior_imgs, axis=0)
            sample[common.PRIOR_IMGS] = prior_imgs
        if prior_segs is not None:
            prior_segs = tf.image.resize_bilinear(tf.expand_dims(prior_segs, axis=0), [32,32])
            prior_segs = tf.squeeze(prior_segs, axis=0)
            sample[common.PRIOR_SEGS] = prior_segs
            
        # sample.pop(common.DEPTH, None)
#        sample.pop(common.NUM_SLICES, None)  

        return sample
     
    def get_one_shot_iterator(self):
        """Gets an iterator that iterates across the dataset once.
        Returns:
          An iterator of type tf.data.Iterator.
        """
    
        files = self._get_all_files(self.split_name)
        
        dataset = (
             tf.data.TFRecordDataset(files)
             .map(self._parse_function, num_parallel_calls=self.num_readers)
             .map(self._preprocessing, num_parallel_calls=self.num_readers)
             )
    
        if self.shuffle_data:
          dataset = dataset.shuffle(buffer_size=100)
    
        if self.repeat_data:
          dataset = dataset.repeat()  # Repeat forever for training.
        else:
          dataset = dataset.repeat(1)
    
        dataset = dataset.batch(self.batch_size).prefetch(self.batch_size)
        return dataset.make_one_shot_iterator()
    
    def _get_all_files(self, split_name):
        """Gets all the files to read data from.
        Returns:
          A list of input files.
        """
        file_pattern = _FILE_PATTERN
        file_pattern = os.path.join(self.dataset_dir,
                                    file_pattern % split_name)
        return tf.gfile.Glob(file_pattern)       
     