#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 02:04:15 2019

@author: acm528_02
"""

from __future__ import print_function, division, absolute_import, unicode_literals

#import os
#import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
import matplotlib.pyplot as plt
#from tf_unet_multi_task import util
#import inspect
#import time
import pickle
VGG_MEAN = [103.939, 116.779, 123.68]
from tf_tesis2.layer_multi_task import (new_conv_layer_bn, new_conv_layer, upsampling_layer, new_fc_layer,
                                       rm)
from tf_tesis2.module import (nonlocal_dot)
from tf_tesis2.utils import (conv2d, atrous_conv2d, split_separable_conv2d, fc_layer)
from tf_tesis2 import feature_extractor, resnet_v1_beta
RM = rm
resnet_v1_50_beta = resnet_v1_beta.resnet_v1_50_beta
extract_features = feature_extractor.extract_features

def crn_resnetv50_encoder(in_node, 
                        n_class, 
                        z_class, 
                        batch_size, 
                        affine_flag, 
                        z_flag, 
                        output_stride=None, 
                        multi_grid=None,
                        reuse=None,
                        seq_length=None, 
                        is_training=True ):
    """u-net with smaller depth and batch norm"""
    """
    allow negative in angle regression task, the output dim is 6 because predict affine parameters in here
    """
    f_root = 32
    channels = in_node.get_shape().as_list()[-1]
    
            
    with tf.variable_scope("Encoder"):
        features, end_points = extract_features(images=in_node,
                                                 output_stride=8,
                                                 multi_grid=None,
#                                                 depth_multiplier=1.0,
#                                                 divisible_by=None,
#                                                 final_endpoint=None,
                                                 model_variant='resnet_v1_50_beta',
                                                 weight_decay=0.0001,
                                                 reuse=None,
                                                 is_training=is_training,
                                                 fine_tune_batch_norm=True,
                                                 regularize_depthwise=False,
                                                 preprocess_images=True,
                                                 preprocessed_images_dtype=tf.float32,
                                                 num_classes=None,
                                                 global_pool=False,
#                                                 nas_stem_output_num_conv_filters=20,
#                                                 nas_training_hyper_parameters=None,
                                                 use_bounded_activation=False)
            
#        features, end_points = resnet_v1_50_beta(inputs=in_node,
#                                            num_classes=None,
#                                            is_training=is_training,
#                                            global_pool=False,
#                                            output_stride=output_stride,
#                                            multi_grid=multi_grid,
#                                            reuse=reuse,
#                                            scope='resnet_v1_50')
        
    output = {}
    output["features"] = features
    layer_dict = {
                "pool1": end_points["Encoder/resnet_v1_50/conv1_3"],
                "conv2_2": end_points["Encoder/resnet_v1_50/block1/unit_3/bottleneck_v1/conv1"],
                "pool3": end_points["Encoder/resnet_v1_50/block2/unit_4/bottleneck_v1/conv1"],
                "pool4": end_points["Encoder/resnet_v1_50/block4"]}
    if z_flag:     
        # fc for z_axis classification
        with tf.variable_scope("Z_Classification"):
            map = tf.reduce_mean(features, axis=[1,2])
            relu1 = tf.nn.relu(map, name='relu1' )
            fc1 = new_fc_layer( relu1, [relu1.get_shape().as_list()[1], 512], 1, "fc1")
            relu2 = tf.nn.relu(fc1, name='relu2' )
            z_output = new_fc_layer(relu2, [512, 1], 1, "z_output")
            output['z_output'] = z_output

    if affine_flag:
        # fc for sptatial transform parameter
        with tf.variable_scope("Affine_transform"):
            map = tf.reduce_mean(features, axis=[1,2])
            relu1 = tf.nn.relu(map, name='relu1' )
            fc1 = new_fc_layer( relu1, [relu1.get_shape().as_list()[1], 512], 1, "fc1")
            relu2 = tf.nn.relu(fc1, name='relu2' )
            theta = new_fc_layer(relu2, [512, 6], 1, "theta")
            output['theta'] = theta
            print(layer_dict)
    return output, layer_dict


def crn_resnetv50_decoder( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = output["features"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        # guidance_in = guidance_in[...,1:]
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1 = RM(output["features"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training, upsample=False)
        guidance1_a = tf.nn.softmax(guidance1)

        feature2, guidance2 = RM(layer_dict["pool3"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)

        feature3, guidance3 = RM(layer_dict["conv2_2"], feature2, guidance2_a, None, embed, name='RM_3', is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        
        feature4, guidance4 = RM(layer_dict["pool1"], feature3, guidance3_a, None, embed, classifier=True, name='RM_1', is_training=is_training)
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
#    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = guidance4
    return output, layer_dict, []

    
def crn_atrous_encoder_sep(in_node, n_class, z_class, batch_size, affine_flag, z_flag, seq_length=None, is_training=True ):
    """u-net with smaller depth and batch norm"""
    """
    allow negative in angle regression task, the output dim is 6 because predict affine parameters in here
    """
    f_root = 32
    channels = in_node.get_shape().as_list()[-1]
    
            
    with tf.variable_scope("Encoder"):
        relu1_1 = new_conv_layer_bn( in_node, [3,3,channels,f_root], "conv1_1", is_training )
        relu1_2 = new_conv_layer_bn( relu1_1, [3,3,f_root,f_root], "conv1_2", is_training )
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')
    
        relu2_1 = new_conv_layer_bn(pool1, [3,3,f_root,f_root*2], "conv2_1", is_training)
        relu2_2 = new_conv_layer_bn(relu2_1, [3,3,f_root*2,f_root*2], "conv2_2", is_training)
#        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                               padding='SAME', name='pool2')
        
        
        relu3_1 = atrous_conv2d(relu2_2, [3,3,f_root*2,f_root*4], rate=2, scope="conv3_1", is_training=is_training)
        relu3_2 = atrous_conv2d(relu3_1, [3,3,f_root*4,f_root*4], rate=2, scope="conv3_2", is_training=is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
        
        relu4_1 = atrous_conv2d(pool3, [3,3,f_root*4,f_root*8], rate=4, scope="conv4_1", is_training=is_training)
        relu4_2 = atrous_conv2d(relu4_1, [3,3,f_root*8,f_root*8], rate=4, scope="conv4_2", is_training=is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')

    output = {}

    if z_flag:     
        # fc for z_axis classification
        with tf.variable_scope("Z_Classification"):
            map = tf.reduce_mean(pool4, axis=[1,2])
            relu1 = tf.nn.relu(map, name='relu1' )
            fc1 = new_fc_layer( relu1, [relu1.get_shape().as_list()[1], 512], 1, "fc1")
            relu2 = tf.nn.relu(fc1, name='relu2' )
            z_output = new_fc_layer(relu2, [512, 1], 1, "z_output")
            output['z_output'] = z_output

    if affine_flag:
        # fc for sptatial transform parameter
        with tf.variable_scope("Affine_transform"):
            map = tf.reduce_mean(pool4, axis=[1,2])
            relu1 = tf.nn.relu(map, name='relu1' )
            fc1 = new_fc_layer( relu1, [relu1.get_shape().as_list()[1], 512], 1, "fc1")
            relu2 = tf.nn.relu(fc1, name='relu2' )
            theta = new_fc_layer(relu2, [512, 6], 1, "theta")
            output['theta'] = theta

    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
#    print(layer_dict)
    return output, layer_dict
    

def crn_atrous_decoder_sep( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training)
        guidance1_a = tf.nn.softmax(guidance1)
        feature2, guidance2 = RM(layer_dict["pool3"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)
        feature3, guidance3 = RM(layer_dict["conv2_2"], feature2, guidance2_a, None, embed, upsample=False, name='RM_3', is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        feature4, guidance4 = RM(layer_dict["pool1"], feature3, guidance3_a, None, embed, classifier=True, name='RM_1', is_training=is_training)
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
#    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = guidance4
    return output, layer_dict, []


def crn_atrous_decoder_sep2( output, guidance, batch_size, layer_dict, embed=32, is_training=True):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, classifier=True, name='RM_5', is_training=is_training)
        logits = tf.image.resize_bilinear(guidance1, [256, 256], name='logits')
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "output": logits,
                  })

    output['output_map'] = logits
    return output, layer_dict, []


def crn_atrous_decoder_sep3( output, guidance, batch_size, layer_dict, n_class=14, embed=32, is_training=True):         
    with tf.variable_scope("Decoder"):   
        h, w, c = output["features"].get_shape().as_list()[1:]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')

        logits = conv2d(output["features"], [1,1, c, n_class], activate=None, scope="logits", is_training=is_training)
        logits = tf.image.resize_bilinear(logits, [256, 256], name='logits')
        
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "output": logits,
                  })

    output['output_map'] = logits
    return output, layer_dict, []