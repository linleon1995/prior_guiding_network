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
import matplotlib.pyplot as plt
#from tf_unet_multi_task import util
#import inspect
#import time
import pickle
VGG_MEAN = [103.939, 116.779, 123.68]
from tf_tesis2.layer_multi_task import (fc_layer, new_conv_layer_bn, new_conv_layer, upsampling_layer, new_fc_layer,
                                        SRAM, RM, SRAM_prof, RM_prof, RM_com, RM_new_aggregation, RM_ori, rm)
from tf_tesis2.module import (nonlocal_dot)
from tf_tesis2.utils import (conv2d, atrous_conv2d, split_separable_conv2d)
from tf_tesis2 import feature_extractor
RM = rm

#split_separable_conv2d(inputs, 
#                          filter_shape, 
#                          strides=[1,1,1,1],
#                          padding='SAME',
#                          dilations=[1,1], 
#                          channel_multiplier=1, 
#                          activate_func=None, 
#                          bn_flag=True, 
#                          is_training=True,
#                          reuse=False,
#                          scope=None):
def res_utils(inputs,
              depth=64,
              stride=1,
              unit_nums=1,
              rate=1,
              is_training=True,
              scope='resnet_block'):  
    with tf.variable_scope(scope):
        shortcut = inputs
        for unit in range(unit_nums):
            in_depth = inputs.get_shape().as_list()[3]
            with tf.variable_scope('unit'+str(unit)):
                if rate == 1:
                    relu1 = conv2d(inputs, [1,1,in_depth,depth], activate=tf.nn.relu, scope="conv1", is_training=is_training )
                    relu2 = conv2d(relu1, [3,3,depth,depth], activate=tf.nn.relu, scope="conv2", is_training=is_training )
                    inputs = conv2d(relu2 , [1,1,depth,depth*4], strides=[1,stride,stride,1], activate=tf.nn.relu, scope="conv3", is_training=is_training )
                else:
                    relu1 = atrous_conv2d(inputs, [1,1,in_depth,depth], rate=rate, scope="conv1", is_training=is_training)
                    relu2 = atrous_conv2d(relu1, [3,3,depth,depth], rate=rate, scope="conv2", is_training=is_training)
                    inputs = atrous_conv2d(relu2, [1,1,depth,depth*4], strides=[1,stride,stride,1], rate=rate, scope="conv3", is_training=is_training)
        
        out_depth = inputs.get_shape().as_list()[3]
        if in_depth != out_depth:
            shortcut = conv2d(shortcut , [1,1,in_depth,out_depth], scope="proj_shortcut", bn_flag=False)
        
        # TODO: spatial size different
        outputs = tf.add(shortcut, inputs)
    return outputs

def resnetv50(inputs,
              n_class,
              is_training=True,
              scope='resnet_v50'):
    """resnetv50"""
    in_depth = inputs.get_shape().as_list()[3]
    with tf.variable_scope(scope):
        with tf.variable_scope('block1'):
            conv1_1 = conv2d(inputs , [3,3,in_depth,64], strides=[1,2,2,1], activate=tf.nn.relu, scope="conv1", is_training=is_training )
            conv1_2 = conv2d(conv1_1 , [3,3,64,64], activate=tf.nn.relu, scope="conv2", is_training=is_training )
            conv1_3 = conv2d(conv1_2 , [3,3,64,128], activate=tf.nn.relu, scope="conv3", is_training=is_training )
        block2 = res_utils(conv1_3, depth=64, stride=2, unit_nums=3, is_training=is_training, scope='block2')
        block3 = res_utils(block2, depth=128, stride=2, unit_nums=4, is_training=is_training, scope='block3')
        block4 = res_utils(block3, depth=256, stride=2, unit_nums=6, rate=2, is_training=is_training, scope='block4')
        block5 = res_utils(block4, depth=512, stride=1, unit_nums=3, rate=4, is_training=is_training, scope='block5')

        end_points = {
                "pool1": conv1_3,
                "pool2": block2,
                "pool3": block3,
                "pool4": block4,
                "pool5": block5}
    return block5, end_points
        

def crn_encoder_sep_resnet50(in_node, n_class, z_class, batch_size, is_training=True ):
    """u-net with smaller depth and batch norm"""
    """
    allow negative in angle regression task, the output dim is 6 because predict affine parameters in here
    """
    # resnet
    with tf.variable_scope("Encoder"):
        features, end_points = resnetv50(inputs=in_node,
                                         n_class=n_class,
                                         is_training=is_training)
    output = {}
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
#        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
#        relu1 = tf.nn.relu( fc1, name='relu1' )
#        fc2 = new_fc_layer( relu1, [1000, 500], 1, "fc2")
        fc2 = tf.reduce_mean(features, axis=[1,2])
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [fc2.get_shape().as_list()[1], n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output

    return output, end_points


#def crn_encoder_sep_resnet50(in_node, n_class, z_class, batch_size, is_training=True ):
#    """u-net with smaller depth and batch norm"""
#    """
#    allow negative in angle regression task, the output dim is 6 because predict affine parameters in here
#    """
#    f_root = 32
#    channels = in_node.get_shape().as_list()[-1]
#    
#    # resnet
#    with tf.variable_scope("Encoder"):
#        output_stride = 16
#        multi_grid = [1, 2, 4]
#        model_variant = 'resnet_v1_50_beta'
#        fine_tune_batch_norm = False
#        
#        features, end_points = feature_extractor.extract_features(
#              in_node,
#              output_stride=output_stride,
#              multi_grid=multi_grid,
#              model_variant=model_variant,
##              depth_multiplier=model_options.depth_multiplier,
##              divisible_by=model_options.divisible_by,
##              weight_decay=weight_decay,
##              reuse=reuse,
#              is_training=is_training,
##              preprocessed_images_dtype=model_options.preprocessed_images_dtype,
#              fine_tune_batch_norm=fine_tune_batch_norm,
##              nas_stem_output_num_conv_filters=(
##                  model_options.nas_stem_output_num_conv_filters),
##              nas_training_hyper_parameters=nas_training_hyper_parameters,
##              use_bounded_activation=model_options.use_bounded_activation
#              )
#        for v in end_points:
#            print(30*'=')
#            print(v)
#            
#            print(v, end_points[v])
#            
#        
#    output = {}
#    layer_dict = {
#            "pool1": end_points["Encoder/resnet_v1_50/conv1_3"],
#            "pool2": end_points["Encoder/resnet_v1_50/block1/unit_3/bottleneck_v1/conv1"],
#            "pool3": end_points["Encoder/resnet_v1_50/block2/unit_4/bottleneck_v1/conv1"],
#            "pool4": end_points["Encoder/resnet_v1_50/block3/unit_6/bottleneck_v1/conv1"],
#            "pool5": end_points["Encoder/resnet_v1_50/block4"],
#            }
##    pool5_flat = tf.layers.flatten(pool5)
#    
#        
#    # fc for z_axis classification
#    with tf.variable_scope("Z_Classification"):
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        relu1 = tf.nn.relu( fc1, name='relu1' )
##        fc2 = new_fc_layer( relu1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(features, axis=[1,2])
#        relu2 = tf.nn.relu( fc2, name='relu2' )
#        fc3 = new_fc_layer( relu2, [fc2.get_shape().as_list()[1], n_class*(z_class+1)], 1, "fc3")
#        z_output = tf.nn.relu( fc3, name='relu3' )
#        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
#        output['z_output'] = z_output
#
#    return output, layer_dict
    

def crn_decoder_sep_resnet50( output, guidance, batch_size, layer_dict, embed=16, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool5"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, f1, vis1 = RM(layer_dict["pool5"], zero_tensor, guidance_in, None, embed, upsample=False, name='RM_5', is_training=is_training)
        guidance1_a = tf.nn.softmax(guidance1)
        feature2, guidance2, f2, vis2 = RM(layer_dict["pool4"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)
        feature3, guidance3, f3, vis3 = RM(layer_dict["pool3"], feature2, guidance2_a, None, embed, name='RM_3', is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        feature4, guidance4, f4, vis4 = RM(layer_dict["pool2"], feature3, guidance3_a, None, embed, name='RM_2', is_training=is_training)
        guidance4_a = tf.nn.softmax(guidance4)
        feature5, guidance5, f5, vis5 = RM(layer_dict["pool1"], feature4, guidance4_a, None, embed, classifier=True, name='RM_1', is_training=is_training)

    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance5,
                  })
    info = [f1,f2,f3,f4, f5, vis1, vis2, vis3, vis4, vis5]
    output['output_map'] = guidance5
    return output, layer_dict, info


def crn_encoder_sep_guid(in_node, n_class, z_class, batch_size, is_training=True ):
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
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
    
        relu3_1 = new_conv_layer_bn( pool2, [3,3,f_root*2,f_root*4], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,f_root*4,f_root*4], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,f_root*4,f_root*8], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,f_root*8,f_root*8], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        
##        #bottle neck
#        relu5_1 = new_conv_layer_bn( pool4, [3,3,f_root*8,f_root*16], "conv5_1", is_training)
#        relu5_2 = new_conv_layer_bn( relu5_1, [3,3,f_root*16,f_root*16], "conv5_2", is_training)
#        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                           padding='SAME', name='pool5')
        
    output = {}
#    pool5_flat = tf.layers.flatten(pool5)
    
        
#    # fc for z_axis classification
#    with tf.variable_scope("Z_Classification"):
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        relu1 = tf.nn.relu( fc1, name='relu1' )
##        fc2 = new_fc_layer( relu1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(pool4, axis=[1,2])
#        relu2 = tf.nn.relu( fc2, name='relu2' )
#        fc3 = new_fc_layer( relu2, [fc2.get_shape().as_list()[1], n_class*(z_class+1)], 1, "fc3")
#        z_output = tf.nn.relu( fc3, name='relu3' )
#        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
#        output['z_output'] = z_output
#        
#    # fc for angle regression
#    with tf.variable_scope("Angle_Regression"):  
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        fc2 = new_fc_layer( fc1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(pool4, axis=[1,2])
#        fc3 = new_fc_layer( fc2, [fc2.get_shape().as_list()[1], 6], 1, "fc3")
#        output['angle_output'] = fc3
        
    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
    return output, layer_dict
    

def crn_decoder_sep_guid( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, f1, vis1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training)
        guidance1_a = tf.nn.softmax(guidance1)
        feature2, guidance2, f2, vis2 = RM(layer_dict["pool3"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)
        feature3, guidance3, f3, vis3 = RM(layer_dict["pool2"], feature2, guidance2_a, None, embed, name='RM_3', is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        feature4, guidance4, f4, vis4 = RM(layer_dict["pool1"], feature3, guidance3_a, None, embed, classifier=True, name='RM_1', is_training=is_training)
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = guidance4
    return output, layer_dict, info


def crn_encoder_sep_com(in_node, n_class, z_class, batch_size, is_training=True ):
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
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
    
        relu3_1 = new_conv_layer_bn( pool2, [3,3,f_root*2,f_root*4], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,f_root*4,f_root*4], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
        
        relu4_1 = new_conv_layer_bn( pool3, [3,3,f_root*4,f_root*8], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,f_root*8,f_root*8], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        
##        #bottle neck
#        relu5_1 = new_conv_layer_bn( pool4, [3,3,f_root*8,f_root*16], "conv5_1", is_training)
#        relu5_2 = new_conv_layer_bn( relu5_1, [3,3,f_root*16,f_root*16], "conv5_2", is_training)
#        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
#                           padding='SAME', name='pool5')
        
    output = {}
#    pool5_flat = tf.layers.flatten(pool5)
    
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
#        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
#        relu1 = tf.nn.relu( fc1, name='relu1' )
#        fc2 = new_fc_layer( relu1, [1000, 500], 1, "fc2")
        fc2 = tf.reduce_mean(pool4, axis=[1,2])
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [fc2.get_shape().as_list()[1], n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output
        
#    # fc for angle regression
#    with tf.variable_scope("Angle_Regression"):  
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        fc2 = new_fc_layer( fc1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(pool4, axis=[1,2])
#        fc3 = new_fc_layer( fc2, [fc2.get_shape().as_list()[1], 6], 1, "fc3")
#        output['angle_output'] = fc3
        
    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
    return output, layer_dict
    

def crn_decoder_sep_com( output, guidance, n_class, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, com1, f1, vis1 = RM_com(layer_dict["pool4"], zero_tensor, guidance_in, None, None, embed, 
                                                     name='RM_5', is_training=is_training)
        guidance1_a = tf.nn.softmax(guidance1)
        feature2, guidance2, com2, f2, vis2 = RM_com(layer_dict["pool3"], feature1, guidance1_a, None, com1, embed, 
                                               name='RM_4', is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)
        feature3, guidance3, com3, f3, vis3 = RM_com(layer_dict["pool2"], feature2, guidance2_a, None, com2, embed, 
                                               name='RM_3', is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        feature4, guidance4, com4, f4, vis4 = RM_com(layer_dict["pool1"], feature3, guidance3_a, None, com3, embed, 
                                               name='RM_1', is_training=is_training)
        fusion = tf.concat([feature4, guidance4, com4], axis=-1)
        # TODO: shape problem
#        print(fusion, feature4, guidance4, com4, 'sfg', tf.shape(fusion)[3])
        logits = conv2d(fusion, [1,1,78, n_class], activate=None, scope="logits", is_training=is_training )
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": logits,
                  })
    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = logits
    return output, layer_dict, info


def crn_encoder_sep_new_aggregation(in_node, n_class, z_class, batch_size, is_training=True ):
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
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = new_conv_layer_bn( pool2, [3,3,f_root*2,f_root*4], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,f_root*4,f_root*4], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,f_root*4,f_root*8], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,f_root*8,f_root*8], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        

    output = {}

#    pool5_flat = tf.layers.flatten(pool5)
    
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
#        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
#        relu1 = tf.nn.relu( fc1, name='relu1' )
#        fc2 = new_fc_layer( relu1, [1000, 500], 1, "fc2")
        fc2 = tf.reduce_mean(pool4, axis=[1,2])
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [fc2.get_shape().as_list()[1], n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output
        
#    # fc for angle regression
#    with tf.variable_scope("Angle_Regression"):  
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        fc2 = new_fc_layer( fc1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(pool4, axis=[1,2])
#        fc3 = new_fc_layer( fc2, [fc2.get_shape().as_list()[1], 6], 1, "fc3")
#        output['angle_output'] = fc3
        
    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
    return output, layer_dict
    

def crn_decoder_sep_new_aggregation( output, guidance, n_class, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance[...,1:], [h, w], name='guidance_in')
        guidance_in = tf.tile(guidance_in, [1,1,1,embed])
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, f1, vis1 = RM_new_aggregation(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training)
        feature2, guidance2, f2, vis2 = RM_new_aggregation(layer_dict["pool3"], feature1, guidance1, None, embed, name='RM_4', is_training=is_training)
        feature3, guidance3, f3, vis3 = RM_new_aggregation(layer_dict["pool2"], feature2, guidance2, None, embed, name='RM_3', is_training=is_training)
        feature4, guidance4, f4, vis4 = RM_new_aggregation(layer_dict["pool1"], feature3, guidance3, None, embed, classifier=True, name='RM_1', is_training=is_training)
        
        g1 = tf.image.resize_bilinear(guidance1, [256,256])
        g2 = tf.image.resize_bilinear(guidance2, [256,256])
        g3 = tf.image.resize_bilinear(guidance3, [256,256])
        g4 = tf.image.resize_bilinear(guidance4, [256,256])
        
        g1_conv = conv2d(g1, [1,1,g1.get_shape().as_list()[3],n_class], scope="g1_conv", is_training=is_training)
        g2_conv = conv2d(g2, [1,1,g1.get_shape().as_list()[3],n_class], scope="g2_conv", is_training=is_training)
        g3_conv = conv2d(g3, [1,1,g1.get_shape().as_list()[3],n_class], scope="g3_conv", is_training=is_training)
        g4_conv = conv2d(g4, [1,1,g1.get_shape().as_list()[3],n_class], scope="g4_conv", is_training=is_training)
        
        logits = tf.add_n([g1_conv,g2_conv,g3_conv,g4_conv])
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = logits
    return output, layer_dict, info



def crn_encoder(in_node, n_class, z_class, batch_size, is_training=True ):
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
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = new_conv_layer_bn( pool2, [3,3,f_root*2,f_root*4], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,f_root*4,f_root*4], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,f_root*4,f_root*8], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,f_root*8,f_root*8], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        

    output = {}

    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
    return output, layer_dict
    

def crn_decoder( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance[...,1:], [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, f1, vis1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, tf.nn.sigmoid, embed, name='RM_5', is_training=is_training)
        feature2, guidance2, f2, vis2 = RM(layer_dict["pool3"], feature1, guidance1, tf.nn.sigmoid, embed, name='RM_4', is_training=is_training)
        feature3, guidance3, f3, vis3 = RM(layer_dict["pool2"], feature2, guidance2, tf.nn.sigmoid, embed, name='RM_3', is_training=is_training)
        feature4, guidance4, f4, vis4 = RM(layer_dict["pool1"], feature3, guidance3, None, embed, name='RM_1', is_training=is_training)
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = guidance4
    return output, layer_dict, info



def crn_encoder_sep(in_node, n_class, z_class, batch_size, is_training=True ):
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
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')

        relu3_1 = new_conv_layer_bn( pool2, [3,3,f_root*2,f_root*4], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,f_root*4,f_root*4], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,f_root*4,f_root*8], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,f_root*8,f_root*8], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        

    output = {}

#    pool5_flat = tf.layers.flatten(pool5)
    
        
    # fc for z_axis classification
#    with tf.variable_scope("Z_Classification"):
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        relu1 = tf.nn.relu( fc1, name='relu1' )
##        fc2 = new_fc_layer( relu1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(pool4, axis=[1,2])
#        relu2 = tf.nn.relu( fc2, name='relu2' )
#        fc3 = new_fc_layer( relu2, [fc2.get_shape().as_list()[1], n_class*(z_class+1)], 1, "fc3")
#        z_output = tf.nn.relu( fc3, name='relu3' )
#        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
#        output['z_output'] = z_output
        
#    # fc for angle regression
#    with tf.variable_scope("Angle_Regression"):  
##        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 1000], 1, "fc1")
##        fc2 = new_fc_layer( fc1, [1000, 500], 1, "fc2")
#        fc2 = tf.reduce_mean(pool4, axis=[1,2])
#        fc3 = new_fc_layer( fc2, [fc2.get_shape().as_list()[1], 6], 1, "fc3")
#        output['angle_output'] = fc3
        
    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
    return output, layer_dict
    

#def crn_decoder_sep( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
#    with tf.variable_scope("Decoder"):   
#        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
#        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
#        zero_tensor = tf.zeros([batch_size, h, w, embed])
#
#        feature1, guidance1, f1, vis1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training)
#        guidance1_a = tf.nn.softmax(guidance1)
#        feature2, guidance2, f2, vis2 = RM(layer_dict["pool3"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
#        guidance2_a = tf.nn.softmax(guidance2)
#        feature3, guidance3, f3, vis3 = RM(layer_dict["pool2"], feature2, guidance2_a, None, embed, name='RM_3', is_training=is_training)
#        guidance3_a = tf.nn.softmax(guidance3)
#        feature4, guidance4, f4, vis4 = RM(layer_dict["pool1"], feature3, guidance3_a, None, embed, classifier=True, name='RM_1', is_training=is_training)
#    layer_dict.update({
#                  "guidance_in": guidance_in,
#                  "feature1": feature1, "guidance1": guidance1,
#                  "feature2": feature2, "guidance2": guidance2,
#                  "feature3": feature3, "guidance3": guidance3,
#                  "feature4": feature4, "guidance4": guidance4,
##                  "feature5": feature5, "guidance5": guidance5,
#                  "output": feature4,
#                  })
#    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
#    output['output_map'] = feature4
#    return output, layer_dict, info


def crn_decoder_sep( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
    with tf.variable_scope("Decoder"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, f1, vis1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training)
        guidance1_a = tf.nn.softmax(guidance1)
        feature2, guidance2, f2, vis2 = RM(layer_dict["pool3"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)
        feature3, guidance3, f3, vis3 = RM(layer_dict["pool2"], feature2, guidance2_a, None, embed, name='RM_3', is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        feature4, guidance4, f4, vis4 = RM(layer_dict["pool1"], feature3, guidance3_a, None, embed, classifier=True, name='RM_1', is_training=is_training)
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = guidance4
    return output, layer_dict, info


def crn_atrous_encoder_sep(in_node, n_class, z_class, batch_size, seq_length=None, is_training=True ):
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

        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
        map = tf.reduce_mean(pool4, axis=[1,2])
        relu1 = tf.nn.relu(map, name='relu1' )
        fc1 = new_fc_layer( relu1, [relu1.get_shape().as_list()[1], 512], 1, "fc1")
        relu2 = tf.nn.relu(fc1, name='relu2' )
        z_output = new_fc_layer(relu2, [512, 1], 1, "z_output")
        output['z_output'] = z_output
        
#    with tf.variable_scope("Z_Classification"):
#        conv1 = conv2d(pool4, [1,1,f_root*8,1], activate=tf.nn.relu, scope="conv1", is_training=is_training)   
#        conv1 = tf.layers.flatten(conv1)
#        fc1 = new_fc_layer(conv1, [conv1.get_shape().as_list()[1], 512], 1, "fc1")
#        relu2 = tf.nn.relu(fc1, name='relu2' )
#        z_output = new_fc_layer(relu2, [512, z_class], 1, "z_output")
#        output['z_output'] = z_output
        

        
    layer_dict = {
            "conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
            "conv2_1": relu2_1, "conv2_2": relu2_2,
            "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
            "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
#            "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
            }
#    print(layer_dict)
    return output, layer_dict
    

#def crn_atrous_decoder_sep( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
#    with tf.variable_scope("Decoder"):   
#        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
#        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
#        zero_tensor = tf.zeros([batch_size, h, w, embed])
#
#        feature1, guidance1, f1, vis1 = RM(layer_dict["pool4"], zero_tensor, guidance_in, None, embed, name='RM_5', is_training=is_training)
#        guidance1_a = tf.nn.softmax(guidance1)
#        feature2, guidance2, f2, vis2 = RM(layer_dict["pool3"], feature1, guidance1_a, None, embed, name='RM_4', is_training=is_training)
#        guidance2_a = tf.nn.softmax(guidance2)
#        feature3, guidance3, f3, vis3 = RM(layer_dict["conv2_2"], feature2, guidance2_a, None, embed, upsample=False, name='RM_3', is_training=is_training)
#        guidance3_a = tf.nn.softmax(guidance3)
#        feature4, guidance4, f4, vis4 = RM(layer_dict["pool1"], feature3, guidance3_a, None, embed, classifier=True, name='RM_1', is_training=is_training)
#    layer_dict.update({
#                  "guidance_in": guidance_in,
#                  "feature1": feature1, "guidance1": guidance1,
#                  "feature2": feature2, "guidance2": guidance2,
#                  "feature3": feature3, "guidance3": guidance3,
#                  "feature4": feature4, "guidance4": guidance4,
##                  "feature5": feature5, "guidance5": guidance5,
#                  "output": feature4,
#                  })
#    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
#    output['output_map'] = feature4
#    return output, layer_dict, info

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


def crn_atrous_decoder_sep2( output, guidance, batch_size, layer_dict, embed=32, is_training=True ):         
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