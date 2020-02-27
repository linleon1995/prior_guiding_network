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
#from tf_tesis2.layer_multi_task import (weight_variable, weight_variable_deconv, bias_variable, 
#                            conv2d, deconv2d, upsampling2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
#                            cross_entropy, batch_norm, softmax, fc_layer, new_conv_layer_bn, new_conv_layer, 
#                            upsampling_layer, new_fc_layer)
from tf_tesis2.layer_multi_task import (fc_layer, new_conv_layer_bn, new_conv_layer, upsampling_layer, new_fc_layer,
                                        SRAM, feature_combine, RM)




def unet_prior_guide(in_node, guidance, n_class, z_class, batch_size, is_training=True ):
    """u-net with smaller depth and batch norm"""
    channels = in_node.get_shape().as_list()[-1]
    with tf.variable_scope("Encoder"):
        relu1_1 = new_conv_layer_bn( in_node, [3,3,channels,32], "conv1_1", is_training )
        relu1_2 = new_conv_layer_bn( relu1_1, [3,3,32,32], "conv1_2", is_training )
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')
    
        relu2_1 = new_conv_layer_bn(pool1, [3,3,32,64], "conv2_1", is_training)
        relu2_2 = new_conv_layer_bn(relu2_1, [3,3,64,64], "conv2_2", is_training)
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
    
        relu3_1 = new_conv_layer_bn( pool2, [3,3,64,128], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,128,128], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,128,256], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,256,256], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        
        #bottle neck
        relu5_1 = new_conv_layer_bn( pool4, [3,3,256,512], "conv5_1", is_training)
        relu5_2 = new_conv_layer_bn( relu5_1, [3,3,512,512], "conv5_2", is_training)
        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')
        
    output = {}
    pool5_flat = tf.layers.flatten(pool5)
    
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output
        
    # fc for angle regression
    with tf.variable_scope("Angle_Regression"):  
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, 1], 1, "fc3")
        angle_output = tf.nn.relu( fc3, name='relu3' )
        output['angle_output'] = angle_output
        
        
    with tf.variable_scope("Decoder"):    
        guidance_in = tf.image.resize_bilinear(guidance, [8, 8], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, pool5.get_shape().as_list()[1], pool5.get_shape().as_list()[2], 32])
        
        feature1, guidance1 = RM(pool5, zero_tensor, guidance_in, tf.nn.softmax, 32, 'RM_5', is_training)
        feature2, guidance2 = RM(pool4, feature1, guidance1, tf.nn.softmax, 32, 'RM_4', is_training)
        feature3, guidance3 = RM(pool3, feature2, guidance2, tf.nn.softmax, 32, 'RM_3', is_training)
        feature4, guidance4 = RM(pool2, feature3, guidance3, tf.nn.softmax, 32, 'RM_2', is_training)
        feature5, guidance5 = RM(pool1, feature4, guidance4, None, 32, 'RM_1', is_training)
     
        
    layer_dict = {"conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
                  "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
                  "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
                  "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
                  "feature1": feature1, "guidance1": guidance,
                  "feature2": feature1, "guidance2": guidance,
                  "feature3": feature1, "guidance3": guidance,
                  "feature4": feature1, "guidance4": guidance,
                  "feature5": feature1, "guidance5": guidance,
                  "conv5_1": relu5_1, "conv5_2": relu5_2,
                  "output": guidance5,
                  }
    output['output_map'] = guidance5
    return output, layer_dict


def unet_prior_guide2(in_node, guidance, n_class, z_class, batch_size, is_training=True ):
    """u-net with smaller depth and batch norm"""
    """
    allow negative in angle regression task, the output dim is 6 because predict affine parameters in here
    """
    channels = in_node.get_shape().as_list()[-1]
    with tf.variable_scope("Encoder"):
        relu1_1 = new_conv_layer_bn( in_node, [3,3,channels,32], "conv1_1", is_training )
        relu1_2 = new_conv_layer_bn( relu1_1, [3,3,32,32], "conv1_2", is_training )
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')
    
        relu2_1 = new_conv_layer_bn(pool1, [3,3,32,64], "conv2_1", is_training)
        relu2_2 = new_conv_layer_bn(relu2_1, [3,3,64,64], "conv2_2", is_training)
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
    
        relu3_1 = new_conv_layer_bn( pool2, [3,3,64,128], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,128,128], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,128,256], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,256,256], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        
        #bottle neck
        relu5_1 = new_conv_layer_bn( pool4, [3,3,256,512], "conv5_1", is_training)
        relu5_2 = new_conv_layer_bn( relu5_1, [3,3,512,512], "conv5_2", is_training)
        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')
        
    output = {}
    pool5_flat = tf.layers.flatten(pool5)
    
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output
        
    # fc for angle regression
    with tf.variable_scope("Angle_Regression"):  
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        fc2 = new_fc_layer( fc1, [2048, 500], 1, "fc2")
        fc3 = new_fc_layer( fc2, [500, 6], 1, "fc3")
        output['angle_output'] = fc3
        
        
    with tf.variable_scope("Decoder"):    
        guidance_in = tf.image.resize_bilinear(guidance, [8, 8], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, pool5.get_shape().as_list()[1], pool5.get_shape().as_list()[2], 32])
        
        feature1, guidance1 = RM(pool5, zero_tensor, guidance_in, tf.nn.softmax, 32, 'RM_5', is_training)
        feature2, guidance2 = RM(pool4, feature1, guidance1, tf.nn.softmax, 32, 'RM_4', is_training)
        feature3, guidance3 = RM(pool3, feature2, guidance2, tf.nn.softmax, 32, 'RM_3', is_training)
        feature4, guidance4 = RM(pool2, feature3, guidance3, tf.nn.softmax, 32, 'RM_2', is_training)
        feature5, guidance5 = RM(pool1, feature4, guidance4, None, 32, 'RM_1', is_training)
     
        
    layer_dict = {"conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
                  "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
                  "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
                  "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
                  "conv5_1": relu5_1, "conv5_2": relu5_2, "pool5": pool5,
                  "feature1": feature1, "guidance1": guidance,
                  "feature2": feature1, "guidance2": guidance,
                  "feature3": feature1, "guidance3": guidance,
                  "feature4": feature1, "guidance4": guidance,
                  "feature5": feature1, "guidance5": guidance,
                  "output": guidance5,
                  }
    output['output_map'] = guidance5
    return output, layer_dict

       
def unet_concat_prior( in_node, n_class, z_class, batch_size, global_prior, is_training=True ):
    """u-net with smaller depth and batch norm"""
    channels = in_node.get_shape().as_list()[-1]
    with tf.variable_scope("Unet_encoder"):
        relu1_1 = new_conv_layer_bn( in_node, [3,3,channels,32], "conv1_1", is_training )
        relu1_2 = new_conv_layer_bn( relu1_1, [3,3,32,32], "conv1_2", is_training )
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                         padding='SAME', name='pool1')
    
        relu2_1 = new_conv_layer_bn(pool1, [3,3,32,64], "conv2_1", is_training)
        relu2_2 = new_conv_layer_bn(relu2_1, [3,3,64,64], "conv2_2", is_training)
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool2')
    
        relu3_1 = new_conv_layer_bn( pool2, [3,3,64,128], "conv3_1", is_training)
        relu3_2 = new_conv_layer_bn( relu3_1, [3,3,128,128], "conv3_2", is_training)
        pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool3')
    
        relu4_1 = new_conv_layer_bn( pool3, [3,3,128,256], "conv4_1", is_training)
        relu4_2 = new_conv_layer_bn( relu4_1, [3,3,256,256], "conv4_2", is_training)
        pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool4')
        
        #bottle neck
        relu5_1 = new_conv_layer_bn( pool4, [3,3,256,512], "conv5_1", is_training)
        relu5_2 = new_conv_layer_bn( relu5_1, [3,3,512,512], "conv5_2", is_training)
    
    output = {}
#    batch_size = in_node.get_shape().as_list()[0]
    pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool5')
    pool5_flat = tf.layers.flatten(pool5)
    
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output
        
    # fc for angle regression
    with tf.variable_scope("Angle_Regression"):  
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, 1], 1, "fc3")
        angle_output = tf.nn.relu( fc3, name='relu3' )
        output['angle_output'] = angle_output
        
    
    with tf.variable_scope("Unet_decoder"):    
        prior_ch = global_prior.get_shape().as_list()[-1]
        prior1 = tf.image.resize_bilinear(global_prior, [32, 32], name='prior1')
        upsample6 = upsampling_layer(relu5_2, 2, "Bilinear", "upsample6")
        relu6_1 = new_conv_layer_bn( tf.concat([relu4_2, upsample6, prior1], axis=-1), [3,3,768+prior_ch,256], "conv6_1", is_training )
        relu6_2 = new_conv_layer_bn( relu6_1, [3,3,256,256], "conv6_2", is_training )
        
        prior2 = tf.image.resize_bilinear(global_prior, [64, 64], name='prior2')
        upsample7 = upsampling_layer(relu6_2, 2, "Bilinear", "upsample7")
        relu7_1 = new_conv_layer_bn( tf.concat([relu3_2, upsample7, prior2], axis=-1), [3,3,384+prior_ch,128], "conv7_1", is_training )
        relu7_2 = new_conv_layer_bn( relu7_1, [3,3,128,128], "conv7_2", is_training )
        
        prior3 = tf.image.resize_bilinear(global_prior, [128, 128], name='prior2')
        upsample8 = upsampling_layer(relu7_2, 2, "Bilinear", "upsample8")
        relu8_1 = new_conv_layer_bn( tf.concat([relu2_2, upsample8, prior3], axis=-1), [3,3,192+prior_ch,64], "conv8_1", is_training )
        relu8_2 = new_conv_layer_bn( relu8_1, [3,3,64,64], "conv8_2", is_training )
        
        upsample9 = upsampling_layer(relu8_2, 2, "Bilinear", "upsample9")
        relu9_1 = new_conv_layer_bn( tf.concat([relu1_2, upsample9, global_prior], axis=-1), [3,3,96+prior_ch,32], "conv9_1", is_training )
        relu9_2 = new_conv_layer_bn( relu9_1, [3,3,32,32], "conv9_2", is_training )
        
        output_map = new_conv_layer_bn( relu9_2, [1,1,32,n_class], "logits", is_training )
    
    layer_dict = {"conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
                  "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
                  "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
                  "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
                  "conv5_1": relu5_1, "conv5_2": relu5_2,
                  "upsample6": upsample6, "conv6_1": relu6_1, "conv6_2": relu6_2,
                  "upsample7": upsample7, "conv7_1": relu7_1, "conv7_2": relu7_2,
                  "upsample8": upsample8, "conv8_1": relu8_1, "conv8_2": relu8_2,
                  "upsample9": upsample9, "conv9_1": relu9_1, "conv9_2": relu9_2,
                  "output": output_map,
                  }
    output['output_map'] = output_map
    return output, layer_dict


def unet( in_node, n_class, is_training=True ):
    """u-net with smaller depth and batch norm"""
    channels = in_node.get_shape().as_list()[-1]
    
    relu1_1 = new_conv_layer_bn( in_node, [3,3,channels,32], "conv1_1", is_training )
    relu1_2 = new_conv_layer_bn( relu1_1, [3,3,32,32], "conv1_2", is_training )
    pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='SAME', name='pool1')

    relu2_1 = new_conv_layer_bn(pool1, [3,3,32,64], "conv2_1", is_training)
    relu2_2 = new_conv_layer_bn(relu2_1, [3,3,64,64], "conv2_2", is_training)
    pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    relu3_1 = new_conv_layer_bn( pool2, [3,3,64,128], "conv3_1", is_training)
    relu3_2 = new_conv_layer_bn( relu3_1, [3,3,128,128], "conv3_2", is_training)
    pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

    relu4_1 = new_conv_layer_bn( pool3, [3,3,128,256], "conv4_1", is_training)
    relu4_2 = new_conv_layer_bn( relu4_1, [3,3,256,256], "conv4_2", is_training)
    pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')
    
    #bottle neck
    relu5_1 = new_conv_layer_bn( pool4, [3,3,256,512], "conv5_1", is_training)
    relu5_2 = new_conv_layer_bn( relu5_1, [3,3,512,512], "conv5_2", is_training)
    
        
    upsample6 = upsampling_layer(relu5_2, 2, "Bilinear", "upsample6")
    relu6_1 = new_conv_layer_bn( tf.concat([relu4_2, upsample6], axis=-1), [3,3,768,256], "conv6_1", is_training )
    relu6_2 = new_conv_layer_bn( relu6_1, [3,3,256,256], "conv6_2", is_training )
    
    upsample7 = upsampling_layer(relu6_2, 2, "Bilinear", "upsample7")
    relu7_1 = new_conv_layer_bn( tf.concat([relu3_2, upsample7], axis=-1), [3,3,384,128], "conv7_1", is_training )
    relu7_2 = new_conv_layer_bn( relu7_1, [3,3,128,128], "conv7_2", is_training )
    
    upsample8 = upsampling_layer(relu7_2, 2, "Bilinear", "upsample8")
    relu8_1 = new_conv_layer_bn( tf.concat([relu2_2, upsample8], axis=-1), [3,3,192,64], "conv8_1", is_training )
    relu8_2 = new_conv_layer_bn( relu8_1, [3,3,64,64], "conv8_2", is_training )
    
    upsample9 = upsampling_layer(relu8_2, 2, "Bilinear", "upsample9")
    relu9_1 = new_conv_layer_bn( tf.concat([relu1_2, upsample9], axis=-1), [3,3,96,32], "conv9_1", is_training )
    relu9_2 = new_conv_layer_bn( relu9_1, [3,3,32,32], "conv9_2", is_training )
    
    output = new_conv_layer_bn( relu9_2, [1,1,32,n_class], "logits", is_training )
    
    layer_dict = {"conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
                  "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
                  "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
                  "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
                  "conv5_1": relu5_1, "conv5_2": relu5_2,
                  "upsample6": upsample6, "conv6_1": relu6_1, "conv6_2": relu6_2,
                  "upsample7": upsample7, "conv7_1": relu7_1, "conv7_2": relu7_2,
                  "upsample8": upsample8, "conv8_1": relu8_1, "conv8_2": relu8_2,
                  "upsample9": upsample9, "conv9_1": relu9_1, "conv9_2": relu9_2,
                  "output": output,
                  }
    return output, layer_dict


def unet_vanilla( in_node, n_class, is_training=True ):
    # TODO: 2X2conv 
    """vanilla u-net"""
    channels = in_node.get_shape().as_list()[-1]
    
    relu1_1 = new_conv_layer_bn( in_node, [3,3,channels,32], "conv1_1" )
    relu1_2 = new_conv_layer_bn( relu1_1, [3,3,32,32], "conv1_2" )
    pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                     padding='SAME', name='pool1')

    relu2_1 = new_conv_layer_bn(pool1, [3,3,32,64], "conv2_1")
    relu2_2 = new_conv_layer_bn(relu2_1, [3,3,64,64], "conv2_2")
    pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool2')

    relu3_1 = new_conv_layer_bn( pool2, [3,3,64,128], "conv3_1")
    relu3_2 = new_conv_layer_bn( relu3_1, [3,3,128,128], "conv3_2")
    pool3 = tf.nn.max_pool(relu3_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool3')

    relu4_1 = new_conv_layer_bn( pool3, [3,3,128,256], "conv4_1")
    relu4_2 = new_conv_layer_bn( relu4_1, [3,3,256,256], "conv4_2")
    pool4 = tf.nn.max_pool(relu4_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool4')
    
    #bottle neck
    relu5_1 = new_conv_layer_bn( pool4, [3,3,256,512], "conv5_1")
    relu5_2 = new_conv_layer_bn( relu5_1, [3,3,512,512], "conv5_2")
    
    
    upsample6 = upsampling_layer(relu5_2, 2, "Bilinear", "upsample6")
    relu6_1 = new_conv_layer_bn( tf.concat([relu4_2, upsample6], axis=-1), [3,3,512,256], "conv6_1" )
    relu6_2 = new_conv_layer_bn( relu6_1, [3,3,256,256], "conv6_2" )
    
    upsample7 = upsampling_layer(relu6_2, 2, "Bilinear", "upsample7")
    relu7_1 = new_conv_layer_bn( tf.concat([relu3_2, upsample7], axis=-1), [3,3,256,128], "conv7_1" )
    relu7_2 = new_conv_layer_bn( relu7_1, [3,3,128,128], "conv7_2" )
    
    upsample8 = upsampling_layer(relu7_2, 2, "Bilinear", "upsample8")
    relu8_1 = new_conv_layer_bn( tf.concat([relu2_2, upsample8], axis=-1), [3,3,128,64], "conv8_1" )
    relu8_2 = new_conv_layer_bn( relu8_1, [3,3,64,64], "conv8_2" )
    
    upsample9 = upsampling_layer(relu8_2, 2, "Bilinear", "upsample9")
    relu9_1 = new_conv_layer_bn( tf.concat([relu1_2, upsample9], axis=-1), [3,3,64,32], "conv9_1" )
    relu9_2 = new_conv_layer_bn( relu9_1, [3,3,32,32], "conv9_2" )
    
    output = new_conv_layer_bn( relu9_2, [1,1,32,n_class], "logits" )
    
    layer_dict = {"conv1_1": relu1_1, "conv1_2": relu1_2, "pool1": pool1, 
                  "conv2_1": relu2_1, "conv2_2": relu2_2, "pool2": pool2,
                  "conv3_1": relu3_1, "conv3_2": relu3_2, "pool3": pool3,
                  "conv4_1": relu4_1, "conv4_2": relu4_2, "pool4": pool4,
                  "conv5_1": relu5_1, "conv5_2": relu5_2,
                  "upsample6": upsample6, "conv6_1": relu6_1, "conv6_2": relu6_2,
                  "upsample7": upsample7, "conv7_1": relu7_1, "conv7_2": relu7_2,
                  "upsample8": upsample8, "conv8_1": relu8_1, "conv8_2": relu8_2,
                  "upsample9": upsample9, "conv9_1": relu9_1, "conv9_2": relu9_2,
                  "output": output,
                  }
    return output, relu5_2, layer_dict

   
def unet_multi_task_fine_newz( in_node, n_class, z_class, batch_size, is_training=True ):
    """u-net with smaller depth and batch norm"""    
    # TODO: relu??
    # TODO: varibles, flag
    
    # U-net
    output = {}
#    batch_size = in_node.get_shape().as_list()[0]
    with tf.variable_scope("U-net"):
        output_map, relu5_2, layer_dict = unet(in_node, n_class, is_training)
        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')
        pool5_flat = tf.layers.flatten(pool5)
        output['output_map'] = output_map
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, n_class*(z_class+1)], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        z_output = tf.reshape(z_output, [batch_size, n_class, z_class+1])
        output['z_output'] = z_output
        
    # fc for angle regression
    with tf.variable_scope("Angle_Regression"):  
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, 1], 1, "fc3")
        angle_output = tf.nn.relu( fc3, name='relu3' )
        output['angle_output'] = angle_output

    return output, layer_dict


def unet_multi_task_PriorMultiplyLogits( in_node, global_prior, n_class, z_class, is_training=True ):
    # U-net
    output = {}
    with tf.variable_scope("U-net"):
        logits, relu5_2, layer_dict = unet(in_node, n_class, is_training)
        pool5 = tf.nn.max_pool(relu5_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool5')
        pool5_flat = tf.layers.flatten(pool5)
        
    # fc for z_axis classification
    with tf.variable_scope("Z_Classification"):
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, z_class], 1, "fc3")
        z_output = tf.nn.relu( fc3, name='relu3' )
        output['z_output'] = z_output
        
    # fc for angle regression
    with tf.variable_scope("Angle_Regression"):  
        fc1 = new_fc_layer( pool5_flat, [pool5_flat.get_shape().as_list()[1], 2048], 1, "fc1")
        relu1 = tf.nn.relu( fc1, name='relu1' )
        fc2 = new_fc_layer( relu1, [2048, 500], 1, "fc2")
        relu2 = tf.nn.relu( fc2, name='relu2' )
        fc3 = new_fc_layer( relu2, [500, 1], 1, "fc3")
        angle_output = tf.nn.relu( fc3, name='relu3' )
        output['angle_output'] = angle_output
    
    # multiply global prior with logits    
    output['output_map'] = tf.multiply(logits, global_prior)
    return output, layer_dict