#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 00:01:26 2019

@author: acm528_02
"""

import tensorflow as tf
import numpy as np
from tf_tesis2.utils import (fc_layer, conv2d, atrous_conv2d, split_separable_conv2d, batch_norm, upsampling_layer)


   
def SRAM(in_node, guidance, is_gamma=False, name='SRAM', is_training=True):
    with tf.variable_scope(name):
        channels = in_node.get_shape().as_list()[-1]

        conv1 = conv2d( in_node, [3,3,channels,channels], activate=tf.nn.relu, scope="conv1", is_training=is_training )
        conv2 = conv2d( conv1, [3,3,channels,channels], activate=tf.nn.relu, scope="conv2", is_training=is_training )
        
        guidance_tile = tf.tile(guidance, [1,1,1,channels])
        
        if is_gamma:
            gamma = tf.Variable(0, dtype=tf.float32)
            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
        else:
            output = in_node + tf.multiply(conv2, guidance_tile)

    return output, [conv1,conv2]


def RM(in_node, 
                    feature, 
                    guidance, 
                    activate=None, 
                    out_filter=32, 
                    classifier=False, 
                    upsample=True, 
                    name='Refining_Module', 
                    is_training=True):
    """refining module
    """
    with tf.variable_scope(name):
        in_node_shape = in_node.get_shape().as_list()
        s = in_node_shape[1]
        channels = in_node_shape[-1]
        n_class = guidance.get_shape().as_list()[-1]
        
        # feature embedding
        conv_r1 = conv2d( in_node, [1,7,channels,out_filter], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training )
        conv_r1 = conv2d( conv_r1, [7,1,out_filter,out_filter], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training )
        f=[]
        vis=[]
        
        # template guiding
        for i in range(n_class):
            if not is_training:
                attention1, conv = SRAM(conv_r1, guidance[...,i:i+1], name='SRAM_'+str(i), is_training=is_training)
            else:
                attention1, _ = SRAM(conv_r1, guidance[...,i:i+1], name='SRAM_'+str(i), is_training=is_training)
            if not is_training:
                vis.append(conv)
                vis.append(attention1)  
            f.append(attention1)
        if not is_training:
            vis.append(conv_r1)
        all_guid_f = tf.concat(f, -1)
        
        # new feature
        embedding = conv2d(all_guid_f, [3,3,out_filter*n_class,out_filter], activate=tf.nn.relu, scope="embedding", is_training=is_training )
        fusion = tf.add(embedding, feature)
        
        ###
        f2=[]
        for i in range(n_class):
            if not is_training:
                attention1, conv = SRAM(fusion, guidance[...,i:i+1], name='SRAM2_'+str(i), is_training=is_training)
            else:
                attention1, _ = SRAM(fusion, guidance[...,i:i+1], name='SRAM2_'+str(i), is_training=is_training)
            f2.append(attention1)
        all_guid_f2 = tf.concat(f2, -1)     
        new_feature = conv2d(all_guid_f2, [3,3,out_filter*n_class,out_filter], activate=tf.nn.relu, scope="embedding2", is_training=is_training )

       ###

        # new guidance
        if classifier:
            new_guidance = conv2d(new_feature, [1,1,out_filter,n_class], scope="fusion", is_training=is_training )
        else:
            new_guidance = conv2d(new_feature, [3,3,out_filter,n_class], scope="fusion", is_training=is_training )
        
        if upsample:
            new_feature = tf.image.resize_bilinear(new_feature, [2*s, 2*s], name='new_feature')        
            new_guidance = tf.image.resize_bilinear(new_guidance, [2*s, 2*s], name='new_guidance')
    return new_feature, new_guidance, f, vis