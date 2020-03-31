'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import math
import tensorflow as tf
from core import utils
fc_layer = utils.fc_layer
conv2d = utils.conv2d
GCN = utils.GCN


def refinement_network(features,
                       guidance,
                       output_stride,
                       batch_size,
                       layers_dict,
                       num_class=None,
                       embed=32,
                       further_attention=False,
                       class_split_in_each_stage=False,
                       input_guidance_in_each_stage=False,
                       is_training=None,
                       scope=None):
    """
    """
    # TODO: necessary for using batch_size??
    # TODO: check guidance shape. The shape [?,256,256,1] should cause error
    num_stage = len(layers_dict)
    num_up = math.sqrt(output_stride)
    num_down = num_stage - num_up
    upsample_flags = num_down*[False] + num_up*[True]
        
    with tf.variable_scope(scope, 'refinement_network') as sc:
        guidance_in = guidance
        batch_size, h, w = layers_dict["low_level5"].get_shape().as_list()[1:3]
        guidance_in_lowres = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in_lowres')
        zero_tensor = tf.zeros([batch_size, h, w, embed])
            
        layers_dict["guidance_in"] = guidance_in
        feature, guidance = rm_class(in_node=layers_dict["low_level5"],
                                    feature=zero_tensor,
                                    guidance=guidance_in_lowres,
                                    upsample=upsample_flags[0],
                                    scope='RM_5',
                                    is_training=is_training)
        layers_dict["guidance1"] = guidance
        layers_dict["feature1"] = feature
        
        guidance = tf.nn.softmax(guidance, axis=3)
        # TODO: got to be a better way
        if input_guidance_in_each_stage:
            guidance = guidance + guidance_in_lowres
            
                
        for stage in range(num_stage-1, -1, 0):
            if class_split_in_each_stage:
                feature, guidance = rm_class(in_node=layers_dict["low_level"+str(stage)],
                                            feature=feature,
                                            guidance=guidance,
                                            upsample=upsample_flags[num_stage-stage],
                                            further_attention=further_attention,
                                            scope='RM_'+str(stage),
                                            is_training=is_training)
            else:
                feature, guidance = rm(in_node=layers_dict["low_level"+str(stage)],
                                        feature=feature,
                                        guidance=feature,
                                        num_class=num_class,
                                        upsample=upsample_flags[num_stage-stage],
                                        further_attention=further_attention,
                                        scope='RM_'+str(stage),
                                        is_training=is_training)        
            
            
            layers_dict["guidance"+str(num_stage+1-stage)] = guidance
            layers_dict["feature"+str(num_stage+1-stage)] = feature

            guidance = tf.nn.softmax(guidance, axis=3)
            if input_guidance_in_each_stage:
                h, w = layers_dict["low_level"+str(stage)].get_shape().as_list()[1:3]
                cue = tf.image.resize_bilinear(guidance_in, [h, w])
                guidance = guidance + cue
                 
    return guidance, layers_dict


def rm_class(in_node,
             feature,
             guidance,
             upsample=True,
             further_attention=False,
             is_training=None,
             scope=None):
    """refining module
    feature: [H,W,Channels]
    guidance: [H,W,Class]
    """
    with tf.variable_scope(scope, 'rm_class'):
        h, w = in_node.get_shape().as_list()[1:3]
        num_class = guidance.get_shape().as_list()[3]
        num_filters = feature.get_shape().as_list()[3]
        
        # feature embedding
        conv_r1 = GCN(in_node, num_filters, ks=7, is_training=is_training)
        
        # sram attention
        def sram_attention_branch(sram_input):
            feature_list = []
            f = num_filters*(num_class+1)
            feature_list.append(feature)
                
            for c in range(num_class):
                if further_attention: scope='sram2_'+str(c) 
                else: scope='sram1_'+str(c)
                attention_class = sram(sram_input, guidance[...,c:c+1], scope=scope, is_training=is_training)
                feature_list.append(attention_class)
                
            attention = conv2d(tf.concat(feature_list, axis=3), [1,1,f,num_filters], 
                               activate=tf.nn.relu, scope="fuse", is_training=is_training) 
            return attention
        
        attention = sram_attention_branch(sram_input=conv_r1)
        if further_attention:
            attention = sram_attention_branch(sram_input=attention)
        
        new_guidance = conv2d(attention, [1,1,num_filters,num_class], 
                              activate=None, scope="guidance", is_training=is_training)
        if upsample:
            new_feature = tf.image.resize_bilinear(attention, [2*h, 2*w], name='new_feature')
            new_guidance = tf.image.resize_bilinear(new_guidance, [2*h, 2*w], name='new_guidance')
        
    return new_feature, new_guidance
    

def rm(in_node,
        feature,
        guidance,
        num_class=14,
        upsample=True,
        further_attention=False,
        is_training=None,
        scope=None):
    """refining module
    feature: [H,W,Channels]
    guidance: [H,W,Channels]
    """
    with tf.variable_scope(scope, 'rm'):
        h, w = in_node.get_shape().as_list()[1:3]
        num_filters = feature.get_shape().as_list()[3]
        
        # feature embedding
        conv_r1 = GCN(in_node, num_filters, ks=7, is_training=is_training)
        
        # sram attention
        attention = sram(in_node, guidance, scope='sram1', is_training=is_training)
        attention = conv2d(tf.concat([attention,conv_r1], axis=3), [1,1,2*num_filters,num_filters], 
                           activate=tf.nn.relu, scope="fuse", is_training=is_training) 
        if further_attention:
            attention = sram(attention, guidance, scope='sram2', is_training=is_training)
         
        new_guidance = conv2d(attention, [1,1,num_filters,num_class], 
                              activate=None, scope="guidance", is_training=is_training)
        if upsample:
            new_feature = tf.image.resize_bilinear(attention, [2*h, 2*w], name='new_feature')
            new_guidance = tf.image.resize_bilinear(new_guidance, [2*h, 2*w], name='new_guidance')
        
    return new_feature, new_guidance


def sram(in_node,
         guidance,
         is_gamma=False,
         scope=None,
         is_training=True):
    """Single Residual Attention Module"""
    with tf.variable_scope(scope, "sram"):
        channels = in_node.get_shape().as_list()[3]
        conv1 = conv2d(in_node, [3,3,channels,channels], activate=tf.nn.relu, scope="conv1", is_training=is_training)
        conv2 = conv2d(conv1, [3,3,channels,channels], activate=tf.nn.relu, scope="conv2", is_training=is_training)
        
        guidance_filters = guidance.get_shape().as_list()[3]
        if guidance_filters == 1:
            guidance_tile = tf.tile(guidance, [1,1,1,channels])
        elif guidance_filters == channels:
            guidance_tile = guidance
        else:
            raise ValueError("Unknown guidance filters number")

        if is_gamma:
            gamma = tf.Variable(0, dtype=tf.float32)
            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
        else:
            output = in_node + tf.multiply(conv2, guidance_tile)

        tf.add_to_collection(scope+"/sram_embed", [in_node, conv1, conv2, output])
        return output