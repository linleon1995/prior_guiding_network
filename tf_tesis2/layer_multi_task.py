# tf_unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# tf_unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tf_unet.  If not, see <http://www.gnu.org/licenses/>.


'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import numpy as np
# from tf_tesis2.module import (nonlocal_dot)
from tf_tesis2.utils import (conv2d, global_avg_pool, fc_layer)




def rm(in_node,
       feature,
       guidance,
       end_points=None,
       num_filters=32,
       classifier=False,
       upsample=True,
       further_attention=False,
       class_features=False,
       low_high_fusion='later',
       each_class_fusion='concat_and_conv',
       name='rm',
       is_training=True,
       ):
    """refining module"""
    with tf.variable_scope(name):
        in_node_shape = in_node.get_shape().as_list()
        s = in_node_shape[1]
        channels = in_node_shape[3]
        num_class = guidance.get_shape().as_list()[3]
        
        # feature embedding
        conv_r1 = conv2d(in_node, [1,7,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
        conv_r1 = conv2d(conv_r1, [7,1,num_filters,num_filters], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training)
        
        feature_list = []
        for i in range(num_class):
            with tf.variable_scope('SRAM_'+str(i)):
                # TODO: further_attention; add second SRAM
                attention1 = sram(conv_r1, guidance[...,i:i+1], scope='attention1', is_training=is_training)
                
                # low-level, high-level feature fusion
                if class_features:
                    if low_high_fusion == 'summation':
                        fusion = tf.add(attention1, feature[...,num_filters*i:num_filters*(i+1)])
                    elif low_high_fusion == 'concat_and_conv':
                        fusion = tf.concat([attention1, feature[...,num_filters*i:num_filters*(i+1)]], axis=3)
                        fusion = conv2d(fusion, [3,3,num_filters*2,num_filters], activate=tf.nn.relu, scope="fusion", is_training=is_training)       
                else:
                    if low_high_fusion == 'summation':
                        fusion = tf.add(attention1, feature)
                    elif low_high_fusion == 'concat_and_conv':
                        fusion = tf.concat([attention1, feature], axis=3)
                        fusion = conv2d(fusion, [3,3,num_filters*2,num_filters], activate=tf.nn.relu, scope="fusion", is_training=is_training)
                    elif low_high_fusion == 'later':
                        fusion = attention1
                        
                feature_list.append(fusion)
        if low_high_fusion == 'later':
            feature_list.append(feature)
        
        # feature fusion in different class
        if class_features:
            new_feature = tf.concat(feature_list, axis=3)
        else:
            if each_class_fusion == 'concat_and_conv':
                new_feature = tf.concat(feature_list, axis=3)
                if low_high_fusion == 'later':
                    new_feature = conv2d(new_feature, [3,3,num_filters*(num_class+1),num_filters], activate=tf.nn.relu, scope="class_fusion", is_training=is_training)
                else:
                    new_feature = conv2d(new_feature, [3,3,num_filters*num_class,num_filters], activate=tf.nn.relu, scope="class_fusion", is_training=is_training)
            elif each_class_fusion == 'summation':
                new_feature = tf.add_n(feature_list)
                
        # Final output, get guidance and feature in next stage
        if classifier:
            kernel_size = 1 
            activate_func=None
        else:
            kernel_size = 3
            activate_func=tf.nn.relu
        
        if upsample:
#            new_guidance = tf.image.resize_bilinear(new_guidance, [2*s, 2*s], name='new_guidance')
            new_feature = tf.image.resize_bilinear(new_feature, [2*s, 2*s], name='new_feature')
            
        output_filters = new_feature.get_shape().as_list()[3]
        new_guidance = conv2d(new_feature, [kernel_size,kernel_size,output_filters,num_class], activate=activate_func, scope="new_feature", is_training=is_training)
        
    if end_points is not None:            
        return new_feature, new_guidance, end_points
    else:
        return new_feature, new_guidance
        
    

def sram(in_node,
         guidance,
         is_gamma=False,
         scope="sram",
         is_training=True,         
         ):
    """Single Residual Attention Module"""
    with tf.variable_scope(scope):
        channels = in_node.get_shape().as_list()[3]
        conv1 = conv2d(in_node, [3,3,channels,channels], activate=tf.nn.relu, scope="conv1", is_training=is_training)     
        conv2 = conv2d(conv1, [3,3,channels,channels], activate=tf.nn.relu, scope="conv2", is_training=is_training)
        
        guidance_tile = tf.tile(guidance, [1,1,1,channels])
        
        if is_gamma:
            gamma = tf.Variable(0, dtype=tf.float32)
            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
        else:
            output = in_node + tf.multiply(conv2, guidance_tile)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
    return output


def SRAM(in_node, guidance, is_gamma=False, name='SRAM', is_training=True):
    with tf.variable_scope(name):
        batch_size = in_node.get_shape().as_list()[0]
        channels = in_node.get_shape().as_list()[-1]

        conv1 = conv2d( in_node, [3,3,channels,channels], activate=tf.nn.relu, scope="conv1", is_training=is_training )
        conv2 = conv2d( conv1, [3,3,channels,channels], activate=tf.nn.relu, scope="conv2", is_training=is_training )
        
        guidance_tile = tf.tile(guidance, [1,1,1,channels])
        
        if is_gamma:
            gamma = tf.Variable(0, dtype=tf.float32)
            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
        else:
            output = in_node + tf.multiply(conv2, guidance_tile)
            
#        # feature select
#        important_bar = global_avg_pool(output, keep_dims=False)
#        fc1 = fc_layer(inputs=important_bar,
#                     layer_size=[channels,channels],
#                     _std=1,
#                     scope='fc1')
#        fc1 = tf.expand_dims(tf.expand_dims(fc1, axis=1), axis=1)
#        fc1 = tf.nn.sigmoid(fc1)
#        output = tf.multiply(output, fc1)
    return output, [conv1,conv2]




def RM(in_node, feature, guidance, activate=None, out_filter=32, classifier=False, upsample=True, name='Refining_Module', is_training=True):
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
       
        fusion = tf.add(conv_r1, feature)
        
        ###
        f2=[]
        for i in range(n_class):
            if not is_training:
                attention2, conv = SRAM(fusion, guidance[...,i:i+1], name='SRAM2_'+str(i), is_training=is_training)
            else:
                attention2, _ = SRAM(fusion, guidance[...,i:i+1], name='SRAM2_'+str(i), is_training=is_training)
            f2.append(attention2)
        
        new_feature = tf.add_n(f2)

        # new guidance
        if classifier:
            new_guidance = conv2d(new_feature, [1,1,out_filter,n_class], scope="fusion", is_training=is_training )
        else:
            new_guidance = conv2d(new_feature, [3,3,out_filter,n_class], scope="fusion", is_training=is_training )
        
        if upsample:
            new_feature = tf.image.resize_bilinear(new_feature, [2*s, 2*s], name='new_feature')        
            new_guidance = tf.image.resize_bilinear(new_guidance, [2*s, 2*s], name='new_guidance')
    return new_feature, new_guidance, f, vis


#def RM(in_node, feature, guidance, activate=None, out_filter=32, classifier=False, upsample=True, name='Refining_Module', is_training=True):
#    with tf.variable_scope(name):
#        in_node_shape = in_node.get_shape().as_list()
#        s = in_node_shape[1]
#        channels = in_node_shape[-1]
#        n_class = guidance.get_shape().as_list()[-1]
#        
#        # feature embedding
#        conv_r1 = conv2d( in_node, [1,7,channels,out_filter], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training )
#        conv_r1 = conv2d( conv_r1, [7,1,out_filter,out_filter], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training )
#        f=[]
#        vis=[]
#        
#        # template guiding
#        for i in range(n_class):
#            if not is_training:
#                attention1, conv = SRAM(conv_r1, guidance[...,i:i+1], name='SRAM_'+str(i), is_training=is_training)
#            else:
#                attention1, _ = SRAM(conv_r1, guidance[...,i:i+1], name='SRAM_'+str(i), is_training=is_training)
#            if not is_training:
#                vis.append(conv)
#                vis.append(attention1)  
#            f.append(attention1)
#        if not is_training:
#            vis.append(conv_r1)
#
##        
##        # new feature
#        embedding = tf.add_n(f)
#        fusion = tf.add(embedding, feature)
#        
#        ###
#        f2=[]
#        for i in range(n_class):
#            if not is_training:
#                attention2, conv = SRAM(fusion, guidance[...,i:i+1], name='SRAM2_'+str(i), is_training=is_training)
#            else:
#                attention2, _ = SRAM(fusion, guidance[...,i:i+1], name='SRAM2_'+str(i), is_training=is_training)
#            f2.append(attention2)
#        
#        new_feature = tf.add_n(f2)
#
#        # new guidance
#        if classifier:
#            new_guidance = conv2d(new_feature, [1,1,out_filter,n_class], scope="fusion", is_training=is_training )
#        else:
#            new_guidance = conv2d(new_feature, [3,3,out_filter,n_class], scope="fusion", is_training=is_training )
#        
#        if upsample:
#            new_feature = tf.image.resize_bilinear(new_feature, [2*s, 2*s], name='new_feature')        
#            new_guidance = tf.image.resize_bilinear(new_guidance, [2*s, 2*s], name='new_guidance')
#    return new_feature, new_guidance, f, vis


def RM_com(in_node, feature, guidance, com_f=None, activate=None, out_filter=32, classifier=False, name='Refining_Module', is_training=True):
    with tf.variable_scope(name):
        if com_f is not None:
            in_node = tf.concat([in_node, com_f], axis=-1)
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
            fusion = tf.add(attention1, feature)
            f.append(fusion)
            if not is_training:
                vis.append(conv)
                vis.append(attention1)
                vis.append(fusion)
        if not is_training:
            vis.append(conv_r1)
        all_guid_f = tf.concat(f, -1)
        all_guid_f = tf.image.resize_bilinear(all_guid_f, [2*s, 2*s], name='upsampling')
              
        # new feature
        new_feature = conv2d(all_guid_f, [3,3,out_filter*n_class,out_filter], activate=tf.nn.relu, scope="new_feature", is_training=is_training )

        # new guidance
        if classifier:
            new_guidance = conv2d(all_guid_f, [1,1,out_filter*n_class,n_class], scope="new_guidance", is_training=is_training )
        else:
            new_guidance = conv2d(all_guid_f, [3,3,out_filter*n_class,n_class], scope="new_guidance", is_training=is_training )
#        print(tf.shape(conv_r1), 's', conv_r1.shape, 'sss', conv_r1.shape[1])
        com_f_up = tf.image.resize_bilinear(conv_r1, [2*tf.shape(conv_r1)[1], 2*tf.shape(conv_r1)[2]], name='com_f_up')
    return new_feature, new_guidance, com_f_up, f, vis


""""seperate RM using old layer"""
#def SRAM(in_node, guidance, is_gamma=False, name='SRAM', is_training=True):
#    with tf.variable_scope(name):
#        channels = in_node.get_shape().as_list()[-1]
#
#        conv1 = new_conv_layer_bn( in_node, [3,3,channels,channels], "conv1", is_training )
#        conv2 = new_conv_layer_bn( conv1, [3,3,channels,channels], "conv2", is_training )
#        
#        guidance_tile = tf.tile(guidance, [1,1,1,channels])
#        
#        if is_gamma:
#            gamma = tf.Variable(0, dtype=tf.float32)
#            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
#        else:
#            output = in_node + tf.multiply(conv2, guidance_tile)
#
#    return output, [conv1,conv2]
#
#
#def RM(in_node, feature, guidance, activate=None, out_filter=32, classifier=False, name='Refining_Module', is_training=True):
#    with tf.variable_scope(name):
#        in_node_shape = in_node.get_shape().as_list()
#        s = in_node_shape[1]
#        channels = in_node_shape[-1]
#        n_class = guidance.get_shape().as_list()[-1]
#        
#        conv_r1 = new_conv_layer_bn( in_node, [1,7,channels,out_filter], "conv_r1_1", is_training )
#        conv_r1 = new_conv_layer_bn( conv_r1, [7,1,out_filter,out_filter], "conv_r1_2", is_training )
#        f=[]
#        vis=[]
#        
#        for i in range(14):
#            attention1, conv = SRAM(conv_r1, guidance[...,i:i+1], True, 'SRAM_'+str(i), is_training)
#            fusion = tf.add(attention1, feature)
#            f.append(fusion)
#            vis.append(conv)
#            vis.append(attention1)
#            vis.append(fusion)
#        vis.append(conv_r1)
##        attention2 = SRAM(fusion, guidance, 'SRAM_2', is_training)
#        attention2 = tf.concat(f, -1)
#        new_feature = tf.image.resize_bilinear(attention2, [2*s, 2*s], name='new_feature')
##        new_feature = tf.add_n(f)
#        new_feature = new_conv_layer_bn(new_feature, [3,3,out_filter*n_class,out_filter], "feature_compact", is_training )
#        
#        new_guidance = tf.image.resize_bilinear(attention2, [2*s, 2*s], name='new_guidance')
#        if classifier:
#            new_guidance = new_conv_layer_bn(new_guidance, [1,1,out_filter*n_class,n_class], "conv_r2", is_training )
#        else:
#            new_guidance = new_conv_layer_bn(new_guidance, [3,3,out_filter*n_class,n_class], "conv_r2", is_training )
#            
#        if activate is not None:    
#            new_guidance = activate(new_guidance)
#        
#    return new_feature, new_guidance, f, vis




def new_fc_layer( bottom, layer_size, _std, name ):
    input_size, output_size = layer_size
    shape = tf.shape(bottom)
    dim = tf.reduce_prod( shape[1:] )
    x = tf.reshape(bottom, [-1, dim])

    with tf.variable_scope(name) as scope:
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=[input_size, output_size], stddev=_std))
        w = w / tf.sqrt(input_size/2)
        b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[output_size]))
        fc = tf.nn.bias_add( tf.matmul(x, w), b, name='fc')
    return fc
    

def new_conv_layer( bottom, filter_shape, name ):
    features = filter_shape[-1]
    _std = np.sqrt(2 / (filter_shape[0]**2 * features))
    
    with tf.variable_scope( name ) as scope:  
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=filter_shape, stddev=_std))
        b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[filter_shape[-1]]))

        conv = tf.nn.conv2d( bottom, w, [1,1,1,1], padding='SAME')
        bias = tf.nn.bias_add(conv, b)
        relu = tf.nn.relu( bias, name='relu' )
    return relu #relu


def new_conv_layer_bn( bottom, filter_shape, name, is_training=False ):
    features = filter_shape[-1]
    _std = np.sqrt(2 / (filter_shape[0]**2 * features))
    
    with tf.variable_scope( name, reuse=tf.AUTO_REUSE ) as scope:
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=filter_shape, stddev=_std))

        conv = tf.nn.conv2d( bottom, w, [1,1,1,1], padding='SAME')
        bn = batch_norm(conv, is_training=is_training, name=name+'_bn')
        relu = tf.nn.relu( bn, name='relu' )
    return relu #relu
    
def new_depthsep_conv(in_node, filter_shape, rate=[1,1], channel_multiplier=1, activate_func=tf.nn.relu, 
                      bn_flag=True, is_training=True, name=None):
    assert type(name) == str
    
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    features = filter_shape[-1]
    _std = np.sqrt(2 / (filter_shape[0]**2 * features))
    with tf.variable_scope( name ):
        w_depth = tf.get_variable(
                "W_depth",
                initializer=tf.truncated_normal(shape=[kernel_h, kernel_w, in_channels, channel_multiplier], stddev=_std))
        w_point = tf.get_variable(
                "W_point",
                initializer=tf.truncated_normal(shape=[1,1, channel_multiplier*in_channels, out_channels], stddev=_std))
        conv = tf.nn.separable_conv2d(
                input=in_node,
                depthwise_filter=w_depth,
                pointwise_filter=w_point,
                strides=[1,1,1,1],
                padding='SAME',
                rate=rate,
                name=name,
                )
        if bn_flag:
            out_node = batch_norm(conv, is_training=is_training, name=name+'_bn')
        else:
            b = tf.get_variable(
                "b",
                initializer=tf.constant(shape=out_channels, stddev=0.1))
            out_node = tf.nn.bias_add(conv, b)
        output = activate_func(out_node)    
    return output
        
    
def upsampling_layer(x, n, mode, name):
    x_shape = tf.shape(x)

    new_h = x_shape[1]*n
    new_w = x_shape[2]*n
    if mode is 'Nearest_Neighbor':
        x_up = tf.image.resize_nearest_neighbor(x, [new_h, new_w], name=name)
        
    if mode is 'Bilinear':
        x_up = tf.image.resize_bilinear(x, [new_h, new_w], name=name)
    
    if mode is 'Bicubic':
        x_up = tf.image.resize_bicubic(x, [new_h, new_w], name=name)

    return x_up


#def fc_layer(x, W, b):
#    x = tf.matmul(x, W)
#    x = tf.add(x, b)
#    return x
    

def batch_norm(x, name, is_training):
    # BN for the first input
    x = tf.layers.batch_normalization(x, training=is_training)
    return x


def vertical_fold(inputs, flip_flag=True):
    batch_size, h, w, channels = inputs.get_shape().as_list()
    upper = inputs[:,0:h//2]
    lower = inputs[:,h//2:]
    if flip_flag:
        lower = lower[:,::-1]
    outputs = tf.concat([upper,lower], axis=-1)
    return outputs


def horizontal_fold(inputs, flip_flag=True):
    batch_size, h, w, channels = inputs.get_shape().as_list()
    left = inputs[:,:,0:w//2]
    right = inputs[:,:,w//2:]
    if flip_flag:
        right = right[:,:,::-1]
    outputs = tf.concat([left, right], axis=-1)
    return outputs

