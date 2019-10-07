#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:15:42 2019

@author: acm528_02
"""

from __future__ import print_function, division, absolute_import, unicode_literals
from tf_tesis2.utils import (fc_layer, conv2d, split_separable_conv2d, batch_norm, upsampling_layer, global_avg_pool, global_max_pool)
from tf_tesis2.cell import ConvLSTMCell, ConvGRUCell
import tensorflow as tf
import numpy as np


def bidirectional_GRU(features, batch_size, nx, ny, n_class, seq_length=3, is_training=True):
    filters_num = features[0].get_shape().as_list()[3]
    with tf.variable_scope("BiConvGRU") as scope:
        with tf.variable_scope("forward_cell") as scope:
            cell_forward = ConvGRUCell(shape=[ny, nx],
                                       filters=filters_num, 
                                       kernel=[3, 3], 
                                       normalize=False,
#                                       activation=tf.nn.softmax, 
                                       )
    
        with tf.variable_scope("backward_cell") as scope:
            cell_backward = ConvGRUCell(shape=[ny, nx],
                                        filters=filters_num, 
                                        kernel=[3, 3],  
                                        normalize=False,
#                                        activation=tf.nn.softmax, 
                                        )

        outputs, outputs_fw, outputs_bw = tf.nn.static_bidirectional_rnn(
                                    cell_fw=cell_forward,
                                    cell_bw=cell_backward,
                                    inputs=features,
                                    initial_state_fw=cell_forward.zero_state(batch_size, dtype=tf.float32),
                                    initial_state_bw=cell_backward.zero_state(batch_size, dtype=tf.float32),
                                    dtype=tf.float32,
                                )
#        outputs = tf.add(outputs[seq_length//2][:,0:ny], outputs[seq_length//2][:,ny:])
#        outputs = outputs / 2
#        states = tf.concat((outputs[seq_length//2][:,0:ny], outputs[seq_length//2][:,ny:]), -1)
#        outputs = conv2d(states, [1,1,filters_num*2,filters_num], scope='outputs', is_training=is_training)
        a=[]
        for i in range(seq_length):
            with tf.variable_scope("rnn_outputs", reuse=tf.AUTO_REUSE):
                states = tf.concat((outputs[i][:,0:ny], outputs[i][:,ny:]), -1)
                o = conv2d(states, [1,1,filters_num*2,filters_num], scope='outputs', is_training=is_training)
            a.append(o)
        outputs = tf.concat(a, 0)
    return outputs


def nonlocal_dot(inputs, key=None, in_dims=64, raeduced_ratio=8, softmax=True, at_map=False, gamma_flag=True, is_training=True, name='NonLocal'):
    # TODO: gamma
    # TODO: modify with new conv2d format, e.g., avtivation function
    with tf.variable_scope(name):
        if key is not None:
            proj_key = conv2d(key, [1,1,in_dims,in_dims//raeduced_ratio], bn_flag=False, scope='key_conv')
        else:
            proj_key = conv2d(inputs, [1,1,in_dims,in_dims//raeduced_ratio], bn_flag=False, scope='key_conv')
        proj_query = conv2d(inputs, [1,1,in_dims,in_dims//raeduced_ratio], bn_flag=False, scope='query_conv')
        value = proj_value = conv2d(inputs, [1,1,in_dims,in_dims], bn_flag=False, scope='value_conv')
        
        key_flat = tf.reshape(proj_key, [tf.shape(proj_key)[0], -1, tf.shape(proj_key)[-1]])
        query_flat = tf.reshape(proj_query, [tf.shape(proj_query)[0], -1, tf.shape(proj_query)[-1]])
        value_flat = tf.reshape(proj_value, [tf.shape(proj_value)[0], -1, tf.shape(proj_value)[-1]])
    
        f = tf.matmul(key_flat, tf.transpose(query_flat, [0, 2, 1]))
        if softmax:
            f = tf.nn.softmax(f)
        else:
            f = f / tf.cast(tf.shape(f)[-1], tf.float32)
            
        # Compute f * g ("self-attention") -> (B,HW,C)
        fg = tf.matmul(f, value_flat)
        # Expand and fix the static shapes TF lost track of.
        fg = tf.reshape(fg, tf.shape(value))
        
        if gamma_flag:
            gamma = tf.Variable(0)
            out = inputs + gamma * fg
        else:
            out = inputs + fg
            
        if at_map:
            return out, fg
        else:
            return out

              
def ca_weight(query, key):
    
#    d = 
    attention_map = tf.nn.softmax(d, -1)
    return attention_map

def ca_map(attention, v):
    pass
    return cc_attention

def cca(inputs, in_dims, is_training=True, name='CrissCrossAttention'):
    with tf.variable_scope(name):
        proj_key = conv2d(inputs, [1,1,in_dims,in_dims//8], is_training, scope='key_conv')
        proj_query = conv2d(inputs, [1,1,in_dims,in_dims//8], is_training, scope='query_conv')
        proj_value = conv2d(inputs, [1,1,in_dims,in_dims], is_training, scope='value_conv')
        
        energy = ca_weight(proj_query, proj_key)
        attention = tf.nn.softmax(energy, -1)
        out = ca_map(attention, proj_value)
        gamma = tf.Variable(0)
        out = gamma*out + inputs
    return out

def rcca(inputs, in_dims, is_training=True, times=2):
    for step in range(times):
        inputs = cca(inputs, in_dims, is_training, 'CrissCrossAttention'+str(step))
    return inputs

def region_rcca():
    pass
        
def coam(target, reference):
    pass
#    overlap = tf.multiply(target, reference)
    
def COAM(target, reference, x_grid, y_grid, max_avg_flag, pairwise_func):
    """cross objects attetntion module"""
    assert target.get_shape().ndims == reference.get_shape().ndims == 4
    n, h, w, c = target.get_shape().as_list()
    # get grid
    if x_grid is not None and y_grid is not None:
        'grid'
        x = tf.linspace(-1.0, 1.0, w)
        y = tf.linspace(-1.0, 1.0, h)
        x_t, y_t = tf.meshgrid(x, y)
        x_grid = tf.reshape(x_t, [1, h, w, 1])
        y_grid = tf.reshape(y_t, [1, h, w, 1])
        
        x_dir = tf.multiply(target, x_t) - tf.multiply(reference, x_grid)
        y_dir = tf.multiply(target, y_t) - tf.multiply(reference, y_grid)
        
    # get direction between reference and target
    if max_avg_flag == 'max':
        x_dir = global_max_pool(x_dir)
        y_dir = global_max_pool(y_dir)
    elif max_avg_flag == 'avg':
        x_dir = global_avg_pool(x_dir)    
        y_dir = global_avg_pool(y_dir)
    else:
        raise ValueError("Unknown function selection: {}".format(max_avg_flag))

    # generate attention map
    # get cross attention feature
    
    return output