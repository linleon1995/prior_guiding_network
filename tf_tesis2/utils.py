'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np

#TODO: pretrain mode
#TODO: batch norm for small batch size


      
def fc_layer(inputs, 
             layer_size,
             _std,
             reuse=False,
             scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.
  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.
  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
  Returns:
    Computed features after split separable conv2d.
  """
    input_size, output_size = layer_size
    shape = tf.shape(inputs)
    dim = tf.reduce_prod( shape[1:] )
    x = tf.reshape(inputs, [-1, dim])

    with tf.variable_scope(scope, reuse=reuse):
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=[input_size, output_size], stddev=_std))
        w = w / tf.sqrt(input_size/2)
        b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[output_size]))
        output = tf.nn.bias_add( tf.matmul(x, w, name='fc'), b)
    return output


def conv2d(inputs, 
          filter_shape, 
          strides=[1,1,1,1],
          padding='SAME',
          dilations=[1,1,1,1],  
          activate=None, 
          bn_flag=True, 
          is_training=True, 
          reuse=False,
          scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.
  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.
  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
  Returns:
    Computed features after split separable conv2d.
  """
    assert inputs.get_shape().ndims == 4
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    _std = np.sqrt(2 / (kernel_h * kernel_w * out_channels))

    with tf.variable_scope(scope, reuse=reuse) as scope:
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=filter_shape, stddev=_std))
        conv = tf.nn.conv2d(inputs, 
                            w, 
                            strides=strides, 
                            padding=padding,
#                            dilations=dilations,
                            name='conv',
                            )
        
        if bn_flag:
            output = batch_norm(conv, is_training=is_training, scope='batch_norm')
        else:
            b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[out_channels]))
            output = tf.nn.bias_add(conv, b)
            
        if activate is not None:
            output = activate(output) 

    return output


def atrous_conv2d(inputs, 
          filter_shape, 
          strides=[1,1,1,1],
          padding='SAME',
          rate=1,  
          activate=None, 
          bn_flag=True, 
          is_training=True, 
          reuse=False,
          scope=None):
    # TODO: temporally implement for api 1.4.1, merge with conv2d after update tensorflow version
    """Splits a separable conv2d into depthwise and pointwise conv2d.
  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.
  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
  Returns:
    Computed features after split separable conv2d.
  """
    assert inputs.get_shape().ndims == 4
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    _std = np.sqrt(2 / (kernel_h * kernel_w * out_channels))

    with tf.variable_scope(scope, reuse=reuse) as scope:
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=filter_shape, stddev=_std))
        conv = tf.nn.atrous_conv2d(inputs, 
                            w, 
                            rate=rate,
                            padding=padding,
                            name='atrous_conv',
                            )
        
        if bn_flag:
            output = batch_norm(conv, is_training=is_training, scope='batch_norm')
        else:
            b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[out_channels]))
            output = tf.nn.bias_add(conv, b)
            
        if activate is not None:
            output = activate(output) 

    return output

      
def split_separable_conv2d(inputs, 
                          filter_shape, 
                          strides=[1,1,1,1],
                          padding='SAME',
                          dilations=[1,1], 
                          channel_multiplier=1, 
                          activate_func=None, 
                          bn_flag=True, 
                          is_training=True,
                          reuse=False,
                          scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.
  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.
  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
  Returns:
    Computed features after split separable conv2d.
  """
    assert inputs.get_shape().ndims == 4
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    _std = np.sqrt(2 / (kernel_h * kernel_w * out_channels))
    
    with tf.variable_scope(scope, reuse=reuse):
        w_depthwise = tf.get_variable(
                "W_depth",
                initializer=tf.truncated_normal(shape=[kernel_h, kernel_w, in_channels, channel_multiplier], stddev=_std))
        w_pointwise = tf.get_variable(
                "W_point",
                initializer=tf.truncated_normal(shape=[1,1, channel_multiplier*in_channels, out_channels], stddev=_std))
        conv = tf.nn.separable_conv2d(
                input=inputs,
                depthwise_filter=w_depthwise,
                pointwise_filter=w_pointwise,
                strides=strides,
                padding=padding,
                rate=dilations,
                name='separable_conv',
                )
        
        if bn_flag:
            out_node = batch_norm(conv, is_training=is_training, name='batch_norm')
        else:
            b = tf.get_variable(
                "b",
                initializer=tf.constant(shape=out_channels, stddev=0.1))
            out_node = tf.nn.bias_add(conv, b)
        
        if activate_func is not None:
            output = activate_func(out_node)    
    return output
        

def batch_norm(inputs, scope, is_training):
    """BN for the first input"""
    assert inputs.get_shape().ndims == 4
    with tf.variable_scope(scope):
        output = tf.layers.batch_normalization(inputs, training=is_training)
    return output

    
def upsampling_layer(inputs, n, mode, scope='upsampling'):
    """upsampling layer
    Args:
    Return:
    """
    assert inputs.get_shape().ndims == 4
    x_shape = tf.shape(inputs)
    new_h = x_shape[1]*n
    new_w = x_shape[2]*n
    
    with tf.variable_scope(scope):
        if mode is 'Nearest_Neighbor':
            output = tf.image.resize_nearest_neighbor(inputs, [new_h, new_w], name='nearest_neighbor')
            
        if mode is 'Bilinear':
            output = tf.image.resize_bilinear(inputs, [new_h, new_w], name='bilinear')
        
        if mode is 'Bicubic':
            output = tf.image.resize_bicubic(inputs, [new_h, new_w], name='bicubic')

    return output


def global_avg_pool(inputs, keep_dims=False):
    """global_avg_pooling"""
    assert inputs.get_shape().ndims == 4
    with tf.variable_scope('global_average_pooling'):
        output = tf.reduce_mean(inputs, [1, 2], keep_dims=keep_dims)
    return output
    

def global_max_pool(inputs, keep_dims=False):
    """global_max_pooling"""
    assert inputs.get_shape().ndims == 4
    with tf.variable_scope('global_average_pooling'):
        output = tf.reduce_sum(inputs, [1, 2], keepdims=keep_dims)
    return output







