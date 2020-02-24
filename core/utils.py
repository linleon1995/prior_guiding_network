import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


# Quantized version of sigmoid function.
q_sigmoid = lambda x: tf.nn.relu6(x + 3) * 0.16667


    
def fc_layer(inputs, 
             layer_size,
             _std,
             reuse=None,
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
    # shape = tf.shape(inputs)
    # dim = tf.reduce_prod( shape[1:] )
    # x = tf.reshape(inputs, [-1, dim])
    x = tf.layers.flatten(inputs)
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
    # TODO: remove numpy dependency
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
        output = tf.reduce_max(inputs, [1, 2], keepdims=keep_dims)
    return output


def resize_bilinear(images, size, output_dtype=tf.float32):
  """Returns resized images as output_type.
  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
      for the images.
    output_dtype: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_dtype.
  """
  images = tf.image.resize_bilinear(images, size, align_corners=True)
  return tf.cast(images, dtype=output_dtype)


def scale_dimension(dim, scale):
  """Scales the input dimension.
  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.
  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)


def split_separable_conv2d(inputs,
                           filters,
                           kernel_size=3,
                           rate=1,
                           weight_decay=0.00004,
                           depthwise_weights_initializer_stddev=0.33,
                           pointwise_weights_initializer_stddev=0.06,
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
  outputs = slim.separable_conv2d(
      inputs,
      None,
      kernel_size=kernel_size,
      depth_multiplier=1,
      rate=rate,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=depthwise_weights_initializer_stddev),
      weights_regularizer=None,
      scope=scope + '_depthwise')
  return slim.conv2d(
      outputs,
      filters,
      1,
      weights_initializer=tf.truncated_normal_initializer(
          stddev=pointwise_weights_initializer_stddev),
      weights_regularizer=slim.l2_regularizer(weight_decay),
      scope=scope + '_pointwise')


def get_label_weight_mask(labels, ignore_label, num_classes, label_weights=1.0):
  """Gets the label weight mask.
  Args:
    labels: A Tensor of labels with the shape of [-1].
    ignore_label: Integer, label to ignore.
    num_classes: Integer, the number of semantic classes.
    label_weights: A float or a list of weights. If it is a float, it means all
      the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, label_weights = [0.1, 0.5] means the weight for label 0 is 0.1
      and the weight for label 1 is 0.5.
  Returns:
    A Tensor of label weights with the same shape of labels, each element is the
      weight for the label with the same index in labels and the element is 0.0
      if the label is to ignore.
  Raises:
    ValueError: If label_weights is neither a float nor a list, or if
      label_weights is a list and its length is not equal to num_classes.
  """
  if not isinstance(label_weights, (float, list)):
    raise ValueError(
        'The type of label_weights is invalid, it must be a float or a list.')

  if isinstance(label_weights, list) and len(label_weights) != num_classes:
    raise ValueError(
        'Length of label_weights must be equal to num_classes if it is a list, '
        'label_weights: %s, num_classes: %d.' % (label_weights, num_classes))

  not_ignore_mask = tf.not_equal(labels, ignore_label)
  not_ignore_mask = tf.cast(not_ignore_mask, tf.float32)
  if isinstance(label_weights, float):
    return not_ignore_mask * label_weights

  label_weights = tf.constant(label_weights, tf.float32)
  weight_mask = tf.einsum('...y,y->...',
                          tf.one_hot(labels, num_classes, dtype=tf.float32),
                          label_weights)
  return tf.multiply(not_ignore_mask, weight_mask)


def get_batch_norm_fn(sync_batch_norm_method):
  """Gets batch norm function.
  Currently we only support the following methods:
    - `None` (no sync batch norm). We use slim.batch_norm in this case.
  Args:
    sync_batch_norm_method: String, method used to sync batch norm.
  Returns:
    Batchnorm function.
  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  if sync_batch_norm_method == 'None':
    return slim.batch_norm
  else:
    raise ValueError('Unsupported sync_batch_norm_method.')


def get_batch_norm_params(decay=0.9997,
                          epsilon=1e-5,
                          center=True,
                          scale=True,
                          is_training=True,
                          sync_batch_norm_method='None',
                          initialize_gamma_as_zeros=False):
  """Gets batch norm parameters.
  Args:
    decay: Float, decay for the moving average.
    epsilon: Float, value added to variance to avoid dividing by zero.
    center: Boolean. If True, add offset of `beta` to normalized tensor. If
      False,`beta` is ignored.
    scale: Boolean. If True, multiply by `gamma`. If False, `gamma` is not used.
    is_training: Boolean, whether or not the layer is in training mode.
    sync_batch_norm_method: String, method used to sync batch norm.
    initialize_gamma_as_zeros: Boolean, initializing `gamma` as zeros or not.
  Returns:
    A dictionary for batchnorm parameters.
  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  batch_norm_params = {
      'is_training': is_training,
      'decay': decay,
      'epsilon': epsilon,
      'scale': scale,
      'center': center,
  }
  if initialize_gamma_as_zeros:
    if sync_batch_norm_method == 'None':
      # Slim-type gamma_initialier.
      batch_norm_params['param_initializers'] = {
          'gamma': tf.zeros_initializer(),
      }
    else:
      raise ValueError('Unsupported sync_batch_norm_method.')
  return batch_norm_params