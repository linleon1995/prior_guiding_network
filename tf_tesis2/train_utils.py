#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 14:12:37 2019

@author: acm528_02
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
    

def get_network(network_name, preprocess_images,
                preprocessed_images_dtype=tf.float32, arg_scope=None):
  """Gets the network.
  Args:
    network_name: Network name.
    preprocess_images: Preprocesses the images or not.
    preprocessed_images_dtype: The type after the preprocessing function.
    arg_scope: Optional, arg_scope to build the network. If not provided the
      default arg_scope of the network would be used.
  Returns:
    A network function that is used to extract features.
  Raises:
    ValueError: network is not supported.
  """
  if network_name not in networks_map:
    raise ValueError('Unsupported network %s.' % network_name)
  arg_scope = arg_scope or arg_scopes_map[network_name]()
  def _identity_function(inputs, dtype=preprocessed_images_dtype):
    return tf.cast(inputs, dtype=dtype)
  if preprocess_images:
    preprocess_function = _PREPROCESS_FN[network_name]
  else:
    preprocess_function = _identity_function
  func = networks_map[network_name]
  @functools.wraps(func)
  def network_fn(inputs, *args, **kwargs):
    with slim.arg_scope(arg_scope):
      return func(preprocess_function(inputs, preprocessed_images_dtype),
                  *args, **kwargs)
  return network_fn


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value, axis=-1)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = plt.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
  
    value = tf.cast(value*255, tf.uint8)
    return value

    
def add_auxilary_z_loss(logits,
                        labels,
                        z_class,
                        lambda_z=1.0,
                        class_weights=None):
    """Add auxilarycross entropy loss for supporting task.
    Args:
      logits: 
      labels: Groundtruth labels with shape [batch, n_class, z_class+1].
    """
    labels = tf.one_hot(indices=labels,
                        depth=int(z_class+1),
                        on_value=1,
                        off_value=0,
                        axis=-1,
                        )

    if class_weights is not None:
        class_weights = tf.constant(np.array(class_weights, dtype=np.float32))

        weight_map = tf.multiply(labels, class_weights)
        weight_map = tf.reduce_sum(weight_map, axis=1)

        loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                           labels=labels)
        weighted_loss = tf.multiply(loss_map, weight_map)

        loss = tf.reduce_mean(weighted_loss)
        
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                      labels=labels))
        
    auxilary_z_loss = tf.losses.add_loss(loss)
    return lambda_z * auxilary_z_loss


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale.
  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: Float, loss weight.
    upsample_logits: Boolean, upsample logits or not.
    hard_example_mining_step: An integer, the training step in which the hard
      exampling mining kicks off. Note that we gradually reduce the mining
      percent to the top_k_percent_pixels. For example, if
      hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
      mining percent will gradually reduce from 100% to 25% until 100K steps
      after which we only mine top 25% pixels.
    top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
      < 1.0, only compute the loss for the top k percent pixels (e.g., the top
      20% pixels). This is useful for hard pixel mining.
    scope: String, the scope for the loss.
  Raises:
    ValueError: Label or logits is None.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    not_ignore_mask = tf.to_float(tf.not_equal(scaled_labels,
                                               ignore_label)) * loss_weight
    one_hot_labels = tf.one_hot(
        scaled_labels, num_classes, on_value=1.0, off_value=0.0)

    if top_k_percent_pixels == 1.0:
      # Compute the loss for all pixels.
      tf.losses.softmax_cross_entropy(
          one_hot_labels,
          tf.reshape(logits, shape=[-1, num_classes]),
          weights=not_ignore_mask,
          scope=loss_scope)
    else:
      logits = tf.reshape(logits, shape=[-1, num_classes])
      weights = not_ignore_mask
      with tf.name_scope(loss_scope, 'softmax_hard_example_mining',
                         [logits, one_hot_labels, weights]):
        one_hot_labels = tf.stop_gradient(
            one_hot_labels, name='labels_stop_gradient')
        pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=logits,
            name='pixel_losses')
        weighted_pixel_losses = tf.multiply(pixel_losses, weights)
        num_pixels = tf.to_float(tf.shape(logits)[0])
        # Compute the top_k_percent pixels based on current training step.
        if hard_example_mining_step == 0:
          # Directly focus on the top_k pixels.
          top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
        else:
          # Gradually reduce the mining percent to top_k_percent_pixels.
          global_step = tf.to_float(tf.train.get_or_create_global_step())
          ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
          top_k_pixels = tf.to_int32(
              (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
        top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                      k=top_k_pixels,
                                      sorted=True,
                                      name='top_k_percent_pixels')
        total_loss = tf.reduce_sum(top_k_losses)
        num_present = tf.reduce_sum(
            tf.to_float(tf.not_equal(top_k_losses, 0.0)))
        loss = _div_maybe_zero(total_loss, num_present)
        tf.losses.add_loss(loss)
        
        
def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power):

    global_step = tf.train.get_or_create_global_step()
    
    if learning_policy == 'step':
        learning_rate = tf.train.exponential_decay(
                base_learning_rate,
                global_step,
                learning_rate_decay_step,
                learning_rate_decay_factor,
                staircase=True)
    elif learning_policy == 'poly':
        learning_rate = tf.train.polynomial_decay(
                base_learning_rate, 
                global_step,
                learning_rate_decay_step, 
                end_learning_rate=0,
                power=learning_power,
                )
    else:
        raise ValueError('Unknown learning policy.')
    return learning_rate


def get_model_optimizer(opt_variant,
                        learning_rate,
                        momentum=0.9):
    if opt_variant == "momentum":
        optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    elif opt_variant == "adam":
        optimizer = tf.train.AdamOptimizer(learning_rate)
    else:
        raise ValueError("Unknown optimizer: {}".foramt(opt_variant))
    return optimizer


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.
  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.
  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
#  exclude_list = ['global_step']
#  if not initialize_last_layer:
#    exclude_list.extend(last_layers)
#
#  variables_to_restore = tf.contrib.framework.get_variables_to_restore(
#      exclude=exclude_list)
  variables_to_restore = tf.contrib.framework.get_variables_to_restore()
  
  if variables_to_restore:
    init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
    global_step = tf.train.get_or_create_global_step()

    def restore_fn(unused_scaffold, sess):
      sess.run(init_op, init_feed_dict)
      sess.run([global_step])

    return restore_fn

  return None