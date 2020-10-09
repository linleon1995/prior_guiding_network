#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import functools
import tensorflow as tf
import argparse
from tensorflow.contrib import framework as contrib_framework
import matplotlib.pyplot as plt
from utils import losses
import common

LOSSES_MAP = {"softmax_cross_entropy": losses.add_softmax_cross_entropy_loss_for_each_scale,
              "softmax_dice_loss": losses.add_softmax_dice_loss_for_each_scale,
              "sigmoid_cross_entropy": losses.add_sigmoid_cross_entropy_loss_for_each_scale,
              "sigmoid_dice_loss": losses.add_sigmoid_dice_loss_for_each_scale,
              "softmax_generalized_dice_loss": losses.add_softmax_generalized_dice_loss_for_each_scale,}


def create_training_path(train_logdir):
  idx = 0
  path = os.path.join(train_logdir, "run_{:03d}".format(idx))
  while os.path.exists(path):
    idx += 1
    path = os.path.join(train_logdir, "run_{:03d}".format(idx))
  os.makedirs(path)
  return path


def str2bool(v):
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Unsupported value encountered.')
        

def get_func(func):
  @functools.wraps(func)
  def network_fn(*args, **kwargs):
      return func(*args, **kwargs)
  return network_fn
  
         
def get_loss_func(loss_name):
  if loss_name not in LOSSES_MAP:
    raise ValueError('Unsupported loss %s.' % loss_name)
  func = LOSSES_MAP[loss_name]
  return get_func(func)


def get_losses(output_dict,
               samples,
               loss_dict,
               num_classes,
               seq_length,
               batch_size=None,
               z_class=None):
    """
    The function define the objective function of model.
    Each loss will added through tf.losses.add_loss().
    Please refer to utils/losses.py for more details.
    """
    # Calculate segmentation loss
    label = samples[common.LABEL]

    if seq_length > 1:
      _, t, h, w, c = label.shape.as_list()
      frame_label = tf.reshape(label, [-1, h, w, c])
      label = label[:,seq_length//2]
    else:
      frame_label = label

    scales_to_logits = {"full": output_dict[common.OUTPUT_TYPE]}
    get_loss_func(loss_dict[common.OUTPUT_TYPE]["loss"])(
      scales_to_logits=scales_to_logits,
      labels=label,
      num_classes=num_classes,
      ignore_label=255,
      loss_weight=loss_dict[common.OUTPUT_TYPE]["weights"],
      scope=loss_dict[common.OUTPUT_TYPE]["scope"])

    # Calculate stage prediction (each stage guidance) loss
    if "stage_pred" in loss_dict:
        # i = 0
        stage_pred = tf.get_collection("stage_pred")
        for i, value in enumerate(stage_pred):
          # if "guidance" in name:
            get_loss_func(loss_dict["stage_pred"]["loss"])(
              scales_to_logits={"stage%d" %(i+1): value},
              labels=frame_label,
              num_classes=num_classes,
              ignore_label=255,
              loss_weight=loss_dict["stage_pred"]["weights"],
              scope=loss_dict["stage_pred"]["scope"])
            # i+=1

    # Calculate guidance (initial) loss
    if common.GUIDANCE in loss_dict:
        g = {"transform": output_dict[common.GUIDANCE]}
        get_loss_func(loss_dict[common.GUIDANCE]["loss"])(
          scales_to_logits=g,
          labels=frame_label,
          num_classes=num_classes,
          ignore_label=255,
          loss_weight=loss_dict[common.GUIDANCE]["weights"],
          scope=loss_dict[common.GUIDANCE]["scope"])

    # Calculate longitudinal classification loss
    if common.OUTPUT_Z in loss_dict:
        get_loss_func(loss_dict[common.OUTPUT_Z]["loss"])(
            scales_to_logits={common.OUTPUT_Z: output_dict[common.OUTPUT_Z]},
            labels=samples[common.Z_LABEL],
            num_classes=z_class,
            ignore_label=255,
            loss_weight=loss_dict[common.OUTPUT_Z]["weights"],
            upsample_logits=False,
            scope=loss_dict[common.OUTPUT_Z]["scope"])


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


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      # initialize_first_layer,
                      initialize_last_layer,
                      # first_layer,
                      last_layers=None,
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
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)
  # if not initialize_first_layer:
  exclude_list.append('resnet_v1_50/conv1_1/weights:0')
  variables_to_restore = contrib_framework.get_variables_to_restore(
      exclude=exclude_list)

  # Restore without Adam parameters
  new_v = []
  for v in variables_to_restore:
    if "Adam" not in v.name:
      new_v.append(v)

  variables_to_restore = new_v

  if variables_to_restore:
    init_op, init_feed_dict = contrib_framework.assign_from_checkpoint(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)

    global_step = tf.train.get_or_create_global_step()

    def restore_fn(scaffold, sess):
      sess.run(init_op, init_feed_dict)
      sess.run([global_step])

    return restore_fn

  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.
  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.
  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.
  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in tf.model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none',
                            decay_steps=0.0,
                            end_learning_rate=1e-6,
                            boundaries=None,
                            boundary_learning_rates=None):
  """Gets model's learning rate.
  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power
  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.
    decay_steps: Float, `decay_steps` for polynomial learning rate.
    end_learning_rate: Float, `end_learning_rate` for polynomial learning rate.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries.
    boundary_learning_rates: A list of `Tensor`s or `float`s or `int`s that
      specifies the values for the intervals defined by `boundaries`. It should
      have one more element than `boundaries`, and all elements should have the
      same type.
  Returns:
    Learning rate for the specified learning policy.
  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
    ValueError: If `boundaries` and `boundary_learning_rates` are not set for
      multi_steps learning rate decay.
  """
  global_step = tf.train.get_or_create_global_step()
  adjusted_global_step = tf.maximum(global_step - slow_start_step, 0)
  if decay_steps == 0.0:
    tf.logging.info('Setting decay_steps to total training steps.')
    decay_steps = training_number_of_steps - slow_start_step
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        adjusted_global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        adjusted_global_step,
        decay_steps=decay_steps,
        end_learning_rate=end_learning_rate,
        power=learning_power)
  elif learning_policy == 'cosine':
    learning_rate = tf.train.cosine_decay(
        base_learning_rate,
        adjusted_global_step,
        training_number_of_steps - slow_start_step)
  elif learning_policy == 'multi_steps':
    if boundaries is None or boundary_learning_rates is None:
      raise ValueError('Must set `boundaries` and `boundary_learning_rates` '
                       'for multi_steps learning rate decay.')
    learning_rate = tf.train.piecewise_constant_decay(
        adjusted_global_step,
        boundaries,
        boundary_learning_rates)
  else:
    raise ValueError('Unknown learning policy.')

  adjusted_slow_start_learning_rate = slow_start_learning_rate
  if slow_start_burnin_type == 'linear':
    # Do linear burnin. Increase linearly from slow_start_learning_rate and
    # reach base_learning_rate after (global_step >= slow_start_steps).
    adjusted_slow_start_learning_rate = (
        slow_start_learning_rate +
        (base_learning_rate - slow_start_learning_rate) *
        tf.to_float(global_step) / slow_start_step)
  elif slow_start_burnin_type != 'none':
    raise ValueError('Unknown burnin type.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step,
                  adjusted_slow_start_learning_rate, learning_rate)