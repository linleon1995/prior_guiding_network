#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 15:55:53 2019

@author: acm528_02
"""


from __future__ import print_function, division, absolute_import, unicode_literals

import os
import shutil
import numpy as np
from collections import OrderedDict
import logging

import tensorflow as tf
from tensorflow.python.ops import math_ops
import matplotlib.pyplot as plt
#from tf_unet_multi_task import util
import inspect
import time
VGG_MEAN = [103.939, 116.779, 123.68]
#from tf_tesis2.layer_multi_task import (weight_variable, weight_variable_deconv, bias_variable, 
#                            conv2d, deconv2d, upsampling2d, max_pool, crop_and_concat, pixel_wise_softmax_2,
#                            cross_entropy, batch_norm, softmax, fc_layer, new_conv_layer_bn, new_conv_layer, upsampling_layer)
from tf_tesis2.network import (unet_prior_guide_prof_encoder, unet_prior_guide_prof_decoder, crn_encoder_sep, crn_decoder_sep, 
                               crn_encoder_sep_com, crn_decoder_sep_com, crn_encoder_sep_resnet50, crn_decoder_sep_resnet50)
from tf_tesis2 import stn
from tf_tesis2.dense_crf import crf_inference
from scipy.misc import imresize, imrotate
from tf_tesis2.utils import (conv2d)
from tf_tesis2 import train_utils, common, model
import CT_scan_util_multi_task
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

parser = argparse.ArgumentParser()

parser.add_argument('--base_learning_rate', type=float, default=1e-2,
                    help='The initial learning rate')

parser.add_argument('--learning_power', type=float, default=0.9,
                    help='The power value used in the poly learning policy')

parser.add_argument('--learning_policy', type=str, default='poly',
                    help='')

#parser.add_argument('--learning_rate_decay_step', type=str, default='poly',
#                    help='')
#training_number_of_steps
parser.add_argument('--learning_rate_decay_factor', type=float, default=0.1,
                    help='')

parser.add_argument('--train_batch_size', type=int, default=8,
                    help='The number of images in each batch during training')

parser.add_argument('--z-flag', type=bool, default=True,
                    help='')

parser.add_argument('--train_split', type=str, default='train',
                    help='Which split of the dataset to be used for training')

parser.add_argument('--train-logdir', type=str, default='train',
                    help='Where the checkpoint and logs are stored.')

parser.add_argument('--tf_initial_checkpoint', type=str, default=None,
                    help='The initial checkpoint in tensorflow format.')

parser.add_argument('--lambda-z', type=float, default=None,
                    help='')

parser.add_argument('--num-clones', type=int, default=2,
                    help='Number of clones to deploy.')

parser.add_argument('--last-layers-contain-logits-only', type=bool, default=True,
                    help='Only consider logits as last layers or not.')

parser.add_argument('--last-layer-gradient-multiplier', type=float, default=1.0,
                    help='The gradient multiplier for last layers, which is used to '
                    'boost the gradient of last layers if the value > 1.')

parser.add_argument('--log-steps', type=int, default=10,
                    help='Display logging information at every log_steps.')

def _build_model(data_provider, num_of_classes, ignore_label, provider_list):
    images = provider_list[0]
    labels = provider_list[1]
    
    model_options = '...'
    # build model
    output_logits_dict = model._get_logits(
            images,
            model_options,
            weight_decay=0.0001,
            reuse=None,
            is_training=False,
            fine_tune_batch_norm=False,
            nas_training_hyper_parameters=None)

    
    # get main loss and auxiliary losses
    # TODO: for output, num_classes in six.iteritems(outputs_to_num_classes):
    train_utils.add_softmax_cross_entropy_loss_for_each_scale(output_logits_dict["logits"],
                                                  labels,
                                                  num_of_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  scope=None)
    
    train_utils.add_auxilary_z_loss(output_logits_dict["z_logits"],
                                labels,
                                data_provider.z_class,
                                lambda_z=FLAGS.lambda_z,
                                class_weights=None)
    # Log the summary
    _log_summaries(images, labels, num_of_classes, output_logits_dict["logits"])



def _tower_loss(iterator, num_of_classes, ignore_label, scope, reuse_variable, provider_list):
    """Calculates the total loss on a single tower running the deeplab model.
    Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_of_classes: Number of classes for the dataset.
    ignore_label: Ignore label for the dataset.
    scope: Unique prefix string identifying the deeplab tower.
    reuse_variable: If the variable should be reused.
    Returns:
    The total loss for a batch of data.
    """
    with tf.variable_scope(
            tf.get_variable_scope(), reuse=True if reuse_variable else None):
        _build_model(iterator, num_of_classes, ignore_label)
    
    losses = tf.losses.get_losses(scope=scope)
    for loss in losses:
        tf.summary.scalar('Losses/%s' % loss.op.name, loss)
    
    regularization_loss = tf.losses.get_regularization_loss(scope=scope)
    tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                      regularization_loss)
    
    total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
    return total_loss


def _train_model(data_provider, num_of_classes, ignore_label, provider_list):
    # get learning rate
    global_step = tf.train.get_or_create_global_step()
    
    learning_rate = train_utils.get_model_learning_rate(
            FLAGS.learning_policy,
            FLAGS.base_learning_rate,
            FLAGS.learning_rate_decay_step,
            FLAGS.learning_rate_decay_factor,
            FLAGS.training_number_of_steps,
            FLAGS.learning_power)
    tf.summary.scalar('learning_rate', learning_rate)
    
    # get optimizer
    optimizer = train_utils.get_model_optimizer('adam', learning_rate=learning_rate)
    
    # get total loss
    tower_losses = []
    tower_grads = []
    for i in range(FLAGS.num_clones):
        with tf.device('/gpu:%d' % i):
          # First tower has default name scope.
          name_scope = ('clone_%d' % i) if i else ''
          with tf.name_scope(name_scope) as scope:
              loss = _tower_loss(iterator=data_provider,
                                 num_of_classes=num_of_classes, 
                                 ignore_label=ignore_label, 
                                 scope=scope, 
                                 reuse_variable=(i != 0), 
                                 provider_list=provider_list)
              tower_losses.append(loss)
          
    for i in range(FLAGS.num_clones):
        # TODO: check gpu using
        with tf.device('/gpu:%d' % i):
        #############
          name_scope = ('clone_%d' % i) if i else ''
          with tf.name_scope(name_scope) as scope:
            grads = optimizer.compute_gradients(tower_losses[i])
            tower_grads.append(grads)
    
    with tf.device('/cpu:0'):
        grads_and_vars = _average_gradients(tower_grads)

        # Modify the gradients for biases and last layer variables.
        # TODO: check muliplier
        last_layers = model.get_extra_layer_scopes(
            FLAGS.last_layers_contain_logits_only)
        grad_mult = train_utils.get_model_gradient_multipliers(
            last_layers, FLAGS.last_layer_gradient_multiplier)
        if grad_mult:
          grads_and_vars = tf.contrib.training.multiply_gradients(
              grads_and_vars, grad_mult)
        #############
        # Create gradient update op.
        grad_updates = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)
    
        # Gather update_ops. These contain, for example,
        # the updates for the batch_norm variables created by model_fn.
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        update_ops.append(grad_updates)
        update_op = tf.group(*update_ops)
    
        total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
        
        # Print total loss to the terminal.
        # This implementation is mirrored from tf.slim.summaries.
        should_log = math_ops.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)
        total_loss = tf.cond(
            should_log,
            lambda: tf.Print(total_loss, [total_loss], 'Total loss is :'),
            lambda: total_loss)
        tf.summary.scalar('total_loss', total_loss)
        with tf.control_dependencies([update_op]):
          train_tensor = tf.identity(total_loss, name='train_op')
    
        # Excludes summaries from towers other than the first one.
        summary_op = tf.summary.merge_all(scope='(?!clone_)')

    return train_tensor, summary_op
    

def _log_summaries(input_image, label, num_of_classes, output):
  """Logs the summaries for the model.
  Args:
    input_image: Input image of the model. Its shape is [batch_size, height,
      width, channel].
    label: Label of the image. Its shape is [batch_size, height, width].
    num_of_classes: The number of classes of the dataset.
    output: Output of the model. Its shape is [batch_size, height, width].
  """
  # Add summaries for model variables.
  for model_var in tf.model_variables():
    tf.summary.histogram(model_var.op.name, model_var)

  # Add summaries for images, labels, semantic predictions.
  if FLAGS.save_summaries_images:
    tf.summary.image('samples/%s' % common.IMAGE, input_image)

    # Scale up summary image pixel values for better visualization.
    pixel_scaling = max(1, 255 // num_of_classes)
    summary_label = tf.cast(label * pixel_scaling, tf.uint8)
    tf.summary.image('samples/%s' % common.LABEL, summary_label)

    predictions = tf.expand_dims(tf.argmax(output, 3), -1)
    summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
    tf.summary.image('samples/%s' % common.OUTPUT_TYPE, summary_predictions)    
    

def _average_gradients(tower_grads):
  """Calculates average of gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads, variables = zip(*grad_and_vars)
    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

    # All vars are of the same value, using the first tower here.
    average_grads.append((grad, variables[0]))

  return average_grads

   
def main(unused_argv):
    logging.info("Training on {}".format(FLAGS.train_split))
    
    
    
    # get data, guidance
    data_provider = CT_scan_util_multi_task.MedicalDataProvider(
                                      raw_path=FLAGS.raw_path,
                                      mask_path=FLAGS.mask_path,
                                      subject_list=common.TRAIN_SUBJECT,
                                      resize_ratio=0.5,
                                      data_aug=FLAGS.data_augmentation,
                                      cubic=False,
                                      z_class=FLAGS.z_class,
                                      nx=common.HEIGHT,
                                      ny=common.WIDTH,
                                      HU_window=common.HU_WINDOW,
                                      )
    # TODO: modify data_provider for tf.provider abandon
    x = tf.placeholder("float", shape=[None, common.HEIGHT, common.WIDTH, common.channels], name='x')
    y = tf.placeholder("float", shape=[None, common.HEIGHT, common.WIDTH, data_provider.n_class], name='y')
    is_training = tf.placeholder(tf.bool)
    provider_list = [x, y, is_training]
    if FLAGS.lambda_z is not None: 
        z_label = tf.placeholder("int32", shape=[None, data_provider.n_class], name='z_label')
        provider_list.append(z_label)
    
    train_tensor, summary_op = _train_model(data_provider, data_provider.n_class, data_provider.ignore_label, provider_list)
    
    # training session
    session_config = tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=False)
    
    if FLAGS.tf_initial_checkpoint:
        init_fn = train_utils.get_model_init_fn(
                train_logdir=FLAGS.train_logdir,
                tf_initial_checkpoint=FLAGS.tf_initial_checkpoint,
                initialize_last_layer='not_using',
                last_layers='not_using',
                ignore_missing_vars=True)
    
    scaffold = tf.train.Scaffold(
          init_fn=init_fn,
          summary_op=summary_op,
      )
    
    stop_hook = tf.train.StopAtStepHook(
          last_step=FLAGS.training_number_of_steps)
    
#    saver_hook = tf.train.CheckpointSaverHook(output_path, 
#                                          save_steps=2, 
#                                          saver=saver)
    
    with tf.train.MonitoredTrainingSession(
            master=FLAGS.master,
            config=session_config,
            scaffold=scaffold,
            checkpoint_dir=FLAGS.train_logdir,
            save_checkpoint_secs=FLAGS.save_interval_secs,
            hooks=[stop_hook]) as sess:
          while not sess.should_stop():
              batch_x, batch_y, batch_z, batch_angle, batch_class_gt = data_provider(FLAGS.train_batch_size)
              _feed_dict = {x: batch_x, 
                            y: batch_y, 
                            is_training: is_training,}
              if FLAGS.lambda_z is not None: _feed_dict[z_label] = batch_z
              sess.run([train_tensor], feed_dict=_feed_dict)
              

    
    
if __name__ == '__main__':
    pass
    FLAGS, unparsed = parser.parse_known_args()
    main()    
    