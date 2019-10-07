#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 16:21:16 2019

@author: acm528_02
"""

import tensorlfow as tf
from tf_tesis2 import crn_network, crn_rnn, crn_testing
from tf_tesis2 import feature_extractor

slim = tf.contrib.slim

# A map from network name to network function.
networks_map = {
    'crn_vanilla': crn_network.crn_vanilla,
    'crn_onestage': crn_network.crn_onestage,
    'crn_bilstm': crn_rnn.crn_bilstm,
    'crn_vanilla_relation': crn_testing.crn_vanilla_relation,
    'crn_onestage_relation': crn_testing.crn_onestage_relation,
    'crn_onestage_region_rcca': crn_testing.crn_onestage_region_rcca,
}


def extract_features(images,
                     model_options,
                     weight_decay=0.0001,
                     reuse=None,
                     is_training=False,
                     fine_tune_batch_norm=False,
                     nas_training_hyper_parameters=None):
  """Extracts features by the particular model_variant.
  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. Its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.
  Returns:
    concat_logits: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined by
      the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  """
  features, end_points = feature_extractor.extract_features(
      images,
      output_stride=model_options.output_stride,
      multi_grid=model_options.multi_grid,
      model_variant=model_options.model_variant,
#      depth_multiplier=model_options.depth_multiplier,
#      divisible_by=model_options.divisible_by,
#      weight_decay=weight_decay,
#      reuse=reuse,
      is_training=is_training,
#      preprocessed_images_dtype=model_options.preprocessed_images_dtype,
      fine_tune_batch_norm=fine_tune_batch_norm,
#      nas_stem_output_num_conv_filters=(
#          model_options.nas_stem_output_num_conv_filters),
#      nas_training_hyper_parameters=nas_training_hyper_parameters,
#      use_bounded_activation=model_options.use_bounded_activation
      )

  if not model_options.aspp_with_batch_norm:
    return features, end_points
  else:
    if model_options.dense_prediction_cell_config is not None:
      tf.logging.info('Using dense prediction cell config.')
      dense_prediction_layer = dense_prediction_cell.DensePredictionCell(
          config=model_options.dense_prediction_cell_config,
          hparams={
              'conv_rate_multiplier': 16 // model_options.output_stride,
          })
      concat_logits = dense_prediction_layer.build_cell(
          features,
          output_stride=model_options.output_stride,
          crop_size=model_options.crop_size,
          image_pooling_crop_size=model_options.image_pooling_crop_size,
          weight_decay=weight_decay,
          reuse=reuse,
          is_training=is_training,
          fine_tune_batch_norm=fine_tune_batch_norm)
      return concat_logits, end_points
    else:
      # The following codes employ the DeepLabv3 ASPP module. Note that we
      # could express the ASPP module as one particular dense prediction
      # cell architecture. We do not do so but leave the following codes
      # for backward compatibility.
      batch_norm_params = {
          'is_training': is_training and fine_tune_batch_norm,
          'decay': 0.9997,
          'epsilon': 1e-5,
          'scale': True,
      }

      activation_fn = (
          tf.nn.relu6 if model_options.use_bounded_activation else tf.nn.relu)

      with slim.arg_scope(
          [slim.conv2d, slim.separable_conv2d],
          weights_regularizer=slim.l2_regularizer(weight_decay),
          activation_fn=activation_fn,
          normalizer_fn=slim.batch_norm,
          padding='SAME',
          stride=1,
          reuse=reuse):
        with slim.arg_scope([slim.batch_norm], **batch_norm_params):
          depth = 256
          branch_logits = []

          if model_options.add_image_level_feature:
            if model_options.crop_size is not None:
              image_pooling_crop_size = model_options.image_pooling_crop_size
              # If image_pooling_crop_size is not specified, use crop_size.
              if image_pooling_crop_size is None:
                image_pooling_crop_size = model_options.crop_size
              pool_height = scale_dimension(
                  image_pooling_crop_size[0],
                  1. / model_options.output_stride)
              pool_width = scale_dimension(
                  image_pooling_crop_size[1],
                  1. / model_options.output_stride)
              image_feature = slim.avg_pool2d(
                  features, [pool_height, pool_width],
                  model_options.image_pooling_stride, padding='VALID')
              resize_height = scale_dimension(
                  model_options.crop_size[0],
                  1. / model_options.output_stride)
              resize_width = scale_dimension(
                  model_options.crop_size[1],
                  1. / model_options.output_stride)
            else:
              # If crop_size is None, we simply do global pooling.
              pool_height = tf.shape(features)[1]
              pool_width = tf.shape(features)[2]
              image_feature = tf.reduce_mean(
                  features, axis=[1, 2], keepdims=True)
              resize_height = pool_height
              resize_width = pool_width
            image_feature = slim.conv2d(
                image_feature, depth, 1, scope=IMAGE_POOLING_SCOPE)
            image_feature = _resize_bilinear(
                image_feature,
                [resize_height, resize_width],
                image_feature.dtype)
            # Set shape for resize_height/resize_width if they are not Tensor.
            if isinstance(resize_height, tf.Tensor):
              resize_height = None
            if isinstance(resize_width, tf.Tensor):
              resize_width = None
            image_feature.set_shape([None, resize_height, resize_width, depth])
            branch_logits.append(image_feature)

          # Employ a 1x1 convolution.
          branch_logits.append(slim.conv2d(features, depth, 1,
                                           scope=ASPP_SCOPE + str(0)))

          if model_options.atrous_rates:
            # Employ 3x3 convolutions with different atrous rates.
            for i, rate in enumerate(model_options.atrous_rates, 1):
              scope = ASPP_SCOPE + str(i)
              if model_options.aspp_with_separable_conv:
                aspp_features = split_separable_conv2d(
                    features,
                    filters=depth,
                    rate=rate,
                    weight_decay=weight_decay,
                    scope=scope)
              else:
                aspp_features = slim.conv2d(
                    features, depth, 3, rate=rate, scope=scope)
              branch_logits.append(aspp_features)

          # Merge branch logits.
          concat_logits = tf.concat(branch_logits, 3)
          concat_logits = slim.conv2d(
              concat_logits, depth, 1, scope=CONCAT_PROJECTION_SCOPE)
          concat_logits = slim.dropout(
              concat_logits,
              keep_prob=0.9,
              is_training=is_training,
              scope=CONCAT_PROJECTION_SCOPE + '_dropout')

          return concat_logits, end_points
      


def _get_logits(images,
                model_options,
                weight_decay=0.0001,
                reuse=None,
                is_training=False,
                fine_tune_batch_norm=False,
                nas_training_hyper_parameters=None):
  """Gets the logits by atrous/image spatial pyramid pooling.
  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. Its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.
  Returns:
    outputs_to_logits: A map from output_type to logits.
  """
  features, end_points = extract_features(
      images,
      model_options,
      weight_decay=weight_decay,
      reuse=reuse,
      is_training=is_training,
      fine_tune_batch_norm=fine_tune_batch_norm,
      nas_training_hyper_parameters=nas_training_hyper_parameters)

  if model_options.decoder_output_stride is not None:
    features = refine_by_decoder(
        features,
        end_points,
        crop_size=model_options.crop_size,
        decoder_output_stride=model_options.decoder_output_stride,
        decoder_use_separable_conv=model_options.decoder_use_separable_conv,
        model_variant=model_options.decoder_variant,
        weight_decay=weight_decay,
        reuse=reuse,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        use_bounded_activation=model_options.use_bounded_activation)
    
    outputs_to_logits = features
#  outputs_to_logits = {}
#  for output in sorted(model_options.outputs_to_num_classes):
#    outputs_to_logits[output] = get_branch_logits(
#        features,
#        model_options.outputs_to_num_classes[output],
#        model_options.atrous_rates,
#        aspp_with_batch_norm=model_options.aspp_with_batch_norm,
#        kernel_size=model_options.logits_kernel_size,
#        weight_decay=weight_decay,
#        reuse=reuse,
#        scope_suffix=output)

  return outputs_to_logits



def refine_by_decoder(features,
                      end_points,
                      crop_size=None,
                      decoder_output_stride=None,
                      decoder_use_separable_conv=False,
                      model_variant=None,
                      weight_decay=0.0001,
                      reuse=None,
                      is_training=False,
                      fine_tune_batch_norm=False,
                      use_bounded_activation=False,
                      preprocess_images=False):
  """Adds the decoder to obtain sharper segmentation results.
  Args:
    features: A tensor of size [batch, features_height, features_width,
      features_channels].
    end_points: A dictionary from components of the network to the corresponding
      activation.
    crop_size: A tuple [crop_height, crop_width] specifying whole patch crop
      size.
    decoder_output_stride: A list of integers specifying the output stride of
      low-level features used in the decoder module.
    decoder_use_separable_conv: Employ separable convolution for decoder or not.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference.
  Returns:
    Decoder output with size [batch, decoder_height, decoder_width,
      decoder_channels].
  Raises:
    ValueError: If crop_size is None.
  """
  
  if model_variant == 'crn_vanilla':
      features, end_points = feature_extractor.get_network(
              model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)
  elif model_variant == 'crn_onestage':
      pass
  elif model_variant == 'crn_bilstm':
      pass
  elif model_variant == 'crn_vanilla_relation':
      pass
  else:
      raise ValueError('Unknown model variant %s.' % model_variant)
    
