import tensorflow as tf
# from tensorflow.python.ops import math_ops
from core import features_extractor, stn, voxelmorph, crn_network, utils
import common

slim = tf.contrib.slim
spatial_transformer_network = stn.spatial_transformer_network
voxel_deformable_transform = voxelmorph.voxel_deformable_transform
refinement_network = crn_network.refinement_network
global_extractor = features_extractor.global_extractor

conv2d = utils.conv2d

# TODO: testing case (validation)
# TODO: voxelmorph in 2d or 3d
# TODO: scope for every submodule?

# TODO: input prior images
# TODO: transform prior


def pgb_network(images, 
                model_options,
                affine_transform,
                deformable_transform,
                labels=None,
                prior_imgs=None,
                prior_segs=None,
                num_classes=None,
                num_slices=None,
                prior_slices=None,
                batch_size=None,
                z_label_method=None,
                z_class=None,
                zero_guidance=False,
                fusion_rate=None,
                is_training=None,
                scope=None,
                # **kwargs,
                ):
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
    images:
    priors:
  Returns:
    segmentations:
  """
  if affine_transform:
    assert prior_segs is not None

  if deformable_transform:
    assert prior_segs is not None and prior_imgs is not None
    
  output_dict = {}
  h, w = images.get_shape().as_list()[1:3]
  # images = kwargs.pop(common.IMAGE, None)
  # labels = kwargs.pop(common.LABEL, None)
  # prior_imgs = kwargs.pop(common.PRIOR_IMGS, None)
  # prior_segs = kwargs.pop(common.PRIOR_SEGS, None)
  # labels = kwargs.pop(common.IMAGE, None)
  # labels = kwargs.pop(common.IMAGE, None)
    
  with tf.variable_scope(scope, 'pgb_network') as sc:
#    features, layers_dict = plain_encoder(images, is_training=is_training)
    features, end_points = features_extractor.extract_features(images=images,
                                                               output_stride=model_options.output_stride,
                                                               multi_grid=model_options.multi_grid,
                                                               model_variant=model_options.model_variant,
                                                               reuse=tf.AUTO_REUSE,
                                                               is_training=is_training,
                                                               fine_tune_batch_norm=model_options.fine_tune_batch_norm,
                                                               preprocessed_images_dtype=model_options.preprocessed_images_dtype)

    layers_dict = {"low_level1": end_points["pgb_network/resnet_v1_50/conv1_3"],
                   "low_level2": end_points["pgb_network/resnet_v1_50/block1/unit_3/bottleneck_v1/conv1"],
                   "low_level3": end_points["pgb_network/resnet_v1_50/block2/unit_4/bottleneck_v1/conv1"],
                   "low_level4": features}
      
    if prior_imgs is not None or prior_segs is not None:
      prior_img, prior_seg, z_pred = get_prior(features=layers_dict["low_level4"], 
                                               batch_size=batch_size,
                                               num_classes=num_classes, 
                                               num_slices=num_slices,
                                               prior_slices=prior_slices,
                                               z_classes=z_class,
                                               prior_imgs=prior_imgs,
                                               prior_segs=prior_segs, 
                                               z_label_method=z_label_method)
      if prior_img is not None:
        output_dict[common.PRIOR_IMGS] = prior_img
        
      if prior_seg is not None:
        output_dict[common.PRIOR_SEGS] = prior_seg
        
      guidance = get_guidance(layers_dict["low_level4"],
                              images,
                              prior_img,
                              affine_transform=affine_transform,
                              deformable_transform=deformable_transform,
                              prior_seg=prior_seg,
                              is_training=is_training)

      if deformable_transform:
        transformed_imgs, guidance = guidance
        output_dict['transformed_imgs'] = transformed_imgs   
                           
    else:

      guidance = tf.one_hot(
        tf.squeeze(labels, 3), num_classes, on_value=1.0, off_value=0.0)
      if zero_guidance:
        guidance = tf.zeros_like(guidance)
      z_pred = None      

    output_dict['original_guidance'] = guidance
    output_dict[common.GUIDANCE] = guidance

    if model_options.decoder_type == 'refinement_network':
      logits, layers_dict = refinement_network(features=layers_dict["low_level4"],
                                               guidance=guidance,
                                               output_stride=model_options.output_stride,
                                               batch_size=batch_size,
                                               layers_dict=layers_dict,
                                               is_training=is_training)
    elif model_options.decoder_type == 'unet_structure':
      # TODO: unet_structure
      pass
    elif model_options.decoder_type == 'upsample':
      out_feats = tf.image.resize_bilinear(features, [h, w], name='logits')
      logits = slim.conv2d(out_feats, num_classes, [1, 1], stride=1, scope='segmentations')
      
    output_dict[common.OUTPUT_TYPE] = logits
    
    if z_pred is not None:
      output_dict[common.OUTPUT_Z] = z_pred
    
  return output_dict, layers_dict


def get_prior(features, batch_size, num_classes, num_slices, prior_slices, 
              z_classes, prior_imgs, prior_segs, z_label_method, scope=None):
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
      features:
      num_classes:
      priors_volume
      z_label_method
  Returns:
      prior_one_hot:
  """
  # TODO: necessary input
  # TODO: dimension check
  with tf.variable_scope(scope, 'prior_network') as sc:
    if z_label_method == 'regression':
      z_logits = global_extractor(features, output_dims=1, scope='z_info_extractor')
      z_pred = tf.nn.sigmoid(z_logits)
      indices = tf.cast(tf.multiply(tf.cast(prior_slices, tf.float32), z_pred), tf.int32)
      
      # TODO: gather correctly
      batch_idx = tf.range(batch_size)
      indices = tf.stack([batch_idx, indices[:,0]], axis=1)
      if prior_imgs is not None:
        prior_imgs = tf.transpose(prior_imgs, [0,3,2,1])
        prior_img = tf.gather_nd(params=prior_imgs, indices=indices)
        prior_img = tf.transpose(prior_img, [0,2,1])
        prior_img = tf.cast(prior_img, tf.int32)
      else:
        prior_img = None
      if prior_segs is not None:
        prior_segs = tf.transpose(prior_segs, [0,3,2,1])
        prior_seg = tf.gather_nd(params=prior_segs, indices=indices)
        prior_seg = tf.transpose(prior_seg, [0,2,1])
        prior_seg = tf.cast(prior_seg, tf.int32)
      else:
        prior_seg = None
        
    elif  z_label_method == 'classification':
      z_logits = global_extractor(features, output_dims=z_classes, scope='z_info_extractor')
      z_pred = tf.nn.softmax(z_pred, axis=1)
      z_pred = tf.expand_dims(tf.expand_dims(z_pred, axis=1), axis=1)
      if prior_imgs is not None:
        prior_img = tf.multiply(prior_imgs, z_pred)
        prior_img = tf.cast(prior_img, tf.int32)
      else:
        prior_img = None
      if prior_segs is not None:
        prior_seg = tf.multiply(prior_segs, z_pred)
        prior_seg = tf.cast(prior_seg, tf.int32)
      else:
        prior_seg = None
    else:
      ValueError("Incorrect method name")  

    prior_one_hot  = tf.one_hot(indices=prior_seg,
                                depth=num_classes,
                                on_value=1,
                                off_value=0,
                                axis=3) 
    prior_one_hot = tf.cast(prior_one_hot, tf.float32)
    if prior_imgs is not None:
      prior_img = tf.expand_dims(prior_img, axis=3)
    z_pred = tf.squeeze(z_pred, axis=1)

  return prior_img, prior_one_hot, z_pred


def get_guidance(features,
                 images,
                 prior_img,
                 affine_transform,
                 deformable_transform,
                 prior_seg=None,
                 is_training=None):
  # TODO: auxiliary_information need input segmentation and prior segmentation, also need to consider the testing case
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
    images:
    priors:
  Returns:
    segmentations:
  """

  guidance = prior_seg
  if affine_transform:
    theta = extract_non_images(features, output_dims=6, scope='theta_extractor')
    guidance = spatial_transformer_network(guidance, theta)

  if deformable_transform:
    prior_img = tf.cast(prior_img, tf.float32)
    prior_img = spatial_transformer_network(prior_img, theta)
    guidance = voxel_deformable_transform(moving_images=prior_img, 
                                          fixed_images=images,
                                          moving_segs=guidance,
                                          is_training=is_training)  
  return guidance





