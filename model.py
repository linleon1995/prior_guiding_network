import tensorflow as tf
# TODO: Remove numpy dependency
import numpy as np
# from tensorflow.python.ops import math_ops
from core import features_extractor, stn, voxelmorph, crn_network, utils
import common

slim = tf.contrib.slim
spatial_transformer_network = stn.spatial_transformer_network
voxel_deformable_transform = voxelmorph.voxel_deformable_transform
refinement_network = crn_network.refinement_network
mlp = utils.mlp
__2d_unet_decoder = utils.__2d_unet_decoder

# TODO: testing case (validation)
# TODO: voxelmorph in 2d or 3d

def pgb_network(images, 
                model_options,
                affine_transform,
                # deformable_transform,
                labels=None,
                # prior_imgs=None,
                prior_segs=None,
                num_class=None,
                prior_slice=None,
                batch_size=None,
                z_label_method=None,
                # z_label=None,
                z_class=None,
                guidance_type=None,
                fusion_slice=None,
                prior_dir=None,
                is_training=None,
                scope=None,
                # **kwargs,
                ):
  """
  VoxelMorph: A Learning Framework for Deformable Medical Image Registration
  Args:
    images:
    prior_segs: [1,H,W,Class,Layer]
  Returns:
    segmentations:
  """
  output_dict = {}
  h, w = images.get_shape().as_list()[1:3]
  # images = kwargs.pop(common.IMAGE, None)
  # labels = kwargs.pop(common.LABEL, None)
  # prior_imgs = kwargs.pop(common.PRIOR_IMGS, None)
  # prior_segs = kwargs.pop(common.PRIOR_SEGS, None)
  # labels = kwargs.pop(common.IMAGE, None)
  # labels = kwargs.pop(common.IMAGE, None)
    
  with tf.variable_scope(scope, 'pgb_network'):
    features, end_points = features_extractor.extract_features(images=images,
                                                               output_stride=model_options.output_stride,
                                                               multi_grid=model_options.multi_grid,
                                                               model_variant=model_options.model_variant,
                                                               reuse=tf.AUTO_REUSE,
                                                               is_training=is_training,
                                                               fine_tune_batch_norm=model_options.fine_tune_batch_norm,
                                                               preprocessed_images_dtype=model_options.preprocessed_images_dtype)

    layers_dict = {"low_level5": features,
                   "low_level4": end_points["pgb_network/resnet_v1_50/block3"],
                   "low_level3": end_points["pgb_network/resnet_v1_50/block2"],
                   "low_level2": end_points["pgb_network/resnet_v1_50/block1"],
                   "low_level1": end_points["pgb_network/resnet_v1_50/conv1_3"]}
    
    if z_label_method is not None:
        z_pred = predict_z_dimension(features, z_label_method, z_class)
        output_dict[common.OUTPUT_Z] = z_pred
        
    if model_options.decoder_type == 'refinement_network':
        if guidance_type == "adaptive":
            prior_seg = get_adaptive_guidance(prior_segs, z_pred, z_label_method, num_class, 
                                              prior_slice, fusion_slice)
        elif guidance_type == "come_from_feature":
            prior_seg = slim.conv2d(features, num_class, [1, 1], 1, activation_fn=None, scope='input_guidance')
            prior_seg = tf.nn.softmax(prior_seg)
        else:
            prior_seg = prior_segs
            
        # if guidance_type == "training_data_fusion":
        #     prior_seg = prior_segs
        # elif guidance_type == "zeros":
        #     prior_shape = features.get_shape().as_list()[0:3]
        #     prior_shape.append(num_class)
        #     prior = tf.zeros(prior_shape)
        # elif guidance_type == "adaptive":
        #     prior_seg = get_adaptive_guidance(prior_segs, z_pred, z_label_method, num_class, 
        #                                       prior_slice, fusion_slice)
        # elif guidance_type == "come_from_feature":
        #     prior_seg = slim.conv2d(features, num_class, [1, 1], 1, activation_fn=None, scope='input_guidance')
        # elif guidance_type == "ground_truth":
        #     prior_seg = tf.one_hot(tf.squeeze(labels, 3), num_class, on_value=1.0, off_value=0.0)
        # else:
        #     raise ValueError("Unknown guidace type")
          
        
        # TODO: check prior shape
        
        if affine_transform:
            # TODO: stn split in class --> for loop
            # TODO: variable scope for spatial transform
            output_dict['original_guidance'] = prior_seg
            
            gap_featue = tf.reduce_mean(features, [1, 2], name='global_avg_pool', keep_dims=False)
            theta = mlp(gap_featue, output_dims=6, scope='theta_extractor')
            prior_seg = spatial_transformer_network(prior_seg, theta)
                    
        output_dict[common.GUIDANCE] = prior_seg
    
        logits, layers_dict = refinement_network(features,
                                                prior_seg,
                                                model_options.output_stride,
                                                batch_size,
                                                layers_dict,
                                                num_class,
                                                further_attention=False,
                                                class_split_in_each_stage=False,
                                                input_guidance_in_each_stage=False,
                                                is_training=is_training)
    elif model_options.decoder_type == 'unet_structure':
        logits = __2d_unet_decoder(features, layers_dict, num_class, channels=256, is_training=is_training)
    elif model_options.decoder_type == 'upsample':
        out_feats = tf.image.resize_bilinear(features, [h, w], name='logits')
        logits = slim.conv2d(out_feats, num_class, [1, 1], stride=1, scope='segmentations')
    for v in tf.trainable_variables():
        print(v)
        print(30*"-")
    output_dict[common.OUTPUT_TYPE] = logits
  return output_dict, layers_dict


def get_slice_indice(indice, num_prior_slice, fused_slice):
    # TODO: fused slice could be even number
    # TODO: Non-uniform select --> multiply mask
    # Make sure num_prior_slice always bigger than fused_slice
    print(indice, 50*"o")
    # From prob to slice index
    indice = tf.multiply(indice, num_prior_slice)
    print(indice, 50*"o")
    # Shifhting to avoid out of range access
    if fused_slice > 1:
        shift = tf.divide(tf.cast(fused_slice, tf.int32), 2)
        low = shift
        high = num_prior_slice - shift
        indice = tf.clip_by_value(indice, low, high)
        start = indice - shift
        end = indice + shift
        indice = tf.concat([start, end], axis=1)
    print(indice, 50*"o")
    # Select the neighbor slices
    def _range_fn(tp):
        tp0 = tp[0]
        tp1 = tp[1] + 1
        return tf.range(tp0, tp1)
    indice = tf.map_fn(lambda tp: _range_fn(tp), indice)
    print(indice, 50*"o")
    # One-hot encoding
    indice = tf.one_hot(indices=indice,
                        depth=num_prior_slice,
                        on_value=1,
                        off_value=0,
                        axis=1) 
    print(indice, 50*"o")
    return indice
    

def predict_z_dimension(features, z_label_method, z_class):
    with tf.variable_scope('z_prediction_branch'):
        gap = tf.reduce_mean(features, axis=[1,2], keep_dims=False)
        if z_label_method == 'regression':
            z_logits = mlp(gap, output_dims=1, scope='z_info_extractor')
            z_pred = tf.nn.sigmoid(z_logits)
            z_pred = tf.squeeze(z_pred, axis=1)
        elif z_label_method == "classification":
            z_logits = mlp(gap, output_dims=z_class, scope='z_info_extractor')
            z_pred = tf.nn.softmax(z_logits, axis=1)
            z_pred = tf.expand_dims(tf.expand_dims(z_pred, axis=1), axis=1)  
        else:
            raise ValueError("Unknown z prediction model type")
        
    return z_pred


def get_adaptive_guidance(prior_segs, 
                          z_pred, 
                          z_label_method, 
                          num_class=None, 
                          prior_slice=None, 
                          fusion_slice=None):
    if z_label_method == "regression":
        indice = get_slice_indice(indice=z_pred, num_prior_slice=prior_slice, fused_slice=fusion_slice)
        prior_seg = tf.matmul(prior_segs, indice)
        # TODO: get average?
    elif  z_label_method == 'classification':
        prior_seg = tf.multiply(prior_segs, z_pred)
    else:
        raise ValueError("Unknown model type for z predicition")

    if num_class is not None:
        prior_seg = tf.cast(prior_seg, tf.int32)
        prior_seg  = tf.one_hot(prior_seg, num_class, 1, 0, axis=3) 
        prior_seg = tf.cast(prior_seg, tf.float32)
    return prior_seg


    




