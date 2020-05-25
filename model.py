import tensorflow as tf
# TODO: Remove numpy dependency
import numpy as np
# from tensorflow.python.ops import math_ops
from core import features_extractor, stn, voxelmorph, crn_network, utils
from test_flownet import build_flow_model, FlowNetS
import common
import experiments
import math
spatial_transfom_exp = experiments.spatial_transfom_exp

slim = tf.contrib.slim
spatial_transformer_network = stn.spatial_transformer_network
bilinear_sampler = stn.bilinear_sampler
voxel_deformable_transform = voxelmorph.voxel_deformable_transform
refinement_network = crn_network.refinement_network
mlp = utils.mlp
conv2d = utils.conv2d
__2d_unet_decoder = utils.__2d_unet_decoder
# extractor = utils.simple_extractor

# TODO: testing case (validation)
# TODO: voxelmorph in 2d or 3d



# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11

class WarpingLayer(object):
    def __init__(self, warp_type = 'nearest', name = 'warping'):
        self.warp = warp_type
        self.name = name

    def __call__(self, x, flow):
        # expect shape
        # x:(#batch, height, width, #channel)
        # flow:(#batch, height, width, 2)
        with tf.name_scope(self.name) as ns:
            assert self.warp in ['nearest', 'bilinear']
            if self.warp == 'nearest':
                x_warped = nearest_warp(x, flow)
            else:
                x_warped = bilinear_warp(x, flow)
            return x_warped
        
        
def pgb_network(images, 
                model_options,
                affine_transform,
                # deformable_transform,
                samples=None,
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
                drop_prob=None,
                stn_in_each_class=None,
                reuse=None,
                is_training=None,
                scope=None,
                **kwargs,
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
    weight_decay = kwargs.pop("weight_decay", None)
    fusions = kwargs.pop("fusions", None)
    out_node = kwargs.pop("out_node", None)  
    guid_encoder = kwargs.pop("guid_encoder", None)
    z_model = kwargs.pop("z_model", None)
    guidance_loss = kwargs.pop("guidance_loss", None)
    stage_pred_loss = kwargs.pop("stage_pred_loss", None)
    guid_conv_nums = kwargs.pop("guid_conv_nums", None)
    guid_conv_type = kwargs.pop("guid_conv_type", None)
    # Produce Prior
    prior_seg = get_prior(prior_segs, guidance_type, num_class)
                
    if guid_encoder == "last_stage_feature":
        in_node = tf.concat([images, prior_seg], axis=3)
    else:
        in_node = images
    features, end_points = features_extractor.extract_features(images=in_node,
                                                               output_stride=model_options.output_stride,
                                                               multi_grid=model_options.multi_grid,
                                                               model_variant=model_options.model_variant,
                                                               reuse=reuse,
                                                               is_training=is_training,
                                                               fine_tune_batch_norm=model_options.fine_tune_batch_norm,
                                                               preprocessed_images_dtype=model_options.preprocessed_images_dtype)

    layers_dict = {"low_level5": features,
                   "low_level4": end_points["resnet_v1_50/block3"],
                   "low_level3": end_points["resnet_v1_50/block2"],
                   "low_level2": end_points["resnet_v1_50/block1"],
                   "low_level1": end_points["resnet_v1_50/conv1_3"]}
    
    # Multi-task 
    if z_label_method is not None:
        if z_label_method.split("_")[1] == "regression":
            multi_task_node = 1
        elif z_label_method.split("_")[1] == "classification": 
            if z_label_method.split("_")[0] == "z": 
                multi_task_node = z_class
            elif z_label_method.split("_")[0] == "organ": 
                multi_task_node = num_class
            
        z_logits = predict_z_dimension(features, out_node=multi_task_node, 
                                       extractor_type="simple")
        output_dict[common.OUTPUT_Z] = z_logits
    
    with slim.arg_scope([slim.batch_norm],
                        is_training=is_training):
        with slim.arg_scope([slim.conv2d], 
                          trainable=True,
                          activation_fn=tf.nn.relu, 
                          weights_initializer=tf.initializers.he_normal(), 
                          weights_regularizer=slim.l2_regularizer(weight_decay),
                          kernel_size=[3, 3], 
                          padding='SAME',
                          normalizer_fn=slim.batch_norm):
            if "guid" in fusions or "guid_class" in fusions or "guid_uni" in fusions:
                # Refined by Decoder
                if guid_encoder == "last_stage_feature":
                    prior_seg = slim.conv2d(layers_dict["low_level4"], out_node, kernel_size=[1,1], scope="guidance_embedding")
                elif guid_encoder == "shallow_net":
                    prior_seg = utils.get_guidance(tf.concat([images, prior_seg], axis=3), out_node)
                    
                prior_pred = slim.conv2d(prior_seg, num_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='prior_pred')
                output_dict[common.GUIDANCE] = prior_pred
                
                if "softmax" in guidance_loss:
                    prior_pred = tf.nn.softmax(prior_pred, axis=3)
                elif "sigmoid" in guidance_loss:
                    prior_pred = tf.nn.sigmoid(prior_pred)
                    
            else:
                prior_seg = None
                prior_pred = None

    refine_model = utils.Refine(layers_dict, fusions, prior=prior_seg, stage_pred_loss=stage_pred_loss, 
                                prior_pred=prior_pred, guid_conv_nums=guid_conv_nums, guid_conv_type=guid_conv_type, 
                                embed_node=out_node, weight_decay=weight_decay, is_training=is_training)  
    logits, preds = refine_model.model()    
    # logits, preds = refine_by_decoder(images, prior_seg, prior_pred, stage_pred_loss, layers_dict, fusions, 
    #                                   out_node=out_node, weight_decay=weight_decay, reuse=reuse, 
    #                                   is_training=is_training)
    layers_dict.update(preds)
    
    if drop_prob is not None:
        logits = tf.nn.dropout(logits, rate=drop_prob)
    
    # logits = tf.identity(logits, "output")    
    output_dict[common.OUTPUT_TYPE] = logits    
    
    aa = tf.trainable_variables()
    for v in aa:
      print(30*"-", v.name)
    return output_dict, layers_dict


# def pgb_network(images, 
#                 model_options,
#                 affine_transform,
#                 # deformable_transform,
#                 samples=None,
#                 # prior_imgs=None,
#                 prior_segs=None,
#                 num_class=None,
#                 prior_slice=None,
#                 batch_size=None,
#                 z_label_method=None,
#                 # z_label=None,
#                 z_class=None,
#                 guidance_type=None,
#                 fusion_slice=None,
#                 prior_dir=None,
#                 drop_prob=None,
#                 guid_weight=None,
#                 stn_in_each_class=None,
#                 reuse=None,
#                 is_training=None,
#                 scope=None,
#                 **kwargs,
#                 ):
#     """
#     VoxelMorph: A Learning Framework for Deformable Medical Image Registration
#     Args:
#         images:
#         prior_segs: [1,H,W,Class,Layer]
#     Returns:
#         segmentations:
#     """
#     output_dict = {}
#     h, w = images.get_shape().as_list()[1:3]
#     flow_model_type = kwargs.pop("flow_model_type", None)
#     weight_decay = kwargs.pop("weight_decay", None)
#     # images = kwargs.pop(common.IMAGE, None)
#     # labels = kwargs.pop(common.LABEL, None)
#     # prior_imgs = kwargs.pop(common.PRIOR_IMGS, None)
#     # prior_segs = kwargs.pop(common.PRIOR_SEGS, None)
#     # labels = kwargs.pop(common.IMAGE, None)
#     # labels = kwargs.pop(common.IMAGE, None)
    
#     # with tf.variable_scope(scope, 'pgb_network', reuse=reuse):
#     if guidance_type in ("training_data_fusion", "training_data_fusion_h"):
#         prior_seg = prior_segs[...,0]
#         # TODO: if tf.rank<4
#         # prior_segs = tf.split(prior_segs, num_or_size_splits=z_class, axis=3)
#         # prior_segs = tf.concat(prior_segs, axis=2)
#         # prior_segs = tf.squeeze(prior_segs, axis=3)
#     elif guidance_type == "gt":
#         prior_seg = tf.one_hot(indices=prior_segs[...,0],
#                                 depth=num_class,
#                                 on_value=1,
#                                 off_value=0,
#                                 axis=3)                     
#     else:
#         prior_seg = prior_segs
        
#     in_node = images
#     if affine_transform:
#         flow_model_inputs = tf.concat([images, prior_seg], axis=3)
#         if flow_model_type=="resnet_decoder":
#             in_node = flow_model_inputs
            
#     if flow_model_type in ("unet_3_32_p", "unet_3_32_ps", "unet_guid_p"):
#         inputs = tf.concat([images, prior_seg], axis=3)    
#     elif flow_model_type in ("unet_3_32", "unet_3_32s", "unet_guid"):
#         inputs = images     
        
#     # if flow_model_type == "unet_5_32":
#     #     feature = utils._simple_unet(inputs, out=32, stage=5, channels=32, is_training=is_training)
#     # elif flow_model_type == "unet_3_64":
#     #     feature = utils._simple_unet(inputs, out=32, stage=3, channels=64, is_training=is_training)
#     # elif flow_model_type in ("unet_3_32", "unet_3_32_p"):
#     #     feature = utils._simple_unet(inputs, out=32, stage=3, channels=32, is_training=is_training)
#     # elif flow_model_type in ("unet_3_32s", "unet_3_32_ps"):    
#     #     feature = utils.slim_unet(inputs, num_stage=3, weight_decay=weight_decay, is_training=is_training)
#     # elif flow_model_type == "unet_5_32_GCN":
#     #     feature = utils._simple_unet(inputs, out=32, stage=5, channels=32, is_training=is_training)
#     #     feature = utils.GCN(feature, 32, ks=7, is_training=is_training)
#     # elif flow_model_type == "unet_3_64_GCN":
#     #     feature = utils._simple_unet(inputs, out=32, stage=3, channels=64, is_training=is_training)
#     #     feature = utils.GCN(feature, 32, ks=7, is_training=is_training)
#     # # elif flow_model_type == "fpn_3_256":
#     # #     feature = utils.slim_fpn(inputs, num_stage=3, weight_decay=weight_decay, is_training=is_training)
        
#     # logits = conv2d(feature, [1,1,32,num_class], is_training=is_training, scope="logits")
#     # # prior_seg = tf.nn.softmax(logits, axis=3)
#     # layers_dict = None
    
#     feature = utils.slim_extractor(inputs, num_stage=3, weight_decay=weight_decay, is_training=is_training)
#     prior_seg = slim.conv2d(feature , 14, kernel_size=[1, 1], weights_initializer=tf.initializers.he_normal(), 
#                           weights_regularizer=slim.l2_regularizer(weight_decay),scope="guidance")
#     output_dict[common.GUIDANCE] = prior_seg
#     prior_seg = tf.nn.sigmoid(prior_seg)
#     features, end_points = features_extractor.extract_features(images=in_node,
#                                                             output_stride=model_options.output_stride,
#                                                             multi_grid=model_options.multi_grid,
#                                                             model_variant=model_options.model_variant,
#                                                             reuse=tf.AUTO_REUSE,
#                                                             is_training=is_training,
#                                                             fine_tune_batch_norm=model_options.fine_tune_batch_norm,
#                                                             preprocessed_images_dtype=model_options.preprocessed_images_dtype)

#     layers_dict = {"low_level5": features,
#                 "low_level4": end_points["resnet_v1_50/block3"],
#                 "low_level3": end_points["resnet_v1_50/block2"],
#                 "low_level2": end_points["resnet_v1_50/block1"],
#                 "low_level1": end_points["resnet_v1_50/conv1_3"]}
    
#     if z_label_method is not None:
#         z_pred = predict_z_dimension(features, z_label_method, z_class)
#         output_dict[common.OUTPUT_Z] = z_pred
#         if z_to_prior:
#             prior_seg = get_adaptive_guidance(prior_segs, z_pred, z_label_method, num_class, 
#                                             prior_slice, fusion_slice)

#     if model_options.decoder_type == 'refinement_network':
        

#         # TODO: check prior shape
        
#         if affine_transform:
#             # output_dict = build_flow_model(inputs, samples, flow_model_type, model_options, learning_cases="img-prior")


#             # with tf.variable_scope("flow_model"):
#                 output_dict["original_guidance"] = prior_seg
#                 if flow_model_type == "unet":
#                     flow = utils._simple_unet(flow_model_inputs, out=2, stage=5, channels=32, is_training=True)
#                 elif flow_model_type == "resnet_decoder":
#                     flow = utils._simple_decoder(features, out=2, stage=3, channels=32, is_training=True)
#                 elif flow_model_type == "FlowNet-S":
#                     training_schedule = {
#                         # 'step_values': [400000, 600000, 800000, 1000000],
#                         'step_values': [400000, 600000, 800000, 1000000],
#                         'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
#                         'momentum': 0.9,
#                         'momentum2': 0.999,
#                         'weight_decay': 0.0004,
#                         'max_iter': 30000,
#                         }
#                     net = FlowNetS()
#                     flow_model_inputs = {"input_a": images, "input_b": prior_seg}
#                     flow_dict = net.model(flow_model_inputs, training_schedule, trainable=True)
#                     flow = flow_dict["flow"]

#                 warp_func = WarpingLayer('bilinear')
                
#                 prior_seg = warp_func(prior_seg, flow)
#                 prior_seg = tf.cast(prior_seg, tf.float32)
#                 output_dict["flow"] = flow
                            
#             # TODO: stn split in class --> for loop
#             # TODO: variable scope for spatial transform
        
#         # if "organ_label" in samples:
#         # TODO:
#         # organ_label = tf.reshape(samples, [batch_size,1,1,num_class])
#         # organ_label = tf.cast(organ_label, tf.float32)
#         # prior_seg = prior_seg * organ_label
            
#         output_dict[common.GUIDANCE] = prior_seg
#         # prior_seg = tf.stop_gradient(prior_seg)
#         logits, layers_dict = refinement_network(
#                                                 features,
#                                                 prior_seg,
#                                                 model_options.output_stride,
#                                                 batch_size,
#                                                 layers_dict,
#                                                 num_class,
#                                                 kwargs,
#                                                 further_attention=False,
#                                                 rm_type="feature_guided",
#                                                 input_guidance_in_each_stage=False,
#                                                 is_training=is_training,
#                                                 )
#     elif model_options.decoder_type == 'unet_structure':
#         logits = __2d_unet_decoder(features, layers_dict, num_class, channels=256, is_training=is_training)
#     elif model_options.decoder_type == 'upsample':
#         # features =slim.conv2d(features, 256, [3, 3], stride=1, scope='embed1')
#         # features = tf.compat.v2.image.resize(features, [64, 64])
#         # features =slim.conv2d(features, 128, [3, 3], stride=1, scope='embed2')
#         # features = tf.compat.v2.image.resize(features, [128, 128])
#         features =slim.conv2d(features, 256, [3, 3], stride=1, scope='embed3')
#         # features = tf.compat.v2.image.resize(features, [256, 256])
#         features = tf.image.resize_bilinear(features, [256,256], align_corners=True)
#         logits = slim.conv2d(features, num_class, [1, 1], stride=1, scope='logits')
        
#     if drop_prob is not None:
#         logits = tf.nn.dropout(logits, rate=drop_prob)
#     # for v in tf.trainable_variables():
#     #     print(v)
#     #     print(30*"-")
#     output_dict[common.OUTPUT_TYPE] = logits
#     return output_dict, layers_dict



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
    

def predict_z_dimension(feature, out_node, extractor_type):
    with tf.variable_scope('multi_task_branch'):
        # TODO: neighbor slices
        if extractor_type == "simple":
            gap = tf.reduce_mean(feature, axis=[1,2], keep_dims=False)
            z_logits = mlp(gap, output_dims=out_node, num_layers=2, 
                           decreasing_root=16, scope='z_info_extractor')
        elif extractor_type == "region":
            pass
        else:
            raise ValueError("Unknown Extractor Type")
    return z_logits


# def predict_z_dimension(features, z_label_method, z_class):
#     with tf.variable_scope('z_prediction_branch'):
#         gap = tf.reduce_mean(features, axis=[1,2], keep_dims=False)
#         if z_label_method == 'regression':
#             z_logits = mlp(gap, output_dims=1, scope='z_info_extractor')
#             z_pred = tf.nn.sigmoid(z_logits)
#             z_pred = tf.squeeze(z_pred, axis=1)
#         elif z_label_method == "classification":
#             z_logits = mlp(gap, output_dims=z_class, scope='z_info_extractor')
#             z_pred = tf.nn.softmax(z_logits, axis=1)
#             z_pred = tf.expand_dims(tf.expand_dims(z_pred, axis=1), axis=1)  
#         else:
#             raise ValueError("Unknown z prediction model type")
        
#     return z_pred


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


def get_prior(prior_segs, guidance_type, num_class):
    # TODO: else guidance type should raise ValueError
    if guidance_type in ("training_data_fusion", "training_data_fusion_h"):
        prior_seg = prior_segs[...,0]
        # TODO: if tf.rank<4
        # prior_segs = tf.split(prior_segs, num_or_size_splits=z_class, axis=3)
        # prior_segs = tf.concat(prior_segs, axis=2)
        # prior_segs = tf.squeeze(prior_segs, axis=3)
    elif guidance_type == "gt":
        prior_seg = tf.one_hot(indices=prior_segs[...,0],
                                depth=num_class,
                                on_value=1,
                                off_value=0,
                                axis=3)                     
    else:
        prior_seg = prior_segs
    return prior_seg
    

# def refine_by_decoder(images, prior_seg, prior_pred, stage_pred_loss, layers_dict, fusions, out_node, 
#                       fine_tune_batch_norm=True, weight_decay=0.0, reuse=None, is_training=None):
#     # batch_norm_params = utils.get_batch_norm_params(
#     #     decay=0.9997,
#     #     epsilon=1e-5,
#     #     scale=True,
#     #     is_training=(is_training and fine_tune_batch_norm),
#     #     # sync_batch_norm_method=model_options.sync_batch_norm_method
#     #     )
#     # # batch_norm = utils.get_batch_norm_fn(
#     # #     model_options.sync_batch_norm_method)
#     # batch_norm = slim.batch_norm
#     # with slim.arg_scope(
#     #     [slim.conv2d, slim.separable_conv2d],
#     #     weights_regularizer=slim.l2_regularizer(weight_decay),
#     #     activation_fn=tf.nn.relu,
#     #     normalizer_fn=slim.batch_norm,
#     #     padding='SAME',
#     #     stride=1,
#     #     reuse=reuse):
#     #     with slim.arg_scope([batch_norm], **batch_norm_params):
#     refine_model = utils.Refine(layers_dict, fusions, prior=prior_seg, stage_pred_loss=stage_pred_loss, 
#                                 prior_pred=prior_pred, guid_conv_nums=64, guid_conv_type="conv", 
#                                 embed_node=out_node, weight_decay=weight_decay, is_training=is_training)  
#     logits, preds = refine_model.model()
#     return logits, preds

