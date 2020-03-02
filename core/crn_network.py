'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
slim = tf.contrib.slim
from core import utils
fc_layer = utils.fc_layer
conv2d = utils.conv2d
# from utils import (fc_layer, conv2d, atrous_conv2d, split_separable_conv2d, batch_norm, upsampling_layer)


def refinement_network(features,
                       guidance,
                       output_stride,
                       batch_size,
                       layers_dict,
                       embed=32,
                       is_training=None,
                       scope=None):
    """
    """
    # TODO: necessary for using batch_size??
    # TODO: check guidance shape. The shape [?,256,256,1] should cause error
    # TODO: correct variable scope for feature calling during evaluation
    output = {}
    with tf.variable_scope(scope, 'refinement_network') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d],
            outputs_collections=end_points_collection):
            # guidance_in = guidance
            h, w = layers_dict["low_level4"].get_shape().as_list()[1:3]
            guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
            zero_tensor = tf.zeros([batch_size, h, w, embed])
            if is_training:
                end_points = None
            else:
                end_points = {}
                
            if output_stride == 8:
                upsample_flags = False
            elif output_stride == 16:
                upsample_flags = True
            else:
                ValueError("Unkonwn Number of Output Strides")
                
            output = rm(in_node=layers_dict["low_level4"],
                                    feature=zero_tensor,
                                    guidance=guidance_in,
                                    end_points=end_points,
                                    num_filters=embed,
                                    upsample=upsample_flags,
                                    scope='RM_4',
                                    is_training=is_training)
            if end_points is not None:
                feature1, guidance1, end_points = output
            else:
                feature1, guidance1 = output
            h, w = layers_dict["low_level3"].get_shape().as_list()[1:3]
            # guidance1_gt = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
            # guidance1_a = tf.nn.softmax(guidance1) + guidance1_gt
            guidance1_a = tf.nn.softmax(guidance1)

            output = rm(layers_dict["low_level3"],
                                                feature1,
                                                guidance1_a,
                                                end_points=end_points,
                                                num_filters=embed,
                                                scope='RM_3',
                                                is_training=is_training)
            if end_points is not None:
                feature2, guidance2, end_points = output
            else:
                feature2, guidance2 = output
            h, w = layers_dict["low_level2"].get_shape().as_list()[1:3]
            # guidance2_gt = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
            # guidance2_a = tf.nn.softmax(guidance2) + guidance2_gt
            guidance2_a = tf.nn.softmax(guidance2)
            
            output = rm(layers_dict["low_level2"],
                                                feature2,
                                                guidance2_a,
                                                end_points=end_points,
                                                num_filters=embed,
                                                scope='RM_2',
                                                is_training=is_training)
            if end_points is not None:
                feature3, guidance3, end_points = output
            else:
                feature3, guidance3 = output
            h, w = layers_dict["low_level1"].get_shape().as_list()[1:3]
            # guidance3_gt = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
            # guidance3_a = tf.nn.softmax(guidance3) + guidance3_gt
            guidance3_a = tf.nn.softmax(guidance3)
            
            output = rm(layers_dict["low_level1"],
                                                feature3,
                                                guidance3_a,
                                                end_points=end_points,
                                                num_filters=embed,
                                                classifier=True,
                                                scope='RM_1',
                                                is_training=is_training)
            if end_points is not None:
                feature4, guidance4, end_points = output
            else:
                feature4, guidance4 = output

            layers_dict.update({
                            "guidance_in": guidance_in,
                            "feature1": feature1, "guidance1": guidance1,
                            "feature2": feature2, "guidance2": guidance2,
                            "feature3": feature3, "guidance3": guidance3,
                            "feature4": feature4, "output": guidance4,
                            })
            if not is_training:
                layers_dict.update(end_points)
    return guidance4, layers_dict


# def rm(in_node,
#        feature,
#        guidance,
#        end_points=None,
#        num_filters=32,
#        classifier=False,
#        upsample=True,
#        further_attention=False,
#        class_features=False,
#        low_high_fusion='later',
#        each_class_fusion='concat_and_conv',
#        is_training=None,
#        scope=None):
#     """refining module"""
#     with tf.variable_scope(scope, 'rm'):
#         in_node_shape = in_node.get_shape().as_list()
#         s = in_node_shape[1]
#         channels = in_node_shape[3]
#         num_class = guidance.get_shape().as_list()[3]
#         def extract_semantic(guidance):
#             # TODO: check class and shape
#             h, width = guidance.get_shape().as_list()[1:3]
#             guidance_f = guidance[...,1:]
#             w = tf.reduce_sum(guidance_f, [1, 2])
#             # weight = tf.rsqrt(w)
#             weight = tf.reciprocal(w)
#             mask = tf.is_inf(weight)
#             w = tf.where(mask, x=tf.zeros_like(weight), y=weight)
#             # w = tf.Print(w, [w], "nan?")
#             w = tf.expand_dims(tf.expand_dims(w, axis=1), axis=1)
            
#             w = w * (h*width)
#             # w = tf.sqrt(w)
            
#             new_guid = tf.multiply(guidance_f, w)
#             new_guid = tf.reduce_sum(new_guid, axis=3)
#             new_guid = tf.expand_dims(new_guid, axis=3)
#             return new_guid
#         guidance = extract_semantic(guidance)
#         # guidance = guidance + tf.ones_like(guidance)
        
#         # feature embedding
#         conv_r1 = conv2d(in_node, [1,7,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
#         conv_r1 = conv2d(conv_r1, [7,1,num_filters,num_filters], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training)
#         sram_output = sram(conv_r1, guidance, end_points, scope='attention1', is_training=is_training)
#         if end_points is not None:
#             sram_output, sram_layers = sram_output
#             end_points.update(sram_layers)
#             end_points.update({"fused_guidance": guidance})
#         new_feature = tf.concat([sram_output, feature], axis=3)
#         new_feature = conv2d(new_feature, [3,3,num_filters*2,num_filters], activate=tf.nn.relu, scope="class_fusion", is_training=is_training)

#         # Final output, get guidance and feature in next stage
#         if classifier:
#             kernel_size = 1
#             activate_func=None
#         else:
#             kernel_size = 3
#             activate_func=tf.nn.relu

#         if upsample:
# #            new_guidance = tf.image.resize_bilinear(new_guidance, [2*s, 2*s], name='new_guidance')
#             new_feature = tf.image.resize_bilinear(new_feature, [2*s, 2*s], name='new_feature')

#         output_filters = new_feature.get_shape().as_list()[3]
#         new_guidance = conv2d(new_feature, [kernel_size,kernel_size,output_filters,num_class], activate=activate_func, scope="new_guidance", is_training=is_training)

#     if end_points is not None:
#         return new_feature, new_guidance, end_points
#     else:
#         return new_feature, new_guidance
    
    
def rm(in_node,
       feature,
       guidance,
       end_points=None,
       num_filters=32,
       classifier=False,
       upsample=True,
       further_attention=False,
       class_features=False,
       low_high_fusion='later',
       each_class_fusion='concat_and_conv',
       is_training=None,
       scope=None):
    """refining module"""
    with tf.variable_scope(scope, 'rm'):
        in_node_shape = in_node.get_shape().as_list()
        s = in_node_shape[1]
        channels = in_node_shape[3]
        num_class = guidance.get_shape().as_list()[3]
        
        # guidance = guidance + tf.ones_like(guidance)
        
        # feature embedding
        conv_r1 = conv2d(in_node, [1,7,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
        conv_r1 = conv2d(conv_r1, [7,1,num_filters,num_filters], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training)
        if end_points is not None:
            end_points[scope+'/sram_embeded'] = conv_r1

        feature_list = []
        for i in range(num_class):
            with tf.variable_scope('SRAM_{}'.format(i)):
                # TODO: further_attention; add second SRAM
                sram_output = sram(conv_r1, guidance[...,i:i+1], end_points, scope='attention1', is_training=is_training)
                if end_points is not None:
                    attention1, end_points = sram_output
                else:
                    attention1 = sram_output

                # low-level, high-level feature fusion
                if class_features:
                    if low_high_fusion == 'summation':
                        fusion = tf.add(attention1, feature[...,num_filters*i:num_filters*(i+1)])
                    elif low_high_fusion == 'concat_and_conv':
                        fusion = tf.concat([attention1, feature[...,num_filters*i:num_filters*(i+1)]], axis=3)
                        fusion = conv2d(fusion, [3,3,num_filters*2,num_filters], activate=tf.nn.relu, scope="fusion", is_training=is_training)
                else:
                    if low_high_fusion == 'summation':
                        fusion = tf.add(attention1, feature)
                    elif low_high_fusion == 'concat_and_conv':
                        fusion = tf.concat([attention1, feature], axis=3)
                        fusion = conv2d(fusion, [3,3,num_filters*2,num_filters], activate=tf.nn.relu, scope="fusion", is_training=is_training)
                    elif low_high_fusion == 'later':
                        fusion = attention1

            feature_list.append(fusion)
            if end_points is not None:
                end_points[scope+'/sram_output_class{}'.format(i)] = attention1
                end_points[scope+'/low_high_feature_fusion'] = fusion

        if low_high_fusion == 'later':
            feature_list.append(feature)

        # feature fusion in different class
        if class_features:
            new_feature = tf.concat(feature_list, axis=3)
        else:
            if each_class_fusion == 'concat_and_conv':
                new_feature = tf.concat(feature_list, axis=3)
                if low_high_fusion == 'later':
                    new_feature = conv2d(new_feature, [3,3,num_filters*(num_class+1),num_filters], activate=tf.nn.relu, scope="class_fusion", is_training=is_training)
                else:
                    new_feature = conv2d(new_feature, [3,3,num_filters*num_class,num_filters], activate=tf.nn.relu, scope="class_fusion", is_training=is_training)
            elif each_class_fusion == 'summation':
                new_feature = tf.add_n(feature_list)

        # Final output, get guidance and feature in next stage
        if classifier:
            kernel_size = 1
            activate_func=None
        else:
            kernel_size = 3
            activate_func=tf.nn.relu

        if upsample:
#            new_guidance = tf.image.resize_bilinear(new_guidance, [2*s, 2*s], name='new_guidance')
            new_feature = tf.image.resize_bilinear(new_feature, [2*s, 2*s], name='new_feature')

        output_filters = new_feature.get_shape().as_list()[3]
        new_guidance = conv2d(new_feature, [kernel_size,kernel_size,output_filters,num_class], activate=activate_func, scope="new_guidance", is_training=is_training)

    if end_points is not None:
        return new_feature, new_guidance, end_points
    else:
        return new_feature, new_guidance


def sram(in_node,
         guidance,
         end_points=None,
         is_gamma=False,
         scope=None,
         is_training=True):
    """Single Residual Attention Module"""
    with tf.variable_scope(scope, "sram"):
        channels = in_node.get_shape().as_list()[3]
        conv1 = conv2d(in_node, [3,3,channels,channels], activate=tf.nn.relu, scope="conv1", is_training=is_training)
        conv2 = conv2d(conv1, [3,3,channels,channels], activate=tf.nn.relu, scope="conv2", is_training=is_training)

        guidance_tile = tf.tile(guidance, [1,1,1,channels])

        if is_gamma:
            gamma = tf.Variable(0, dtype=tf.float32)
            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
        else:
            output = in_node + tf.multiply(conv2, guidance_tile)

        if end_points is not None:
            end_points[scope+'/sram_conv1'] = conv1
            end_points[scope+'/sram_conv2'] = conv2
            return output, end_points
        else:
            return output