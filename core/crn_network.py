'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals
import math
import tensorflow as tf
from core import utils
fc_layer = utils.fc_layer
conv2d = utils.conv2d
GCN = utils.GCN


def rm_new(in_node,
            num_filters,
             guidance,
             feature,
             num_class=None,
             scale=None,
             further_attention=False,
             weight_sharing=False,
             is_training=None,
             scope=None):
    """refining module
    feature: [H,W,Channels]
    guidance: [H,W,Class]

    todo
    conv_r1: 2 conv3x3 -> 1 conv1x1 for channel reduction and to keep the inform
    conv3x3 pred for antialias of upsampling
    
    """
    
    with tf.variable_scope(scope, 'rm_new'):
        h, w = in_node.get_shape().as_list()[1:3]
        # num_class = guidance.get_shape().as_list()[3]
        # num_filters = feature.get_shape().as_list()[3]
        channels = in_node.get_shape().as_list()[3]

        # feature embedding
        conv_r1 = conv2d(in_node, [1,1,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
        conv_r1 = tf.tile(conv_r1, [1,1,1,num_class-1])
        
        # sram attention
        new_feature = conv_r1 + conv_r1 * feature
        
        new_guidance = conv2d(new_feature, [1,1,num_filters*(num_class-1),num_class], 
                              activate=None, scope="guidance", is_training=is_training)

        if scale is not None or scale != 1:
            new_feature = tf.compat.v2.image.resize(new_feature, [scale*h, scale*w], name='new_feature')  
            new_guidance = tf.compat.v2.image.resize(new_guidance, [scale*h, scale*w], name='new_guidance')                      
    return new_feature, new_guidance


def rm_ori(in_node,
            num_filters,
             guidance,
             feature,
             num_class=None,
             scale=None,
             further_attention=False,
             weight_sharing=False,
             is_training=None,
             scope=None):
    """refining module
    feature: [H,W,Channels]
    guidance: [H,W,Class]
    """
    with tf.variable_scope(scope, 'rm_ori'):
        h, w = in_node.get_shape().as_list()[1:3]
        # num_class = guidance.get_shape().as_list()[3]
        # num_filters = feature.get_shape().as_list()[3]
        channels = in_node.get_shape().as_list()[3]

        # feature embedding
        conv_r1 = conv2d(in_node, [3,3,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
        conv_r1 = conv2d(conv_r1, [3,3,num_filters,num_filters], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training)
        
        # conv_r1 = GCN(in_node, num_filters, ks=7, is_training=is_training)
        
        # sram attention
        def sram_attention_branch(sram_input, scope):
            feature_list = []
            for c in range(1, num_class):
                if not weight_sharing:
                    sram_scope = "_".join([scope, str(c)])
                else:
                    sram_scope = scope
                attention_class = sram(sram_input, guidance[...,c:c+1], scope=sram_scope, is_training=is_training)
                
                feature_list.append(attention_class)
            attention = tf.concat(feature_list, axis=3)
            
            if feature is not None:
                attention += feature
            # if feature is not None and scope == "sram2":
            #     attention += feature
            return attention
        
        new_feature = sram_attention_branch(sram_input=conv_r1, scope="sram1")
        if further_attention:
            new_feature = sram_attention_branch(sram_input=new_feature, scope="sram2")
        
        new_guidance = conv2d(new_feature, [1,1,num_filters*(num_class-1),num_class], 
                              activate=None, scope="guidance", is_training=is_training)
        if scale is not None or scale != 1:
            new_feature = tf.compat.v2.image.resize(new_feature, [scale*h, scale*w], name='new_feature')
            new_guidance = tf.compat.v2.image.resize(new_guidance, [scale*h, scale*w], name='new_guidance')

    return new_feature, new_guidance


def refinement_network(features,
                       guidance,
                       output_stride,
                       batch_size,
                       layers_dict,
                       num_class=None,
                       kwargs=None,
                       embed=32,
                       further_attention=False,
                       rm_type=None,
                       input_guidance_in_each_stage=False,
                       is_training=None,
                       scope=None):
    """
    """
    # TODO: necessary for using batch_size??
    # TODO: check guidance shape. The shape [?,256,256,1] should cause error
    # scale_list = [1,1,1,4,2] # output_strides=8
    scale_list = [1,1,2,2,2] # output_strides=8
    num_stage = len(layers_dict)
    guidance_in = guidance
    
    layers_dict["init_guid"] = guidance
    h, w = layers_dict["low_level5"].get_shape().as_list()[1:3]
    
    guidance = tf.image.resize_bilinear(guidance, [h, w])
    # guidance = tf.ones_like(guidance)
    
    share = kwargs.pop("share", True)    
    guid_acc = kwargs.pop("guidance_acc", None)    
    with tf.variable_scope(scope, 'refinement_network') as sc:
        features = None
        layers_dict["guidance_in"] = guidance
        
        for stage in range(num_stage, 0, -1):
            guidance_last = guidance
            _, guidance = rm_ori(in_node=layers_dict["low_level"+str(stage)],
                                            num_filters=embed,
                                            guidance=guidance,
                                            feature=features,
                                            num_class=14,
                                            scale=scale_list[num_stage-stage],
                                            weight_sharing=share,
                                            further_attention=further_attention,
                                            scope='RM_'+str(stage),
                                            is_training=is_training)  
                                            
            layers_dict["feature"+str(num_stage-stage+1)] = features
            if stage != 1:
                layers_dict["guidance"+str(num_stage-stage+1)] = guidance
                # guidance = tf.nn.softmax(guidance)
                guidance = tf.nn.sigmoid(guidance)
                tf.add_to_collection("guidance", guidance)
                
                guidance = tf.stop_gradient(guidance)
                    
                h, w = guidance.get_shape().as_list()[1:3]
                guidance_last = tf.image.resize_bilinear(guidance_last, [h, w])
                    
                if guid_acc == "acc":
                    guidance = guidance * guidance_last + guidance_last
                elif guid_acc == "sum":
                    guidance = guidance + guidance_last
                elif guid_acc == "acc_init":
                    guidance = guidance * guidance_last + guidance_last
                    guid_in = tf.image.resize_bilinear(guidance_in, [h, w])
                    p = tf.Variable(0.5)
                    guidance = p*guidance + (1-p)*guid_in
                    
                tf.add_to_collection("guidance", guidance)
    return guidance, layers_dict


# def refinement_network(features,
#                        guidance,
#                        output_stride,
#                        batch_size,
#                        layers_dict,
#                        num_class=None,
#                        kwargs=None,
#                        embed=32,
#                        further_attention=False,
#                        rm_type=None,
#                        input_guidance_in_each_stage=False,
#                        is_training=None,
#                        scope=None):
#     """
#     """
#     # TODO: necessary for using batch_size??
#     # TODO: check guidance shape. The shape [?,256,256,1] should cause error
#     # scale_list = [1,1,1,4,2] # output_strides=8
#     scale_list = [1,1,2,2,2] # output_strides=8
#     num_stage = len(layers_dict)
#     guidance_in = guidance
#     layers_dict["init_guid"] = guidance
#     h, w = layers_dict["low_level5"].get_shape().as_list()[1:3]
    
#     guidance = tf.image.resize_bilinear(guidance, [h, w])
#     # guidance = tf.ones_like(guidance)
    
#     share = kwargs.pop("share", True)    
#     guid_acc = kwargs.pop("guidance_acc", None)    
#     with tf.variable_scope(scope, 'refinement_network') as sc:
#         features = None
#         layers_dict["guidance_in"] = guidance
        
#         for stage in range(num_stage, 0, -1):
#             guidance_last = guidance
#             if stage == num_stage:
#                 features, guidance = rm_ori(in_node=layers_dict["low_level"+str(stage)],
#                                             num_filters=embed,
#                                             guidance=guidance,
#                                             feature=features,
#                                             num_class=14,
#                                             scale=scale_list[num_stage-stage],
#                                             weight_sharing=share,
#                                             further_attention=further_attention,
#                                             scope='RM_'+str(stage),
#                                             is_training=is_training) 
#             else:
#                 features, guidance = rm_new(in_node=layers_dict["low_level"+str(stage)],
#                                                 num_filters=embed,
#                                                 guidance=guidance,
#                                                 feature=features,
#                                                 num_class=14,
#                                                 scale=scale_list[num_stage-stage],
#                                                 weight_sharing=share,
#                                                 further_attention=further_attention,
#                                                 scope='RM_'+str(stage),
#                                                 is_training=is_training)  
                                            
#             layers_dict["feature"+str(num_stage-stage+1)] = features
#             if stage != 1:
#                 layers_dict["guidance"+str(num_stage-stage+1)] = guidance
#                 # guidance = tf.nn.softmax(guidance)
#                 guidance = tf.nn.sigmoid(guidance)
#                 tf.add_to_collection("guidance", guidance)
                
#                 guidance = tf.stop_gradient(guidance)
                    
#                 h, w = guidance.get_shape().as_list()[1:3]
#                 guidance_last = tf.image.resize_bilinear(guidance_last, [h, w])
                    
#                 if guid_acc == "acc":
#                     guidance = guidance * guidance_last + guidance_last
#                 elif guid_acc == "sum":
#                     guidance = guidance + guidance_last
                    
#                 tf.add_to_collection("guidance", guidance)
#     return guidance, layers_dict


# def rm_test(in_node,
#         num_filters,
#         guidance,
#         num_class=14,
#         scale=None,
#         further_attention=False,
#         is_training=None,
#         scope=None):
#     """refining module
#     feature: [H,W,Channels]
#     guidance: [H,W,Channels]
#     """
#     with tf.variable_scope(scope, 'rm'):
#         h, w = in_node.get_shape().as_list()[1:3]
#         # num_filters = feature.get_shape().as_list()[3]
        
#         # feature embedding
#         conv_r1 = GCN(in_node, num_filters, ks=7, is_training=is_training)
        
#         # sram attention
#         attention = sram(conv_r1, guidance, scope='sram1', is_training=is_training)
#         new_feature = conv2d(attention, [1,1,num_filters,num_filters], 
#                            activate=tf.nn.relu, scope="fuse", is_training=is_training) 
#         if further_attention:
#             new_feature = sram(new_feature, guidance, scope='sram2', is_training=is_training)
            
#         # if scale is not None or scale != 1:
#         #     new_feature = tf.image.resize_bilinear(new_feature, [scale*h, scale*w], name='new_feature')
      
#     return new_feature


# def refinement_network2(images,
#                        features,
#                        guidance,
#                        output_stride,
#                        batch_size,
#                        layers_dict,
#                        num_class=None,
#                        embed=32,
#                        further_attention=False,
#                        rm_type=None,
#                        input_guidance_in_each_stage=False,
#                        is_training=None,
#                        scope=None):
#     """
#     """
#     # TODO: necessary for using batch_size??
#     # TODO: check guidance shape. The shape [?,256,256,1] should cause error
#     # scale_list = [1,1,1,4,2] # output_strides=8
#     scale_list = [1,1,2,2,2] # output_strides=8
#     num_stage = len(layers_dict)
#     # num_up = int(math.sqrt(output_stride))
#     # num_same = num_stage - num_up
#     # upsample_flags = num_same*[False] + num_up*[True]
        
#     with tf.variable_scope(scope, 'refinement_network') as sc:
#         guidance_in = guidance
#         _, h, w = layers_dict["low_level5"].get_shape().as_list()[0:3]
#         guidance_in_lowres = tf.image.resize_bilinear(guidance, [h, w])
#         zero_tensor = tf.zeros([batch_size, h, w, embed])
#         feature = zero_tensor
            
#         layers_dict["guidance_in"] = guidance_in
#         g = tf.concat([guidance_in, images], axis=3)
#         g = conv2d(g, [3,3,15,embed], activate=tf.nn.relu, scope="prior_embed", is_training=is_training)
#         f = []
#         for stage in range(1, num_stage+1):
#             conv1 = conv2d(g, [3,3,embed,embed], activate=tf.nn.relu, scope="conv%d_1" %(stage), is_training=is_training)
#             g = conv2d(conv1, [3,3,embed,embed], activate=tf.nn.relu, scope="conv%d_2" %(stage), is_training=is_training)
#             if scale_list[num_stage-stage] == 2:
#                 g = tf.nn.max_pool(g, [1,2,2,1], [1,2,2,1], "VALID")

#             feature = rm_test(in_node=layers_dict["low_level"+str(stage)],
#                                             num_filters=embed,
#                                             guidance=g,
#                                             num_class=num_class,
#                                             # scale=scale_list[num_stage-stage],
#                                             further_attention=further_attention,
#                                             scope='RM_'+str(stage),
#                                             is_training=is_training)  
#             new_feature = tf.image.resize_bilinear(feature, [256, 256], name='new_feature')
#             f.append(new_feature)
            
#             layers_dict["g"+str(stage)] = g
#             layers_dict["feature"+str(stage)] = feature
            
#         final_f = tf.concat(f, axis=3)    
#         logits = conv2d(final_f, [1,1,embed*num_stage,num_class], 
#                               activate=None, scope="logits", is_training=is_training)
        
#     return logits, layers_dict


# def refinement_network(features,
#                        guidance,
#                        output_stride,
#                        batch_size,
#                        layers_dict,
#                        num_class=None,
#                        embed=32,
#                        further_attention=False,
#                        rm_type=None,
#                        input_guidance_in_each_stage=False,
#                        is_training=None,
#                        scope=None):
#     """
#     """
#     # TODO: necessary for using batch_size??
#     # TODO: check guidance shape. The shape [?,256,256,1] should cause error
#     # scale_list = [1,1,1,4,2] # output_strides=8
#     scale_list = [1,1,2,2,2] # output_strides=8
#     num_stage = len(layers_dict)
#     # num_up = int(math.sqrt(output_stride))
#     # num_same = num_stage - num_up
#     # upsample_flags = num_same*[False] + num_up*[True]
        
#     with tf.variable_scope(scope, 'refinement_network') as sc:
#         guidance_in = guidance
#         _, h, w = layers_dict["low_level5"].get_shape().as_list()[0:3]
#         guidance_in_lowres = tf.image.resize_bilinear(guidance, [h, w])
#         zero_tensor = tf.zeros([batch_size, h, w, embed])
            
#         layers_dict["guidance_in"] = guidance_in
#         feature, guidance = rm_class(in_node=layers_dict["low_level5"],
#                                     feature=None,
#                                     guidance=guidance_in_lowres,
#                                     scale=scale_list[0],
#                                     scope='RM_5',
#                                     is_training=is_training)
#         layers_dict["guidance1"] = guidance
#         layers_dict["feature1"] = feature
        
#         # guidance = tf.nn.softmax(guidance, axis=3)
#         # TODO: got to be a better way --> maybe p*guidance_in+(1-p)*guid in each stage
#         # if input_guidance_in_each_stage:
#         #     guidance = guidance + guidance_in_lowres
            
                
#         for stage in range(num_stage-1, 0, -1):
#             if rm_type != "feature_guided":
#                 if input_guidance_in_each_stage:
#                     p = tf.get_variable('weight', [1,1,1,num_class], initializer=tf.constant_initializer(0.5))
#                     h, w = guidance.get_shape().as_list()[1:3]
#                     guidance_in_lowres = tf.image.resize_bilinear(guidance_in, [h, w])
#                     guidance = p*guidance_in_lowres + (tf.ones_like(p)-p)*tf.nn.softmax(guidance, axis=3)
#                 else:
#                     guidance = tf.nn.softmax(guidance, axis=3)
            
#             if rm_type == "class_split_in_each_stage":
#                 feature, guidance = rm_class(in_node=layers_dict["low_level"+str(stage)],
#                                             feature=feature,
#                                             guidance=guidance,
#                                             scale=scale_list[num_stage-stage],
#                                             further_attention=further_attention,
#                                             scope='RM_'+str(stage),
#                                             is_training=is_training)
#                 # if stage != 1:
#                 #     guidance = tf.nn.softmax(guidance, axis=3)
#             elif rm_type == "feature_guided":
#                 feature, guidance = rm_feature(in_node=layers_dict["low_level"+str(stage)],
#                                         feature=feature,
#                                         guidance=feature,
#                                         num_class=num_class,
#                                         scale=scale_list[num_stage-stage],
#                                         further_attention=further_attention,
#                                         scope='RM_'+str(stage),
#                                         is_training=is_training)        
#             else:
#                 feature, guidance = rm_class(in_node=layers_dict["low_level"+str(stage)],
#                                             feature=feature,
#                                             guidance=tf.nn.softmax(guidance, axis=3),
#                                             scale=scale_list[num_stage-stage],
#                                             further_attention=further_attention,
#                                             weight_sharing=True,
#                                             scope='RM_'+str(stage),
#                                             is_training=is_training)
            
#             layers_dict["guidance"+str(num_stage+1-stage)] = guidance
#             layers_dict["feature"+str(num_stage+1-stage)] = feature

#             # guidance = tf.nn.softmax(guidance, axis=3)
#             # if input_guidance_in_each_stage:
#             #     h, w = layers_dict["low_level"+str(stage)].get_shape().as_list()[1:3]
#             #     cue = tf.image.resize_bilinear(guidance_in, [h, w])
#             #     guidance = guidance + cue
                 
#     return guidance, layers_dict


def rm_class(in_node,
             guidance,
             feature,
             scale=None,
             further_attention=False,
             weight_sharing=False,
             is_training=None,
             scope=None):
    """refining module
    feature: [H,W,Channels]
    guidance: [H,W,Class]
    """
    with tf.variable_scope(scope, 'rm_class'):
        h, w = in_node.get_shape().as_list()[1:3]
        num_class = guidance.get_shape().as_list()[3]
        num_filters = feature.get_shape().as_list()[3]
        channels = in_node.get_shape().as_list()[3]

        # feature embedding
        

        conv_r1 = conv2d(in_node, [3,3,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
        conv_r1 = conv2d(conv_r1, [3,3,num_filters,num_filters], activate=tf.nn.relu, scope="conv_r1_2", is_training=is_training)
        
        # conv_r1 = GCN(in_node, num_filters, ks=7, is_training=is_training)
        
        # sram attention
        def sram_attention_branch(sram_input, scope):
            feature_list = []
            f = num_filters*num_class
            if feature is not None:
                feature_list.append(feature)
                f += num_filters
                
            for c in range(num_class):
                if not weight_sharing:
                    scope = "_".join([scope, str(c)])
                attention_class = sram(sram_input, guidance[...,c:c+1], scope=scope, is_training=is_training)
                feature_list.append(attention_class)
                
            attention = conv2d(tf.concat(feature_list, axis=3), [1,1,f,num_filters], 
                               activate=tf.nn.relu, scope="fuse", is_training=is_training) 
            return attention
        
        new_feature = sram_attention_branch(sram_input=conv_r1, scope="sram1")
        if further_attention:
            new_feature = sram_attention_branch(sram_input=new_feature, scope="sram2")
        
        new_guidance = conv2d(new_feature, [1,1,num_filters,num_class], 
                              activate=None, scope="guidance", is_training=is_training)
        if scale is not None or scale != 1:
            new_feature = tf.image.resize_bilinear(new_feature, [scale*h, scale*w], name='new_feature')
            new_guidance = tf.image.resize_bilinear(new_guidance, [scale*h, scale*w], name='new_guidance')
        
    return new_feature, new_guidance
    

def rm_feature(in_node,
        feature,
        guidance,
        num_class=14,
        scale=None,
        further_attention=False,
        is_training=None,
        scope=None):
    """refining module
    feature: [H,W,Channels]
    guidance: [H,W,Channels]
    """
    with tf.variable_scope(scope, 'rm'):
        h, w = in_node.get_shape().as_list()[1:3]
        num_filters = feature.get_shape().as_list()[3]
        
        # feature embedding
        conv_r1 = GCN(in_node, num_filters, ks=7, is_training=is_training)
        
        # sram attention
        attention = sram(conv_r1, guidance, scope='sram1', is_training=is_training)
        new_feature = conv2d(tf.concat([attention,conv_r1], axis=3), [1,1,2*num_filters,num_filters], 
                           activate=tf.nn.relu, scope="fuse", is_training=is_training) 
        if further_attention:
            new_feature = sram(new_feature, guidance, scope='sram2', is_training=is_training)
         
        new_guidance = conv2d(new_feature, [1,1,num_filters,num_class], 
                              activate=None, scope="guidance", is_training=is_training)
        if scale is not None or scale != 1:
            new_feature = tf.image.resize_bilinear(new_feature, [scale*h, scale*w], name='new_feature')
            new_guidance = tf.image.resize_bilinear(new_guidance, [scale*h, scale*w], name='new_guidance')
      
    return new_feature, new_guidance


def sram(in_node,
         guidance,
         is_gamma=False,
         scope=None,
         is_training=True):
    """Single Residual Attention Module"""
    with tf.variable_scope(scope, "sram", reuse=tf.AUTO_REUSE):
        channels = in_node.get_shape().as_list()[3]
        conv1 = conv2d(in_node, [3,3,channels,channels], activate=tf.nn.relu, scope="conv1", is_training=is_training)
        conv2 = conv2d(conv1, [3,3,channels,channels], activate=tf.nn.relu, scope="conv2", is_training=is_training)
        
        guidance_filters = guidance.get_shape().as_list()[3]
        if guidance_filters == 1:
            guidance_tile = tf.tile(guidance, [1,1,1,channels])
        elif guidance_filters == channels:
            guidance_tile = guidance
        else:
            raise ValueError("Unknown guidance filters number")

        if is_gamma:
            gamma = tf.Variable(0, dtype=tf.float32)
            output = in_node + gamma*tf.multiply(conv2, guidance_tile)
        else:
            output = in_node + tf.multiply(conv2, guidance_tile)

        tf.add_to_collection("/sram_embed", {"in_node": in_node,
                                             "conv2": conv2, 
                                             "guidance_tile": guidance_tile, 
                                             "output": output})
        # tf.add_to_collection("/sram_embed", [in_node, conv1, conv2, guidance_tile, output])
        return output