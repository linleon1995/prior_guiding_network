'''
Created on Aug 19, 2016

author: jakeret
'''
from __future__ import print_function, division, absolute_import, unicode_literals

import tensorflow as tf
import numpy as np
from tf_tesis2.utils import (fc_layer, conv2d, atrous_conv2d, split_separable_conv2d, batch_norm, upsampling_layer)
from tf_tesis2.module import nonlocal_dot
from tf_tesis2.core import RM
    


def crn_vanilla(features,
                guidance,
                batch_size,
                layer_dict,
                embed=32,
                is_training=True):
    """crn_vanilla
    """
    with tf.variable_scope("crn_vanilla"):   
        h, w = layer_dict["pool4"].get_shape().as_list()[1:3]
        guidance_in = tf.image.resize_bilinear(guidance, [h, w], name='guidance_in')
        zero_tensor = tf.zeros([batch_size, h, w, embed])

        feature1, guidance1, f1, vis1 = RM(in_node=layer_dict["pool4"], 
                                                        feature=zero_tensor, 
                                                        guidance=guidance_in,  
                                                        out_filter=embed, 
                                                        name='RM_5', 
                                                        is_training=is_training)
        guidance1_a = tf.nn.softmax(guidance1)
        
        feature2, guidance2, f2, vis2 = RM(layer_dict["pool3"], 
                                                        feature1, 
                                                        guidance1_a, 
                                                        None, 
                                                        out_filter=embed, 
                                                        name='RM_4', 
                                                        is_training=is_training)
        guidance2_a = tf.nn.softmax(guidance2)
        
        feature3, guidance3, f3, vis3 = RM(layer_dict["pool2"], 
                                                        feature2, 
                                                        guidance2_a, 
                                                        None, 
                                                        out_filter=embed, 
                                                        name='RM_3', 
                                                        is_training=is_training)
        guidance3_a = tf.nn.softmax(guidance3)
        
        feature4, guidance4, f4, vis4 = RM(layer_dict["pool1"], 
                                                        feature3, 
                                                        guidance3_a, 
                                                        None, 
                                                        out_filter=embed, 
                                                        classifier=True, 
                                                        name='RM_1', 
                                                        is_training=is_training)
    layer_dict.update({
                  "guidance_in": guidance_in,
                  "feature1": feature1, "guidance1": guidance1,
                  "feature2": feature2, "guidance2": guidance2,
                  "feature3": feature3, "guidance3": guidance3,
                  "feature4": feature4, "guidance4": guidance4,
#                  "feature5": feature5, "guidance5": guidance5,
                  "output": guidance4,
                  })
    info = [f1,f2,f3,f4, vis1, vis2, vis3, vis4]
    output['output_map'] = guidance4
    return output, layer_dict, info

    
def crn_onestage():
    pass
    
 