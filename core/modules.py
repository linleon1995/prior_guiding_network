import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
from core import resnet_v1_beta, preprocess_utils, cell, attentions
slim = contrib_slim
resnet_v1_beta_block = resnet_v1_beta.resnet_v1_beta_block

class Refine(object):
  def __init__(self, low_level, fusions, prior_seg=None, prior_pred=None, stage_pred_loss_name=None, guid_conv_nums=2,
               guid_conv_type="conv2d", embed_node=32, predict_without_background=False,
               num_class=14, weight_decay=0.0, scope=None, is_training=None, **kwargs):
    self.low_level = low_level.values()
    self.fusions = fusions
    self.prior_seg = prior_seg
    self.prior_pred = prior_pred
    self.stage_pred_loss_name = stage_pred_loss_name
    if "softmax" in self.stage_pred_loss_name:
      self.stage_pred_activation = tf.nn.softmax
    elif "sigmoid" in self.stage_pred_loss_name:
      self.stage_pred_activation = tf.nn.sigmoid
    self.embed_node = embed_node
    self.guid_conv_type = guid_conv_type
    self.guid_conv_nums = guid_conv_nums
    # self.predict_without_background = predict_without_background
    self.num_class = num_class
    self.weight_decay = weight_decay
    
    self.scope = scope
    # assert len(self.low_level) == len(self.fusions)
    self.num_stage = len(self.low_level)
    self.is_training = is_training
    self.fine_tune_batch_norm = True

    self.apply_sram2 = kwargs.pop("apply_sram2", False)
    self.guid_fuse = kwargs.pop("guid_fuse", "sum")
    
    # TODO: interface in model
    self.model_variants = kwargs.pop("model_variants", None)
    self.height = kwargs.pop("height", None)
    self.width = kwargs.pop("width", None)
    self.g = utils2.guidance_fusion_method
    # self.attention = utils.
    
  def get_fusion_method(self, method):
    if method == "concat":
      return utils2.concat_convolution
    elif method == "sum":
      return utils2.sum_convolution
    elif method in ("guid", "guid_uni"):
      return utils2.guid_attention
    elif method == "guid_class":
      return utils2.guid_class_attention
    elif method == "context_att":
      return utils2.context_attention
    elif method == "self_att":
      return utils2.self_attention
    
  def simple_decoder(self):
    for i in range(len(self.low_level)):
      net = slim.conv2d(self.low_level[i], self.embed_node, kernel_size=[1, 1], scope="embed%d" %(self.num_stage-i))
      if i > 0:
        fuse_func = self.get_fusion_method(self.fusions[i])
        net = fuse_func(net, fusion, self.embed_node, scope=self.fusions[i]+str(self.num_stage-i))
      fusion = slim.conv2d(net, self.embed_node, kernel_size=[3, 3], scope="transform%d" %(self.num_stage-i))
    return fusion
  
  def refine_decoder(self):
    def guid_gen(net, scope):
      # TODO: guid_uni guid guid_class
      net = slim.conv2d(net, self.n_class, kernel_size=[1, 1], scope=scope)
      tf.add_to_collection("stage_pred", net)
      guid = self.g(net, self.guid_fuse, self.num_class, self.embed_node, self.num_stage-i)
      return guid
    
    guid = self.g(self.prior_pred, self.guid_fuse, self.num_class, self.embed_node, self.num_stage-i)
    fusion = self.prior_seg
    # TODO: add_to_collections depend on evaluation
    # tf.add_to_collection("guidance", guid)
    # tf.add_to_collection("feature", fusion)
    for i in range(1, len(self.low_level)):
      net = slim.conv2d(self.low_level[i], self.embed_node, kernel_size=[1, 1], scope="embed%d" %(self.num_stage-i))
      activation = self.stage_pred_activation
      # TODO: input activation to fuse_func
      # TODO: fuse_func variables
      fuse_func = self.get_fusion_method(self.fusions[i])
      net = fuse_func(net, fusion, guid, self.embed_node, scope=self.fusions[i]+str(self.num_stage-i))
      if i < len(self.low_level)-1:
        guid = guid_gen(net, scope="stage_pred%d" %(self.num_stage-i))
        fusion = slim.conv2d(net, self.embed_node, kernel_size=[3, 3], scope="embed%d" %(self.num_stage-i))
    return net
    
  def model(self):
    batch_norm = slim.batch_norm
    batch_norm_params = utils2.get_batch_norm_params(decay=0.9997,
                                              epsilon=1e-5,
                                              scale=True,
                                              is_training=(self.is_training and self.fine_tune_batch_norm),
                                              # sync_batch_norm_method=model_options.sync_batch_norm_method
                                              )

    with tf.variable_scope(self.scope, 'Decoder'):
      with slim.arg_scope([slim.conv2d],
                          trainable=True,
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.initializers.he_normal(),
                          weights_regularizer=slim.l2_regularizer(self.weight_decay),
                          kernel_size=[3, 3],
                          padding='SAME',
                          normalizer_fn=slim.batch_norm):
        with slim.arg_scope([batch_norm], **batch_norm_params):
            if self.model_variants == "unet" or self.model_variants == "fpn":
              feature = self.simple_decoder()
            elif self.model_variants == "refine":
              feature = self.refine_decoder()
            else:
              raise ValueError("Unknown decoder type")
          y = utils2.resize_bilinear(feature, [self.height, self.width])
          y = slim.conv2d(y, self.embed_node, scope="decoder_output")
          y = slim.conv2d(
            y, self.num_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='logits_pred_class%d' %self.num_class)
    return y
  
  
def seq_model(inputs, ny, nx, n_class, weight_decay, is_training, cell_type='ConvGRU'):
  with slim.arg_scope([slim.batch_norm],
                        is_training=is_training):
    with slim.arg_scope([slim.conv2d],
                      weights_initializer=tf.initializers.he_normal(),
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      normalizer_fn=slim.batch_norm):
    # in_shape = inputs.get_shape().as_list()
      in_shape = preprocess_utils.resolve_shape(inputs, rank=5)
      batch_size = in_shape[0]
      # seq_length = in_shape[1]
      nx = in_shape[2]
      ny = in_shape[3]

      if cell_type =='ConvGRU':
        with tf.variable_scope("forward_cell") as scope:
            cell_forward = cell.ConvGRUCell(shape=[ny, nx], filters=n_class, kernel=[3, 3])
            outputs_forward, state_forward = tf.nn.dynamic_rnn(
              cell=cell_forward, dtype=tf.float32, inputs=inputs,
              initial_state=cell_forward.zero_state(batch_size, dtype=tf.float32))
        feats = state_forward
      elif cell_type =='BiConvGRU':
        with tf.variable_scope("forward_cell") as scope:
            cell_forward = cell.ConvGRUCell(shape=[ny, nx], filters=n_class, kernel=[3, 3])
            outputs_forward, state_forward = tf.nn.dynamic_rnn(
              cell=cell_forward, dtype=tf.float32, inputs=inputs,
              initial_state=cell_forward.zero_state(batch_size, dtype=tf.float32))

        inputs_b = inputs[:,::-1]
        with tf.variable_scope("backward_cell") as scope:
            cell_backward = cell.ConvGRUCell(shape=[ny, nx], filters=n_class, kernel=[3, 3])
            outputs_backward, state_backward = tf.nn.dynamic_rnn(
              cell=cell_backward, dtype=tf.float32, inputs=inputs_b,
              initial_state=cell_backward.zero_state(batch_size, dtype=tf.float32))

        feats = tf.concat([state_forward, state_backward], axis=3)

      y = slim.conv2d(feats, n_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='fuse')
    # print(60*"X", inputs, y, state_forward)
    return y
