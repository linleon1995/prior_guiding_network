import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
from core import resnet_v1_beta, preprocess_utils, cell, attentions
slim = contrib_slim
resnet_v1_beta_block = resnet_v1_beta.resnet_v1_beta_block



def get_func(func):
  @functools.wraps(func)
  def network_fn(*args, **kwargs):
      return func(*args, **kwargs)
  return network_fn


class Refine(object):
  def __init__(self, low_level, fusions, prior_seg=None, prior_pred=None, stage_pred_loss_name=None, guid_conv_nums=2,
               guid_conv_type="conv2d", embed_node=32, predict_without_background=False,
               num_class=14, weight_decay=0.0, scope=None, is_training=None, **kwargs):
    self.low_level = list(low_level.values())
    self.fusions = fusions
    self.prior_seg = prior_seg
    self.prior_pred = prior_pred
    self.stage_pred_loss_name = stage_pred_loss_name
    self.embed_node = embed_node
    self.guid_conv_type = guid_conv_type
    self.guid_conv_nums = guid_conv_nums
    self.predict_without_background = predict_without_background
    self.num_class = num_class
    self.weight_decay = weight_decay
    self.scope = scope
    # assert len(self.low_level) == len(self.fusions)
    self.num_stage = len(self.low_level)
    self.is_training = is_training
    self.fine_tune_batch_norm = True

    self.apply_sram2 = kwargs.pop("apply_sram2", False)
    self.guid_fuse = kwargs.pop("guid_fuse", "sum")
    
    self.g = get_func(guidance_fusion_method)
    self.e = slim.conv2d
    self.attention = utils.
    
  
  def get(self):
    batch_norm = slim.batch_norm
    batch_norm_params = get_batch_norm_params(decay=0.9997,
                                              epsilon=1e-5,
                                              scale=True,
                                              is_training=(self.is_training and self.fine_tune_batch_norm),
                                              # sync_batch_norm_method=model_options.sync_batch_norm_method
                                              )

    with tf.variable_scope(self.scope, 'Refine_Network'):
      with slim.arg_scope([slim.conv2d],
                          trainable=True,
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.initializers.he_normal(),
                          weights_regularizer=slim.l2_regularizer(self.weight_decay),
                          kernel_size=[3, 3],
                          padding='SAME',
                          normalizer_fn=slim.batch_norm):
        with slim.arg_scope([batch_norm], **batch_norm_params):
          y = self.model()
    return y
  
  
  def model(self):
    def func(x, fuse, guid=None, apply_second_att=True):
      embed = self.e(x)
      if guid is None:
        guid = self.g(x)
      net = attention(embed, guid)
      net = net + fuse
      if apply_second_att:
        net = attention(net, guid)
      return net
    
    f = func(self.low_level[0], self.e(self.low_level[0]))
    guid = self.g(f)
    fuse = self.e(f)
    for i in range(1, len(self.low_level)):
      f = func(self.low_level[i], fuse, guid)
      if i < len(self.low_level)-1:
        guid = self.g(f)
      fuse = self.e(f)  
    
    y = fuse
    y = resize_bilinear(y, [2*h, 2*w])
    y = slim.conv2d(y, self.embed_node, scope="decoder_output")
    y = slim.conv2d(y, self.num_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='logits_pred_class%d' %self.num_class)
    return y


def slim_sram(in_node,
              guidance,
              num_conv=1,
              conv_type="conv",
              conv_node=64,
              scope=None):
  """Single Residual Attention Module"""
  with tf.variable_scope(scope, "sram", reuse=tf.AUTO_REUSE):
    # channel = in_node.get_shape().as_list()[3]
    net = in_node
    if conv_type == "conv":
      conv_op = slim.conv2d
    elif conv_type == "separable_conv":
      conv_op = slim.separable_conv2d
    else:
      raise ValueError("Unknown convolution type")

    for i in range(num_conv-1):
      net = conv_op(net, conv_node, kernel_size=[3,3], scope=conv_type+str(i+1))
    net = conv_op(net, conv_node, kernel_size=[3,3], scope=conv_type+"out", activation_fn=None)
    # for i in range(num_conv):
    #   net = conv_op(net, conv_node, kernel_size=[3,3], scope=conv_type+str(i+1))

    # guidance_filters = guidance.get_shape().as_list()[3]
    guidance_filters = preprocess_utils.resolve_shape(guidance, rank=4)[3]
    if guidance_filters == 1:
        guidance_tile = tf.tile(guidance, [1,1,1,conv_node])
    elif guidance_filters == conv_node:
        guidance_tile = guidance
    else:
        raise ValueError("Unknown guidance filters number")

    # tf.add_to_collection("/sram_embed", {"in_node": in_node,
    #                                      "conv2": conv2,
    #                                      "guidance_tile": guidance_tile,
    #                                      "output": output})

    output = in_node + tf.multiply(net, guidance_tile)
    tf.add_to_collection(scope+"_guided_feature", tf.multiply(net, guidance_tile))
    return output


def guidance_fusion_method(guid, guid_fuse, num_class=None, out_node=None, level=None):
  if guid_fuse == "sum":
    return tf.reduce_sum(guid, axis=3, keepdims=True)
  elif guid_fuse == "mean":
    return tf.reduce_mean(guid, axis=3, keepdims=True)
  elif guid_fuse == "entropy":
    def func():
      guid = tf.clip_by_value(guid, 1e-10, 1.0)
      guid = -tf.reduce_sum(guid * tf.log(guid), axis=3, keepdims=True)
      return func
  elif guid_fuse == "conv":
    guid = slim.conv2d(guid, out_node, kernel_size=[3,3], activation_fn=None)
  elif guid_fuse == "sum_dilated":
    size = [8,6,4,2,1]
    kernel = tf.ones((size[level], size[level], num_class))
    guid = tf.nn.dilation2d(guid, filter=kernel, strides=(1,1,1,1), rates=(1,1,1,1), padding="SAME")
    guid = guid - tf.ones_like(guid)
    guid = tf.reduce_sum(guid, axis=3, keepdims=True)
  elif guid_fuse == "w_sum":
    w = tf.nn.softmax(tf.reduce_sum(guid, axis=[1,2], keepdims=True), axis=3)
    rev_w = tf.ones_like(w) - w
    guid = tf.reduce_sum(tf.multiply(guid, rev_w), axis=3, keepdims=True)
  elif guid_fuse == "conv_sum":
    k_size_list = [1,1,1,3,5]
    k_size = 2 * k_size_list[level] + 1
    guid = slim.conv2d(guid, 1, kernel_size=[k_size,k_size], activation_fn=None,
                      weights_initializer=tf.ones_initializer(), trainable=False, normalizer_fn=None)
    guid = guid / (k_size*k_size*num_class*1)
  elif guid_fuse == "w_sum_conv":
    # TODO: make it right
    k_size_list = [1,1,1,2,4]
    k_size = 3 * k_size_list[level] + 1
    w = tf.reduce_sum(guid, axis=[1,2], keepdims=True)
    rev_w = (tf.ones_like(w)+1e-5) / (tf.sqrt(w)+1e-5)
    rev_w = tf.tile(rev_w, [1,k_size,k_size,1])
    rev_w = tf.expand_dims(rev_w, axis=4)

    n, h, w, channels_img = preprocess_utils.resolve_shape(guid, rank=4)
    n, fh, fw, channels, out_channels = preprocess_utils.resolve_shape(rev_w, rank=5)
    # F has shape (n, k_size, k_size, channels, out_channels)

    rev_w = tf.transpose(rev_w, [1, 2, 0, 3, 4])
    rev_w = tf.reshape(rev_w, [fh, fw, channels*n, out_channels])

    guid = tf.transpose(guid, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
    guid = tf.reshape(guid, [1, h, w, n*channels_img])

    out = tf.nn.depthwise_conv2d(
              guid,
              filter=rev_w,
              strides=[1, 1, 1, 1],
              padding="SAME") # here no requirement about padding being 'VALID', use whatever you want.
    # Now out shape is (1, H-fh+1, W-fw+1, MB*channels*out_channels), because we used "VALID"

    out = tf.reshape(out, [h, w, n, channels, out_channels])
    out = tf.transpose(out, [2, 0, 1, 3, 4])
    out = tf.reduce_sum(out, axis=3)
    guid = out
  elif guid_fuse == "sum_wo_back"
    flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
    guid = tf.multiply(guid, flag)
    guid = tf.reduce_sum(guid, axis=3, keepdims=True)
  elif guid_fuse == "mean_wo_back":
    flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
    guid = tf.multiply(guid, flag)
    guid = tf.reduce_mean(guid, axis=3, keepdims=True)
  elif guid_fuse == "same":
    pass
  else:
    raise ValueError("Unknown guid fuse")

  tf.add_to_collection("guidance", guid)
  return 