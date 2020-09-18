import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
from core import resnet_v1_beta, preprocess_utils, cell, attentions
slim = contrib_slim
resnet_v1_beta_block = resnet_v1_beta.resnet_v1_beta_block


# def guidance_fusion_method(guid, guid_fuse):
#   if self.guid_fuse == "sum":
#     guid = tf.reduce_sum(guid, axis=3, keepdims=True)
#   elif self.guid_fuse == "mean":
#     guid = tf.reduce_mean(guid, axis=3, keepdims=True)
#   elif self.guid_fuse == "entropy":
#     guid = tf.clip_by_value(guid, 1e-10, 1.0)
#     guid = -tf.reduce_sum(guid * tf.log(guid), axis=3, keepdims=True)
#   elif self.guid_fuse == "conv":
#     if i < len(self.low_level)-1:
#       guid = slim.conv2d(guid, out_node, kernel_size=[3,3], activation_fn=None)
#     else:
#       guid = tf.reduce_sum(guid, axis=3, keepdims=True)
#   elif self.guid_fuse == "sum_dilated":
#     size = [8,6,4,2,1]
#     kernel = tf.ones((size[i], size[i], num_class))
#     guid = tf.nn.dilation2d(guid, filter=kernel, strides=(1,1,1,1),
#                               rates=(1,1,1,1), padding="SAME")
#     guid = guid - tf.ones_like(guid)
#     guid = tf.reduce_sum(guid, axis=3, keepdims=True)
#   elif self.guid_fuse == "w_sum":
#     w = tf.nn.softmax(tf.reduce_sum(guid, axis=[1,2], keepdims=True), axis=3)
#     rev_w = tf.ones_like(w) - w
#     guid = tf.reduce_sum(tf.multiply(guid, rev_w), axis=3, keepdims=True)
#   elif self.guid_fuse == "conv_sum":
#     k_size_list = [1,1,1,3,5]
#     k_size = 2 * k_size_list[i] + 1
#     guid = slim.conv2d(guid, 1, kernel_size=[k_size,k_size], activation_fn=None,
#                       weights_initializer=tf.ones_initializer(), trainable=False, normalizer_fn=None)
#     guid = guid / (k_size*k_size*num_class*1)
#   elif self.guid_fuse == "w_sum_conv":
#     # TODO: make it right
#     k_size_list = [1,1,1,2,4]
#     k_size = 3 * k_size_list[i] + 1
#     w = tf.reduce_sum(guid, axis=[1,2], keepdims=True)
#     rev_w = (tf.ones_like(w)+1e-5) / (tf.sqrt(w)+1e-5)
#     rev_w = tf.tile(rev_w, [1,k_size,k_size,1])
#     rev_w = tf.expand_dims(rev_w, axis=4)

#     n, h, w, channels_img = preprocess_utils.resolve_shape(guid, rank=4)
#     n, fh, fw, channels, out_channels = preprocess_utils.resolve_shape(rev_w, rank=5)
#     # F has shape (n, k_size, k_size, channels, out_channels)

#     rev_w = tf.transpose(rev_w, [1, 2, 0, 3, 4])
#     rev_w = tf.reshape(rev_w, [fh, fw, channels*n, out_channels])

#     guid = tf.transpose(guid, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
#     guid = tf.reshape(guid, [1, h, w, n*channels_img])

#     out = tf.nn.depthwise_conv2d(
#               guid,
#               filter=rev_w,
#               strides=[1, 1, 1, 1],
#               padding="SAME") # here no requirement about padding being 'VALID', use whatever you want.
#     # Now out shape is (1, H-fh+1, W-fw+1, MB*channels*out_channels), because we used "VALID"

#     out = tf.reshape(out, [h, w, n, channels, out_channels])
#     out = tf.transpose(out, [2, 0, 1, 3, 4])
#     out = tf.reduce_sum(out, axis=3)

#     guid = out
#   elif self.guid_fuse == "sum_wo_back":
#     flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
#     guid = tf.multiply(guid, flag)
#     guid = tf.reduce_sum(guid, axis=3, keepdims=True)
#   elif self.guid_fuse == "mean_wo_back":
#     flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
#     guid = tf.multiply(guid, flag)
#     guid = tf.reduce_mean(guid, axis=3, keepdims=True)
#   elif self.guid_fuse == "same":
#     pass
#   else:
#     raise ValueError("Unknown guid fuse")

#   tf.add_to_collection("guidance", guid)
#   return 

# class Refine(object):
#   def __init__(self, low_level, fusions, prior_seg=None, prior_pred=None, stage_pred_loss_name=None, guid_conv_nums=2,
#                guid_conv_type="conv2d", embed_node=32, predict_without_background=False,
#                num_class=14, weight_decay=0.0, scope=None, is_training=None, **kwargs):
#     self.low_level = list(low_level.values())
#     self.fusions = fusions
#     self.prior_seg = prior_seg
#     self.prior_pred = prior_pred
#     self.stage_pred_loss_name = stage_pred_loss_name
#     self.embed_node = embed_node
#     self.guid_conv_type = guid_conv_type
#     self.guid_conv_nums = guid_conv_nums
#     self.predict_without_background = predict_without_background
#     self.num_class = num_class
#     self.weight_decay = weight_decay
#     self.scope = scope
#     # assert len(self.low_level) == len(self.fusions)
#     self.num_stage = len(self.low_level)
#     self.is_training = is_training
#     self.fine_tune_batch_norm = True

#     self.apply_sram2 = kwargs.pop("apply_sram2", False)
#     self.guid_fuse = kwargs.pop("guid_fuse", "sum")
    
#     self.g = guidance_fusion_method(x, self.num_class, self.out_node, level)
#     self.e = slim.conv2d
#     self.attention = utils.
    
  
#   def get(self):
#     batch_norm = slim.batch_norm
#     batch_norm_params = get_batch_norm_params(decay=0.9997,
#                                               epsilon=1e-5,
#                                               scale=True,
#                                               is_training=(self.is_training and self.fine_tune_batch_norm),
#                                               # sync_batch_norm_method=model_options.sync_batch_norm_method
#                                               )

#     with tf.variable_scope(self.scope, 'Refine_Network'):
#       with slim.arg_scope([slim.conv2d],
#                           trainable=True,
#                           activation_fn=tf.nn.relu,
#                           weights_initializer=tf.initializers.he_normal(),
#                           weights_regularizer=slim.l2_regularizer(self.weight_decay),
#                           kernel_size=[3, 3],
#                           padding='SAME',
#                           normalizer_fn=slim.batch_norm):
#         with slim.arg_scope([batch_norm], **batch_norm_params):
#           y = self.model()
#     return y
  
#   def model(self):
#     def func(x, fuse, guid=None, apply_second_att=True):
#       embed = self.e(x)
#       if guid is None:
#         guid = self.g(x)
#       net = attention(embed, guid)
#       net = net + fuse
#       if apply_second_att:
#         net = attention(net, guid)
#       return net
    
#     f = func(self.low_level[0], self.e(self.low_level[0]))
#     guid = self.g(f)
#     fuse = self.e(f)
#     for i in range(1, len(self.low_level)):
#       f = func(self.low_level[i], fuse, guid)
#       if i < len(self.low_level)-1:
#         guid = self.g(f)
#       fuse = self.e(f)  
    
#     y = fuse
#     y = resize_bilinear(y, [2*h, 2*w])
#     y = slim.conv2d(y, self.embed_node, scope="decoder_output")
#     y = slim.conv2d(y, self.num_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='logits_pred_class%d' %self.num_class)
#     return y

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
  def embed(self, x, out_node, scope):
    return slim.conv2d(x, out_node, kernel_size=[1,1], scope=scope)

  def model(self):
    # TODO: reolve_shape
    # TODO: image size
    # TODO: Remove after finish code
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
          y_tm1 = self.prior_seg
          preds = {}

          # TODO: Would default vars value causes error?
          if self.prior_seg is not None or self.prior_pred is not None:
            if "guid" in self.fusions:
              guid = self.prior_seg
            elif "guid_class" in self.fusions:
              guid = self.prior_pred
            elif "guid_uni" in self.fusions or "context_att" in self.fusions or "self_att" in self.fusions:
              guid = tf.reduce_mean(self.prior_pred, axis=3, keepdims=True)
          out_node = self.embed_node
          tf.add_to_collection("guidance", guid)
          
          for i, v in enumerate(self.low_level):
            module_order = self.num_stage-i
            fuse_method = self.fusions[i]
            embed = self.embed(v, out_node, scope="embed%d" %module_order)
            tf.add_to_collection("embed", embed)

            fuse_func = self.get_fusion_method(fuse_method)
            h, w = preprocess_utils.resolve_shape(embed, rank=4)[1:3]

            if y_tm1 is not None:
              y_tm1 = resize_bilinear(y_tm1, [h, w])
              tf.add_to_collection("feature", y_tm1)
            else:
              # TODO: remove
              tf.add_to_collection("feature", tf.zeros_like(embed))

            if fuse_method in ("concat", "sum"):
              if y_tm1 is not None:
                y = fuse_func(embed, y_tm1, out_node, fuse_method+str(module_order))
              else:
                y = tf.identity(embed, name="identity%d" %module_order)
            elif fuse_method in ("guid", "guid_class", "guid_uni", "context_att", "self_att"):
              # guid = resize_bilinear(guid, [h, w])
              if guid is not None:
                guid = resize_bilinear(guid, [h, w])
              # tf.add_to_collection("guid", guid)

              fuse = fuse_func(embed, y_tm1, guid, out_node, fuse_method+str(module_order),
                            num_classes=self.num_class, apply_sram2=self.apply_sram2)

              y = slim.conv2d(fuse, self.embed_node, scope='fuse'+str(i))
              tf.add_to_collection("refining", y)

            if self.stage_pred_loss_name is not None:

              num_class = self.num_class
              if self.predict_without_background:
                num_class -= 1

              stage_pred =  slim.conv2d(fuse, num_class, kernel_size=[1,1], activation_fn=None,
                                          scope="stage_pred%d_pred_class%d" %(module_order,num_class))


              preds["guidance%d" %module_order] = stage_pred

            if fuse_method in ("guid"):
              guid = y
              y_tm1 = None

            elif fuse_method in ("guid_class", "guid_uni", "context_att", "self_att"):
              if i < len(self.low_level)-1:
                if "softmax" in self.stage_pred_loss_name:
                  # guid = tf.nn.softmax(stage_pred, axis=3)
                  guid = tf.nn.softmax(y, axis=3)
                elif "sigmoid" in self.stage_pred_loss_name:
                  guid = tf.nn.sigmoid(stage_pred)

                if self.guid_fuse == "sum":
                  guid = tf.reduce_sum(guid, axis=3, keepdims=True)
                elif self.guid_fuse == "mean":
                  guid = tf.reduce_mean(guid, axis=3, keepdims=True)
                elif self.guid_fuse == "entropy":
                  guid = tf.clip_by_value(guid, 1e-10, 1.0)
                  guid = -tf.reduce_sum(guid * tf.log(guid), axis=3, keepdims=True)
                elif self.guid_fuse == "conv":
                  if i < len(self.low_level)-1:
                    guid = slim.conv2d(guid, out_node, kernel_size=[3,3], activation_fn=None)
                  else:
                    guid = tf.reduce_sum(guid, axis=3, keepdims=True)
                elif self.guid_fuse == "sum_dilated":
                  size = [8,6,4,2,1]
                  kernel = tf.ones((size[i], size[i], num_class))
                  guid = tf.nn.dilation2d(guid, filter=kernel, strides=(1,1,1,1),
                                            rates=(1,1,1,1), padding="SAME")
                  guid = guid - tf.ones_like(guid)
                  guid = tf.reduce_sum(guid, axis=3, keepdims=True)
                elif self.guid_fuse == "w_sum":
                  w = tf.nn.softmax(tf.reduce_sum(guid, axis=[1,2], keepdims=True), axis=3)
                  rev_w = tf.ones_like(w) - w
                  guid = tf.reduce_sum(tf.multiply(guid, rev_w), axis=3, keepdims=True)
                elif self.guid_fuse == "conv_sum":
                  k_size_list = [1,1,1,3,5]
                  k_size = 2 * k_size_list[i] + 1
                  guid = slim.conv2d(guid, 1, kernel_size=[k_size,k_size], activation_fn=None,
                                    weights_initializer=tf.ones_initializer(), trainable=False, normalizer_fn=None)
                  guid = guid / (k_size*k_size*num_class*1)
                elif self.guid_fuse == "w_sum_conv":
                  # TODO: make it right
                  k_size_list = [1,1,1,2,4]
                  k_size = 3 * k_size_list[i] + 1
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
                elif self.guid_fuse == "sum_wo_back":
                  flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
                  guid = tf.multiply(guid, flag)
                  guid = tf.reduce_sum(guid, axis=3, keepdims=True)
                elif self.guid_fuse == "mean_wo_back":
                  flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
                  guid = tf.multiply(guid, flag)
                  guid = tf.reduce_mean(guid, axis=3, keepdims=True)
                elif self.guid_fuse == "same":
                  pass
                else:
                  raise ValueError("Unknown guid fuse")

                tf.add_to_collection("guidance", guid)

              y_tm1 = y
            elif fuse_method in ("concat", "sum"):
              y_tm1 = y

          # h, w = y.get_shape().as_list()[1:3]
          y = resize_bilinear(y, [2*h, 2*w])
          y = slim.conv2d(y, self.embed_node, scope="decoder_output")
          y = slim.conv2d(y, self.num_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='logits_pred_class%d' %self.num_class)

    return y, preds

  def get_fusion_method(self, method):
    if method == "concat":
      return self.concat_convolution
    elif method == "sum":
      return self.sum_convolution
    elif method in ("guid", "guid_uni"):
      return self.guid_attention
    elif method == "guid_class":
      return self.guid_class_attention
    elif method == "context_att":
      return self.context_attention
    elif method == "self_att":
      return self.self_attention

  def concat_convolution(self, x1, x2, out_node, scope):
    with tf.variable_scope(scope, "concat_conv"):
      net = slim.conv2d(tf.concat([x1, x2], axis=3), out_node, scope="conv1")
  #    net = slim.conv2d(net, out_node, scope=scope+"_2")
      return net

  def sum_convolution(self, x1, x2, out_node, scope):
    with tf.variable_scope(scope, "sum_cocnv"):
      net = slim.conv2d(x1 + x2, out_node, scope="conv1")
  #    net = slim.conv2d(net, out_node, scope=scope+"_2")
      return net

  def guid_attention(self, x1, x2, guid, out_node, scope, *args, **kwargs):
    apply_sram2 = kwargs.pop("apply_sram2", False)
    guid_node = preprocess_utils.resolve_shape(guid, rank=4)[3]

    if guid_node != out_node and guid_node != 1:
      raise ValueError("Unknown guidance node number %d, should be 1 or out_node" %guid_node)

    with tf.variable_scope(scope, 'guid'):
      net = slim_sram(x1, guid, self.guid_conv_nums, self.guid_conv_type, self.embed_node, "sram1")
      tf.add_to_collection("sram1", net)
      if x2 is not None:
        net = net + x2
        if apply_sram2:
          net = slim_sram(net, guid, self.guid_conv_nums, self.guid_conv_type, self.embed_node, "sram2")

      tf.add_to_collection("sram2", net)
      return net

  def guid_class_attention(self, x1, x2, guid, out_node, scope, *args, **kwargs):
    num_classes = kwargs.pop("num_classes", None)
    guid_node = guid.get_shape().as_list()[3]
    if guid_node != num_classes:
      raise ValueError("Unknown guidance node number %d, should equal class number" %guid_node)
    with tf.variable_scope(scope, "guid_class"):
      total_att = []
      for i in range(1, num_classes):
        net = slim_sram(x1, guid[...,i:i+1], self.guid_conv_nums, self.guid_conv_type, self.embed_node, "sram1")
        if x2 is not None:
          net = net + x2
          net = slim_sram(net, guid[...,i:i+1], self.guid_conv_nums, self.guid_conv_type, self.embed_node, "sram2")
        total_att.append(net)
      fuse = slim.conv2d(tf.concat(total_att, axis=3), out_node, kernel_size=[1,1], scope="fuse")
    return fuse

  def context_attention(self, x1, x2, guid, out_node, scope, *args, **kwargs):
    guid_node = preprocess_utils.resolve_shape(guid, rank=4)[3]

    if guid_node != out_node and guid_node != 1:
      raise ValueError("Unknown guidance node number %d, should be 1 or out_node" %guid_node)

    with tf.variable_scope(scope, 'guid'):
      context = slim_sram(x1, guid, self.guid_conv_nums, self.guid_conv_type, self.embed_node, "sram1")
      tf.add_to_collection("sram1", context)
      if x2 is not None:
        ca_layer = attentions.self_attention()
        net = ca_layer.attention(x2, context, x2, out_node, "context_att1")
        tf.add_to_collection("context_att1", net)
      return net

  def self_attention(self, x1, x2, guid, out_node, scope, *args, **kwargs):
    guid_node = preprocess_utils.resolve_shape(guid, rank=4)[3]

    if guid_node != out_node and guid_node != 1:
      raise ValueError("Unknown guidance node number %d, should be 1 or out_node" %guid_node)

    with tf.variable_scope(scope, 'guid'):
      net = slim_sram(x1, guid, self.guid_conv_nums, self.guid_conv_type, self.embed_node, "sram1")
      tf.add_to_collection("sram1", net)
      if x2 is not None:
        net = net + x2
        sa_layer = attentions.self_attention()
        net = sa_layer.attention(net, net, net, out_node, "self_att1")
        tf.add_to_collection("self_att1", net)
      return net


def mlp(inputs,
        output_dims,
        num_layers=3,
        decreasing_root=8,
        scope=None,):
  """
  Args:
    inputs:
    output_dims
    num_layers:
    decreasing_root:
  Returns:
  Raises:

  """
  # TODO: raise if input dims smaller than output_dims
  with tf.variable_scope(scope, 'multi_perceptron_network'):
    net = inputs
    for i in range(num_layers):
      dims = net.get_shape().as_list()[1]
      if i < num_layers-1:
        net = fc_layer(net, [dims, dims//decreasing_root], _std=1, reuse=tf.AUTO_REUSE, scope='_'.join(['fc', str(i)]))
      else:
        outputs = fc_layer(net, [dims, output_dims], _std=1, reuse=tf.AUTO_REUSE, scope='_'.join(['fc', str(i)]))
  return outputs


def GCN(x, out_channels, ks=7, scope=None, is_training=None):
    with tf.variable_scope(scope, "global_convolution_network"):
        channels = x.get_shape().as_list()[3]
        x_l1 = conv2d(x, [ks,1,channels,out_channels], scope="x_l1", is_training=False)
        x_l2 = conv2d(x_l1, [1,ks,out_channels,out_channels], scope="x_l2", is_training=False)

        x_r1 = conv2d(x, [1,ks,channels,out_channels], scope="x_r1", is_training=False)
        x_r2 = conv2d(x_r1, [ks,1,out_channels,out_channels], scope="x_r2", is_training=False)

        x = x_l2 + x_r2
        x = batch_norm(x, is_training=is_training, scope='batch_norm')
        x = tf.nn.relu(x)
    return x

# TODO: Organize
def get_batch_norm_fn(sync_batch_norm_method):
  """Gets batch norm function.
  Currently we only support the following methods:
    - `None` (no sync batch norm). We use slim.batch_norm in this case.
  Args:
    sync_batch_norm_method: String, method used to sync batch norm.
  Returns:
    Batchnorm function.
  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  if sync_batch_norm_method == 'None':
    return slim.batch_norm
  else:
    raise ValueError('Unsupported sync_batch_norm_method.')


def get_batch_norm_params(decay=0.9997,
                          epsilon=1e-5,
                          center=True,
                          scale=True,
                          is_training=True,
                          sync_batch_norm_method='None',
                          initialize_gamma_as_zeros=False):
  """Gets batch norm parameters.
  Args:
    decay: Float, decay for the moving average.
    epsilon: Float, value added to variance to avoid dividing by zero.
    center: Boolean. If True, add offset of `beta` to normalized tensor. If
      False,`beta` is ignored.
    scale: Boolean. If True, multiply by `gamma`. If False, `gamma` is not used.
    is_training: Boolean, whether or not the layer is in training mode.
    sync_batch_norm_method: String, method used to sync batch norm.
    initialize_gamma_as_zeros: Boolean, initializing `gamma` as zeros or not.
  Returns:
    A dictionary for batchnorm parameters.
  Raises:
    ValueError: If sync_batch_norm_method is not supported.
  """
  batch_norm_params = {
      'is_training': is_training,
      'decay': decay,
      'epsilon': epsilon,
      'scale': scale,
      'center': center,
  }
  if initialize_gamma_as_zeros:
    if sync_batch_norm_method == 'None':
      # Slim-type gamma_initialier.
      batch_norm_params['param_initializers'] = {
          'gamma': tf.zeros_initializer(),
      }
    else:
      raise ValueError('Unsupported sync_batch_norm_method.')
  return batch_norm_params



def fc_layer(x,
             layer_size,
             _std,
             reuse=None,
             scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.
  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.
  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
  Returns:
    Computed features after split separable conv2d.
  """
    # TODO: tf.layers.dense()?  SEnet-Tensorflow github
    input_size, output_size = layer_size
    with tf.variable_scope(scope, "fc_layer"):
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=[input_size, output_size], stddev=_std))
        w = w / tf.sqrt(input_size/2)
        b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[output_size]))
        output = tf.add(tf.matmul(x, w, name='fc'), b)
    return output


def conv2d(inputs,
          filter_shape,
          strides=[1,1,1,1],
          padding='SAME',
          dilations=[1,1,1,1],
          activate=None,
          is_training=None,
          reuse=False,
          scope=None):
    """Splits a separable conv2d into depthwise and pointwise conv2d.
  This operation differs from `tf.layers.separable_conv2d` as this operation
  applies activation function between depthwise and pointwise conv2d.
  Args:
    inputs: Input tensor with shape [batch, height, width, channels].
    filters: Number of filters in the 1x1 pointwise convolution.
    kernel_size: A list of length 2: [kernel_height, kernel_width] of
      of the filters. Can be an int if both values are the same.
    rate: Atrous convolution rate for the depthwise convolution.
    weight_decay: The weight decay to use for regularizing the model.
    depthwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for depthwise convolution.
    pointwise_weights_initializer_stddev: The standard deviation of the
      truncated normal weight initializer for pointwise convolution.
    scope: Optional scope for the operation.
  Returns:
    Computed features after split separable conv2d.
  """
    # TODO: remove numpy dependency
    assert inputs.get_shape().ndims == 4
    kernel_h, kernel_w, in_channels, out_channels = filter_shape
    _std = np.sqrt(2 / (kernel_h * kernel_w * out_channels))

    with tf.variable_scope(scope, reuse=reuse) as scope:
        w = tf.get_variable(
                "W",
                initializer=tf.truncated_normal(shape=filter_shape, stddev=_std))
        conv = tf.nn.conv2d(inputs,
                            w,
                            strides=strides,
                            padding=padding,
#                            dilations=dilations,
                            name='conv',
                            )

        if is_training is not None:
            output = batch_norm(conv, is_training=is_training, scope='batch_norm')
        else:
            b = tf.get_variable(
                "b",
                initializer=tf.constant(0.1, shape=[out_channels]))
            output = tf.nn.bias_add(conv, b)

        if activate is not None:
            output = activate(output)

    return output


def batch_norm(inputs, scope=None, is_training=None):
    """BN for the first input"""
    assert inputs.get_shape().ndims == 4
    with tf.variable_scope(scope):
        output = tf.layers.batch_normalization(inputs, training=is_training)
    return output


def resize_bilinear(images, size, output_dtype=tf.float32):
  """Returns resized images as output_type.
  Args:
    images: A tensor of size [batch, height_in, width_in, channels].
    size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
      for the images.
    output_dtype: The destination type.
  Returns:
    A tensor of size [batch, height_out, width_out, channels] as a dtype of
      output_dtype.
  """
  images = tf.image.resize_bilinear(images, size, align_corners=False)
  return tf.cast(images, dtype=output_dtype)


def scale_dimension(dim, scale):
  """Scales the input dimension.
  Args:
    dim: Input dimension (a scalar or a scalar Tensor).
    scale: The amount of scaling applied to the input.
  Returns:
    Scaled dimension.
  """
  if isinstance(dim, tf.Tensor):
    return tf.cast((tf.to_float(dim) - 1.0) * scale + 1.0, dtype=tf.int32)
  else:
    return int((float(dim) - 1.0) * scale + 1.0)

# def se_block(inputs, node=32, scope=None):
#   with tf.variable_scope(scope, "se_block", reuse=tf.AUTO_REUSE):
#       channel = inputs.get_shape().as_list()[3]
#       net = inputs
#       net = tf.reduce_mean(net, [1,2], keep_dims=False)
#       net = fc_layer(net, [channel, 32], _std=1, scope="fc1")
#       net = tf.nn.relu(net)
#       net = fc_layer(net, [32, channel], _std=1, scope="fc2")
#       net = tf.nn.sigmoid(net)
#   return net

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
