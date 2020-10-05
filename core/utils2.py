import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim
from core import resnet_v1_beta, preprocess_utils, cell, attentions
slim = contrib_slim
resnet_v1_beta_block = resnet_v1_beta.resnet_v1_beta_block


def guidance_fusion_method(logits, guid_fuse, num_class, out_node, level, k_size_list=None):
  if guid_fuse == "sum":
    guid = tf.reduce_sum(logits, axis=3, keepdims=True)
  elif guid_fuse == "mean":
    guid = tf.reduce_mean(logits, axis=3, keepdims=True)
  elif guid_fuse == "entropy":
    guid = tf.clip_by_value(logits, 1e-10, 1.0)
    guid = -tf.reduce_sum(guid * tf.log(guid), axis=3, keepdims=True)
  elif guid_fuse == "conv":
    guid = slim.conv2d(logits, out_node, kernel_size=[3,3], activation_fn=None)
  elif guid_fuse == "sum_dilated":
    size = [8,6,4,2,1]
    kernel = tf.ones((size[level], size[level], num_class))
    guid = tf.nn.dilation2d(logits, filter=kernel, strides=(1,1,1,1),
                              rates=(1,1,1,1), padding="SAME")
    guid = guid - tf.ones_like(guid)
    guid = tf.reduce_sum(guid, axis=3, keepdims=True)
  elif guid_fuse == "w_sum":
    w = tf.nn.softmax(tf.reduce_sum(logits, axis=[1,2], keepdims=True), axis=3)
    rev_w = tf.ones_like(w) - w
    guid = tf.reduce_sum(tf.multiply(logits, rev_w), axis=3, keepdims=True)
  elif guid_fuse == "conv_sum":
    if k_size_list is None:
      raise ValueError("%s need kernel size list" %guid_fuse)
    k_size = 2 * k_size_list[level] + 1
    guid = slim.conv2d(logits, 1, kernel_size=[k_size,k_size], activation_fn=None,
                      weights_initializer=tf.ones_initializer(), trainable=False, normalizer_fn=None)
    guid = guid / (k_size*k_size*num_class*1)
  elif guid_fuse == "w_sum_conv":
    if k_size_list is None:
      raise ValueError("%s need kernel size list" %guid_fuse)
    k_size = 3 * k_size_list[level] + 1
    w = tf.reduce_sum(logits, axis=[1,2], keepdims=True)
    rev_w = (tf.ones_like(w)+1e-5) / (tf.sqrt(w)+1e-5)
    rev_w = tf.tile(rev_w, [1,k_size,k_size,1])
    rev_w = tf.expand_dims(rev_w, axis=4)

    n, h, w, channels_img = preprocess_utils.resolve_shape(logits, rank=4)
    n, fh, fw, channels, out_channels = preprocess_utils.resolve_shape(rev_w, rank=5)
    # F has shape (n, k_size, k_size, channels, out_channels)

    rev_w = tf.transpose(rev_w, [1, 2, 0, 3, 4])
    rev_w = tf.reshape(rev_w, [fh, fw, channels*n, out_channels])

    guid = tf.transpose(logits, [1, 2, 0, 3]) # shape (H, W, MB, channels_img)
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
  elif guid_fuse == "sum_wo_back":
    flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
    guid = tf.multiply(logits, flag)
    guid = tf.reduce_sum(guid, axis=3, keepdims=True)
  elif guid_fuse == "mean_wo_back":
    flag = tf.concat([tf.zeros([1,1,1,1]), tf.ones([1,1,1,num_class-1])], axis=3)
    guid = tf.multiply(logits, flag)
    guid = tf.reduce_mean(guid, axis=3, keepdims=True)
  elif guid_fuse == "same":
    pass
  else:
    raise ValueError("Unknown guid fuse")

  tf.add_to_collection("guidance", guid)
  return guid


def sram(in_node,
              guidance,
              num_conv=1,
              conv_type="conv",
              conv_node=64,
              scope=None):
    """Single Residual Attention Module"""
    with tf.variable_scope(scope, "sram", reuse=tf.AUTO_REUSE):
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


def concat_convolution(cur, last, out_node, scope=None):
	with tf.variable_scope(scope, "concat_conv"):
		h, w = preprocess_utils.resolve_shape(cur, rank=4)[1:3]
		last = resize_bilinear(last, [h, w])
		net = slim.conv2d(tf.concat([cur, last], axis=3), out_node, scope="conv1")
		return net


def sum_convolution(cur, last, out_node, scope=None):
	with tf.variable_scope(scope, "sum_cocnv"):
		h, w = preprocess_utils.resolve_shape(cur, rank=4)[1:3]
		last = resize_bilinear(last, [h, w])
		net = slim.conv2d(cur+last, out_node, scope="conv1")
		return net


def guid_attention(cur, last, guid, out_node, scope=None, guid_conv_nums=2,
                   guid_conv_type="conv2d", apply_sram2=True):
  """Guid attention module"""
  h, w = preprocess_utils.resolve_shape(cur, rank=4)[1:3]
	guid_node = preprocess_utils.resolve_shape(guid, rank=4)[3]
	if guid_node != out_node and guid_node != 1:
		raise ValueError("Unknown guidance node number %d, should be 1 or out_node" %guid_node)

	with tf.variable_scope(scope, 'guid_attention'):

		guid = resize_bilinear(guid, [h, w])
		last = resize_bilinear(last, [h, w])
		net = sram(cur, guid, guid_conv_nums, guid_conv_type, out_node, "sram1")
		tf.add_to_collection("sram1", net)
		if last is not None:
			net = net + last
			if apply_sram2:
				net = sram(net, guid, guid_conv_nums, guid_conv_type, out_node, "sram2")
		tf.add_to_collection("sram2", net)
		return net


  def guid_class_attention(cur, last, guid, num_class, out_node, scope=None, guid_conv_nums=2,
                           guid_conv_type="conv2d", apply_sram2=True):
    """Guid class attention module"""
    h, w = preprocess_utils.resolve_shape(cur, rank=4)[1:3]
    guid_node = guid.get_shape().as_list()[3]
    if guid_node != num_class:
      raise ValueError("Unknown guidance node number %d, should equal class number" %guid_node)
    with tf.variable_scope(scope, "guid_class_attention"):
      guid = resize_bilinear(guid, [h, w])
			last = resize_bilinear(last, [h, w])
      total_att = []
      for i in range(1, num_class):
        net = sram(cur, guid[...,i:i+1], guid_conv_nums, guid_conv_type, out_node, "sram1")
        if last is not None:
          net = net + last
          if apply_sram2:
          	net = sram(net, guid[...,i:i+1], guid_conv_nums, guid_conv_type, out_node, "sram2")
        total_att.append(net)
      fuse = slim.conv2d(tf.concat(total_att, axis=3), out_node, kernel_size=[1,1], scope="fuse")
    return fuse


  def context_attention(cur, last, guid, out_node, scope=None, guid_conv_nums=2, guid_conv_type="conv2d"):
    guid_node = preprocess_utils.resolve_shape(guid, rank=4)[3]

    if guid_node != out_node and guid_node != 1:
      raise ValueError("Unknown guidance node number %d, should be 1 or out_node" %guid_node)

    with tf.variable_scope(scope, 'context_attention'):
      context = sram(cur, guid, guid_conv_nums, guid_conv_type, embed_node, "sram1")
      tf.add_to_collection("sram1", context)
      if last is not None:
        ca_layer = attentions.self_attention(out_node)
        net = ca_layer(last, context, last, "context_att1")
        tf.add_to_collection("context_att1", net)
      return net


  def self_attention(cur, last, guid, out_node, scope=None, guid_conv_nums=2, guid_conv_type="conv2d"):
    guid_node = preprocess_utils.resolve_shape(guid, rank=4)[3]

    if guid_node != out_node and guid_node != 1:
      raise ValueError("Unknown guidance node number %d, should be 1 or out_node" %guid_node)

    with tf.variable_scope(scope, 'self_attention'):
      net = sram(cur, guid, guid_conv_nums, guid_conv_type, embed_node, "sram1")
      tf.add_to_collection("sram1", net)
      if last is not None:
        net = net + last
        sa_layer = attentions.self_attention(out_node)
        net = sa_layer(net, net, net, "self_att1")
        tf.add_to_collection("self_att1", net)
      return net


def mlp(inputs,
        output_dims,
        num_layers=3,
        decreasing_root=8,
        scope=None,):
  """
  Multilayer Perceptron
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


def gcn(x, out_node, k_size=7, scope=None):
	"""Global Convolutional Network"""
	with tf.variable_scope(scope, "GCN", reuse=tf.AUTO_REUSE):
		x_l1 = slim.conv2d(x, out_node, kernel_size=[k_size,1], scope="x_l1")
		x_l2 = slim.conv2d(x_l1, out_node, kernel_size=[1,k_size], scope="x_l2")

		x_r1 = slim.conv2d(x, out_node, kernel_size=[1,k_size], scope="x_r1")
		x_r2 = slim.conv2d(x_r1, out_node, kernel_size=[k_size,1], scope="x_r2")

		x = x_l2 + x_r2
	return x


def fc_layer(x,
             layer_size,
             _std,
             reuse=None,
             scope=None):
	"""Fully connected layer"""
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


def se_block(inputs, out_node, scope=None):
  # TODO: check feature shape and dimension
  """SENet"""
  with tf.variable_scope(scope, "se_block", reuse=tf.AUTO_REUSE):
      channel = inputs.get_shape().as_list()[3]
      net = tf.reduce_mean(inputs, [1,2], keep_dims=False)
      net = fc_layer(net, [channel, out_node], _std=1, scope="fc1")
      net = tf.nn.relu(net)
      net = fc_layer(net, [out_node, channel], _std=1, scope="fc2")
      net = tf.nn.sigmoid(net)
      net = inputs * net
  return net


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

