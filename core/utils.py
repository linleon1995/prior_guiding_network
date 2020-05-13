import numpy as np
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from tensorflow.contrib import slim as contrib_slim

slim = contrib_slim


# Quantized version of sigmoid function.
q_sigmoid = lambda x: tf.nn.relu6(x + 3) * 0.16667


class Refine(object):
  def __init__(self, low_level, fusions, guidance=None, embed_node=32, num_class=14, 
               weight_decay=0.0, scope=None, is_training=None):
    self.low_level = low_level
    self.fusions = fusions
    self.guidance = guidance
    self.embed_node = embed_node
    self.num_class = num_class
    self.weight_decay = weight_decay
    self.scope = scope
    assert len(self.low_level) == len(self.fusions)
    self.num_stage = len(self.low_level)
    if is_training is not None:
      self.arg_scope = slim.arg_scope([slim.batch_norm], is_training=is_training)
    else:
      self.arg_scope = slim.arg_scope([])
  
  def embed(self, x, out_node, scope):
    return slim.conv2d(x, out_node, kernel_size=[1,1], scope=scope)
    
  def model(self):
    with tf.variable_scope(self.scope, 'Refine_Network'):
      with self.arg_scope:
        with slim.arg_scope([slim.conv2d], 
                          trainable=True,
                          activation_fn=tf.nn.relu, 
                          weights_initializer=tf.initializers.he_normal(), 
                          weights_regularizer=slim.l2_regularizer(self.weight_decay),
                          kernel_size=[3, 3], 
                          padding='SAME',
                          normalizer_fn=slim.batch_norm):
          y_tm1 = None
          preds = {}
          
#          embed5 = self.low_level["low_level5"]
#          y5 = tf.identity(embed5, name="identity5")
#          
#          embed4 = self.low_level["low_level4"]
#          h, w = embed4.get_shape().as_list()[1:3]
#          y_tm1 = tf.image.resize_bilinear(y5, [h, w], align_corners=True)
#          y_tm1 = slim.conv2d(y_tm1, 256, scope="conv4_1")
#          y4 = self.get_fusion_method(self.fusions[0])(embed4, y_tm1, 256, self.fusions[0]+str(4))
#
#          embed3 = self.low_level["low_level3"]
#          h, w = embed3.get_shape().as_list()[1:3]
#          y_tm1 = tf.image.resize_bilinear(y4, [h, w], align_corners=True)
#          y_tm1 = slim.conv2d(y_tm1, 128, scope="conv3_1")
#          y3 = self.get_fusion_method(self.fusions[1])(embed3, y_tm1, 128, self.fusions[1]+str(3))
#
#          embed2 = self.low_level["low_level2"]
#          h, w = embed2.get_shape().as_list()[1:3]
#          y_tm1 = tf.image.resize_bilinear(y3, [h, w], align_corners=True)
#          y_tm1 = slim.conv2d(y_tm1, 64, scope="conv2_1")
#          y2 = self.get_fusion_method(self.fusions[2])(embed2, y_tm1, 64, self.fusions[2]+str(2))
#
#          embed1 = self.low_level["low_level1"]
#          h, w = embed1.get_shape().as_list()[1:3]
#          y_tm1 = tf.image.resize_bilinear(y2, [h, w], align_corners=True)
#          y_tm1 = slim.conv2d(y_tm1, 32, scope="conv1_1")
#          y = self.get_fusion_method(self.fusions[3])(embed1, y_tm1, 32, self.fusions[3]+str(1))

          out_node = self.embed_node
          for i, (k, v) in enumerate(self.low_level.items()):
            embed = self.embed(v, out_node, scope="embed%d" %(self.num_stage-i))
            
            fuse_func = self.get_fusion_method(self.fusions[i])
            if self.fusions[i] in ("concat", "sum"):
              if y_tm1 is not None:
                h, w = embed.get_shape().as_list()[1:3]
                y_tm1 = tf.image.resize_bilinear(y_tm1, [h, w], align_corners=True)
                y = fuse_func(embed, y_tm1, out_node, self.fusions[i]+str(self.num_stage-i))
              else:
                y = tf.identity(embed, name="identity%d" %(self.num_stage-i))
            elif self.fusions[i] in ("guid"):
              guid = self.get_guidance(self.guidance, out_node, self.fusions[i]+str(self.num_stage-i))
              tf.add_to_collection("f", guid)
              y = fuse_func(embed, y_tm1, guid, out_node, self.fusions[i]+str(self.num_stage-i))
              
            
#            out_node //= 2
            preds["guidance%d" %(self.num_stage-i)] = slim.conv2d(y, self.num_class, kernel_size=[1,1], 
                                                                  activation_fn=None, 
                                                                  stride=1, scope="pred%d" %(self.num_stage-i))
            y_tm1 = y
          y = tf.image.resize_bilinear(y, [256, 256], align_corners=True)
          y = slim.conv2d(y, self.embed_node, scope="decoder_output")
          y = slim.conv2d(y, self.num_class, kernel_size=[1, 1], stride=1, activation_fn=None, scope='logits')
    return y, preds
  
  def get_fusion_method(self, method):
    if method == "concat":
      return self.concat_convolution
    elif method == "sum":
      return self.sum_convolution
    elif method == "guid":
      return self.guid_attention
  
  def concat_convolution(self, x1, x2, out_node, scope):
    net = slim.conv2d(tf.concat([x1, x2], axis=3), out_node, scope=scope+"_1")
#    net = slim.conv2d(net, out_node, scope=scope+"_2")
    return net
  
  def sum_convolution(self, x1, x2, out_node, scope):
    net = slim.conv2d(x1 + x2, out_node, scope=scope+"_1")
#    net = slim.conv2d(net, out_node, scope=scope+"_2")
    return net
  
  def guid_attention(self, x1, x2, guid, out_node, scope):
    net = slim.conv2d(x1, out_node, scope=scope+"_1")
    y1 = x1 + net * guid
    if x2 is not None:
      mid = x2 + y1
    else:
      mid = y1
    net = slim.conv2d(mid, out_node, scope=scope+"_2")
    y2 = mid + net * guid
    return y2
  
  def get_guidance(self, guidance, out_node, scope):
    num_stage = 3
    root = 1
    net = guidance
    for s in range(1, num_stage+1):
      channel = out_node * root**(s-1)
      net = slim.conv2d(net, channel, scope='down_conv%d_1' %s)
      net = slim.conv2d(net, channel, scope='down_conv%d_2' %s)
      net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], "VALID")
    net = slim.conv2d(net, channel, scope='down_conv%d_1' %(num_stage+1))
    net = slim.conv2d(net, channel, scope='down_conv%d_2' %(num_stage+1))
    return tf.nn.sigmoid(net)

def slim_unet(net, num_stage, base_channel=32, root=2, weight_decay=1e-3, scope=None, is_training=None):  
    # TODO: batch_norm parameter 
    with tf.variable_scope(scope, 'U-net'):
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
            low_level = []
            for s in range(1, num_stage+1):
              channel = base_channel * root**(s-1)
              net = slim.conv2d(net, channel, scope='down_conv%d_1' %s)
              net = slim.conv2d(net, channel, scope='down_conv%d_2' %s)
              low_level.append(net)
              net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], "VALID")
              
            net = slim.conv2d(net, channel*root, scope='bottle_neck1')
            net = slim.conv2d(net, channel*root, scope='bottle_neck2')
            
            for s in range(num_stage, 0, -1):
              h, w = low_level[s-1].get_shape().as_list()[1:3]
              channel = base_channel * root**s
              
              net = tf.image.resize_bilinear(net, [h, w])
              net = tf.concat([net, low_level[s-1]], axis=3)
              net = slim.conv2d(net, channel, scope='up_conv%d_1' %s)
              net = slim.conv2d(net, channel//root, scope='up_conv%d_2' %s)
    return net      
            
            
def slim_extractor(net, num_stage, base_channel=32, root=2, weight_decay=1e-3, scope=None, is_training=None):  
    # TODO: batch_norm parameter 
    with tf.variable_scope(scope, 'U-net'):
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
            for s in range(1, num_stage+1):
              channel = base_channel * root**(s-1)
              net = slim.conv2d(net, channel, scope='down_conv%d_1' %s)
              net = slim.conv2d(net, channel, scope='down_conv%d_2' %s)
              net = tf.nn.max_pool(net, [1,2,2,1], [1,2,2,1], "VALID")
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
def get_label_weight_mask(labels, ignore_label, num_classes, label_weights=1.0):
  """Gets the label weight mask.
  Args:
    labels: A Tensor of labels with the shape of [-1].
    ignore_label: Integer, label to ignore.
    num_classes: Integer, the number of semantic classes.
    label_weights: A float or a list of weights. If it is a float, it means all
      the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, label_weights = [0.1, 0.5] means the weight for label 0 is 0.1
      and the weight for label 1 is 0.5.
  Returns:
    A Tensor of label weights with the same shape of labels, each element is the
      weight for the label with the same index in labels and the element is 0.0
      if the label is to ignore.
  Raises:
    ValueError: If label_weights is neither a float nor a list, or if
      label_weights is a list and its length is not equal to num_classes.
  """
  if not isinstance(label_weights, (float, list)):
    raise ValueError(
        'The type of label_weights is invalid, it must be a float or a list.')

  if isinstance(label_weights, list) and len(label_weights) != num_classes:
    raise ValueError(
        'Length of label_weights must be equal to num_classes if it is a list, '
        'label_weights: %s, num_classes: %d.' % (label_weights, num_classes))

  not_ignore_mask = tf.not_equal(labels, ignore_label)
  not_ignore_mask = tf.cast(not_ignore_mask, tf.float32)
  if isinstance(label_weights, float):
    return not_ignore_mask * label_weights

  label_weights = tf.constant(label_weights, tf.float32)
  weight_mask = tf.einsum('...y,y->...',
                          tf.one_hot(labels, num_classes, dtype=tf.float32),
                          label_weights)
  return tf.multiply(not_ignore_mask, weight_mask)


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


def conv_block(inputs, channels=32, scope=None, is_training=True):
    with tf.variable_scope(scope, "conv_block"):
      in_channel = inputs.get_shape().as_list()[3]
      conv1 = conv2d(inputs, [3,3,in_channel,channels], activate=tf.nn.relu, is_training=is_training, scope="conv1")
      conv2 = conv2d(conv1, [3,3,channels,channels], activate=tf.nn.relu, is_training=is_training, scope="conv2")
    return conv2


def _simple_decoder(inputs, out, stage=None, channels=None, scope=None, is_training=None):
    for s in range(1, stage+1):
      h, w = inputs.get_shape().as_list()[1:3]
      up = tf.image.resize_bilinear(inputs, [2*h, 2*w])
      inputs = conv_block(up, channels, "up_conv_block"+str(s), is_training)
    outputs = conv2d(inputs, [3,3,channels,out], activate=None, is_training=is_training, scope="output_conv")
    return outputs    
           
           
def _simple_unet(inputs, out, stage=None, channels=None, scope=None, is_training=None):
    with tf.variable_scope(scope, 'simple_unet'):
      down_conv_list = []
      down = inputs
      c = channels
      for s in range(1, stage+1):
        conv = conv_block(down, c, "down_conv_block"+str(s), is_training)
        down_conv_list.append(conv)
        down = tf.nn.max_pool(conv, [1,2,2,1], [1,2,2,1], "VALID")
        c *= 2
        
      conv = conv_block(down, c//2, "latent", is_training)
      
      for s in range(1, stage+1):
        down = down_conv_list[stage-s]
        h, w = down.get_shape().as_list()[1:3]
        up = tf.image.resize_bilinear(conv, [h, w])
        conv = conv_block(tf.concat([up,down], axis=3), c, "up_conv_block"+str(s), is_training)
        c //= 2
        
      return  conv2d(conv, [1,1,channels,out], activate=None, is_training=is_training, scope="output_conv")
        
def __2d_unet_decoder(features, layers_dict, num_class, channels=None, scope=None, is_training=None):
    with tf.variable_scope(scope, '2d_unet'):
        i = 1
        root = 2
        for v in layers_dict.values():
            if i == 1:
                with tf.variable_scope("block1"):
                  conv = conv2d(features, [3,3,2048,channels], activate=tf.nn.relu, is_training=is_training, scope="conv1")
                  conv = conv2d(conv, [3,3,channels,channels], activate=tf.nn.relu, is_training=is_training, scope="conv2")
            else:
                conv = _2d_unet_block(conv, v, channels, is_training, "block"+str(i))
            i += 1    
            channels //= root
        h, w, c = conv.get_shape().as_list()[1:4]
        conv = tf.image.resize_bilinear(conv, [2*h, 2*w])
        conv1 = conv2d(conv, [3,3,c,c], activate=tf.nn.relu, is_training=is_training, scope="conv1")
        conv2 = conv2d(conv1, [3,3,c,c], activate=tf.nn.relu, is_training=is_training, scope="conv2")
        logits = conv2d(conv2, [1,1,c,num_class], activate=None, is_training=is_training, scope="logits")
    return logits
    
    
def _2d_unet_block(inputs, feature, embed=None, is_training=True, scope=None):
    with tf.variable_scope(scope, "2d_unet_block"):
        in_channel = inputs.get_shape().as_list()[3]  
        h, w, c = feature.get_shape().as_list()[1:4]  
        if embed is not None:
            out_channels = embed
        else:
            out_channels = c
        inputs = tf.image.resize_bilinear(inputs, [h, w])
        inputs = conv2d(inputs, [3,3,in_channel,out_channels], activate=tf.nn.relu, is_training=is_training, scope="up_conv")
        concat_feature = tf.concat([inputs, feature], axis=3)
        conv1 = conv2d(concat_feature, [3,3,c+out_channels,out_channels], activate=tf.nn.relu, is_training=is_training, scope="conv1")
        conv2 = conv2d(conv1, [3,3,out_channels,out_channels], activate=tf.nn.relu, is_training=is_training, scope="conv2")
    return conv2

# conv2d(inputs, 
#           filter_shape, 
#           strides=[1,1,1,1],
#           padding='SAME',
#           dilations=[1,1,1,1],  
#           activate=None,
#           is_training=None, 
#           reuse=False,
#           scope=None):

# def _2d_unet_decoder(low_feature_dict, is_training=True, scope=None):
#     with tf.variable_scope(scope, "2d_unet_decoder"):
#         def concat_and_conv(inputs, feature):
#             concat_feature = tf.concat([inputs, feature], axis=3)
#             c = concat_feature.get_shape().as_list()[3] 
#             outputs = conv2d(concat_feature, [3,3,c,c//2], tf.nn.relu, scope="conv_r1_1", is_training=is_training)
#             return outputs
        
#         embed=64
#         inputs = low_feature_dict["low_level4"]
#         for k, v in low_feature_dict:
#             c = v.get_shape().as_list()[3] 
#             inputs = 
#             concat_feature = tf.concat([inputs, v], axis=3)  
#             conv1 = conv2d(concat_feature, [3,3,2*c,c], tf.nn.relu, "conv1", is_training)
#             conv2 = 
#         c = images.get_shape().as_list()[3]
        
#         outputs = conv2d
#         conv2d(in_node, [3,3,channels,num_filters], activate=tf.nn.relu, scope="conv_r1_1", is_training=is_training)
#     return outputs


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


# def atrous_conv2d(inputs, 
#           filter_shape, 
#           strides=[1,1,1,1],
#           padding='SAME',
#           rate=1,  
#           activate=None, 
#           bn_flag=True, 
#           is_training=True, 
#           reuse=False,
#           scope=None):
#     # TODO: temporally implement for api 1.4.1, merge with conv2d after update tensorflow version
#     """Splits a separable conv2d into depthwise and pointwise conv2d.
#   This operation differs from `tf.layers.separable_conv2d` as this operation
#   applies activation function between depthwise and pointwise conv2d.
#   Args:
#     inputs: Input tensor with shape [batch, height, width, channels].
#     filters: Number of filters in the 1x1 pointwise convolution.
#     kernel_size: A list of length 2: [kernel_height, kernel_width] of
#       of the filters. Can be an int if both values are the same.
#     rate: Atrous convolution rate for the depthwise convolution.
#     weight_decay: The weight decay to use for regularizing the model.
#     depthwise_weights_initializer_stddev: The standard deviation of the
#       truncated normal weight initializer for depthwise convolution.
#     pointwise_weights_initializer_stddev: The standard deviation of the
#       truncated normal weight initializer for pointwise convolution.
#     scope: Optional scope for the operation.
#   Returns:
#     Computed features after split separable conv2d.
#   """
#     assert inputs.get_shape().ndims == 4
#     kernel_h, kernel_w, in_channels, out_channels = filter_shape
#     _std = np.sqrt(2 / (kernel_h * kernel_w * out_channels))

#     with tf.variable_scope(scope, reuse=reuse) as scope:
#         w = tf.get_variable(
#                 "W",
#                 initializer=tf.truncated_normal(shape=filter_shape, stddev=_std))
#         conv = tf.nn.atrous_conv2d(inputs, 
#                             w, 
#                             rate=rate,
#                             padding=padding,
#                             name='atrous_conv',
#                             )
        
#         if bn_flag:
#             output = batch_norm(conv, is_training=is_training, scope='batch_norm')
#         else:
#             b = tf.get_variable(
#                 "b",
#                 initializer=tf.constant(0.1, shape=[out_channels]))
#             output = tf.nn.bias_add(conv, b)
            
#         if activate is not None:
#             output = activate(output) 

#     return output

      
# def split_separable_conv2d(inputs, 
#                           filter_shape, 
#                           strides=[1,1,1,1],
#                           padding='SAME',
#                           dilations=[1,1], 
#                           channel_multiplier=1, 
#                           activate_func=None, 
#                           bn_flag=True, 
#                           is_training=True,
#                           reuse=False,
#                           scope=None):
#     """Splits a separable conv2d into depthwise and pointwise conv2d.
#   This operation differs from `tf.layers.separable_conv2d` as this operation
#   applies activation function between depthwise and pointwise conv2d.
#   Args:
#     inputs: Input tensor with shape [batch, height, width, channels].
#     filters: Number of filters in the 1x1 pointwise convolution.
#     kernel_size: A list of length 2: [kernel_height, kernel_width] of
#       of the filters. Can be an int if both values are the same.
#     rate: Atrous convolution rate for the depthwise convolution.
#     weight_decay: The weight decay to use for regularizing the model.
#     depthwise_weights_initializer_stddev: The standard deviation of the
#       truncated normal weight initializer for depthwise convolution.
#     pointwise_weights_initializer_stddev: The standard deviation of the
#       truncated normal weight initializer for pointwise convolution.
#     scope: Optional scope for the operation.
#   Returns:
#     Computed features after split separable conv2d.
#   """
#     assert inputs.get_shape().ndims == 4
#     kernel_h, kernel_w, in_channels, out_channels = filter_shape
#     _std = np.sqrt(2 / (kernel_h * kernel_w * out_channels))
    
#     with tf.variable_scope(scope, reuse=reuse):
#         w_depthwise = tf.get_variable(
#                 "W_depth",
#                 initializer=tf.truncated_normal(shape=[kernel_h, kernel_w, in_channels, channel_multiplier], stddev=_std))
#         w_pointwise = tf.get_variable(
#                 "W_point",
#                 initializer=tf.truncated_normal(shape=[1,1, channel_multiplier*in_channels, out_channels], stddev=_std))
#         conv = tf.nn.separable_conv2d(
#                 input=inputs,
#                 depthwise_filter=w_depthwise,
#                 pointwise_filter=w_pointwise,
#                 strides=strides,
#                 padding=padding,
#                 rate=dilations,
#                 name='separable_conv',
#                 )
        
#         if bn_flag:
#             out_node = batch_norm(conv, is_training=is_training, name='batch_norm')
#         else:
#             b = tf.get_variable(
#                 "b",
#                 initializer=tf.constant(shape=out_channels, stddev=0.1))
#             out_node = tf.nn.bias_add(conv, b)
        
#         if activate_func is not None:
#             output = activate_func(out_node)    
#     return output
        

def batch_norm(inputs, scope=None, is_training=None):
    """BN for the first input"""
    assert inputs.get_shape().ndims == 4
    with tf.variable_scope(scope):
        output = tf.layers.batch_normalization(inputs, training=is_training)
    return output

    
def upsampling_layer(inputs, n, mode, scope='upsampling'):
    """upsampling layer
    Args:
    Return:
    """
    assert inputs.get_shape().ndims == 4
    x_shape = tf.shape(inputs)
    new_h = x_shape[1]*n
    new_w = x_shape[2]*n
    
    with tf.variable_scope(scope):
        if mode is 'Nearest_Neighbor':
            output = tf.image.resize_nearest_neighbor(inputs, [new_h, new_w], name='nearest_neighbor')
            
        if mode is 'Bilinear':
            output = tf.image.resize_bilinear(inputs, [new_h, new_w], name='bilinear')
        
        if mode is 'Bicubic':
            output = tf.image.resize_bicubic(inputs, [new_h, new_w], name='bicubic')

    return output


# def global_avg_pool(inputs, keep_dims=False):
#     """global_avg_pooling"""
#     assert inputs.get_shape().ndims == 4
#     with tf.variable_scope('global_average_pooling'):
#         output = tf.reduce_mean(inputs, [1, 2], keep_dims=keep_dims)
#     return output
    

# def global_max_pool(inputs, keep_dims=False):
#     """global_max_pooling"""
#     assert inputs.get_shape().ndims == 4
#     with tf.variable_scope('global_average_pooling'):
#         output = tf.reduce_max(inputs, [1, 2], keepdims=keep_dims)
#     return output


# def resize_bilinear(images, size, output_dtype=tf.float32):
#   """Returns resized images as output_type.
#   Args:
#     images: A tensor of size [batch, height_in, width_in, channels].
#     size: A 1-D int32 Tensor of 2 elements: new_height, new_width. The new size
#       for the images.
#     output_dtype: The destination type.
#   Returns:
#     A tensor of size [batch, height_out, width_out, channels] as a dtype of
#       output_dtype.
#   """
#   images = tf.image.resize_bilinear(images, size, align_corners=True)
#   return tf.cast(images, dtype=output_dtype)


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


# def split_separable_conv2d(inputs,
#                            filters,
#                            kernel_size=3,
#                            rate=1,
#                            weight_decay=0.00004,
#                            depthwise_weights_initializer_stddev=0.33,
#                            pointwise_weights_initializer_stddev=0.06,
#                            scope=None):
#   """Splits a separable conv2d into depthwise and pointwise conv2d.
#   This operation differs from `tf.layers.separable_conv2d` as this operation
#   applies activation function between depthwise and pointwise conv2d.
#   Args:
#     inputs: Input tensor with shape [batch, height, width, channels].
#     filters: Number of filters in the 1x1 pointwise convolution.
#     kernel_size: A list of length 2: [kernel_height, kernel_width] of
#       of the filters. Can be an int if both values are the same.
#     rate: Atrous convolution rate for the depthwise convolution.
#     weight_decay: The weight decay to use for regularizing the model.
#     depthwise_weights_initializer_stddev: The standard deviation of the
#       truncated normal weight initializer for depthwise convolution.
#     pointwise_weights_initializer_stddev: The standard deviation of the
#       truncated normal weight initializer for pointwise convolution.
#     scope: Optional scope for the operation.
#   Returns:
#     Computed features after split separable conv2d.
#   """
#   outputs = slim.separable_conv2d(
#       inputs,
#       None,
#       kernel_size=kernel_size,
#       depth_multiplier=1,
#       rate=rate,
#       weights_initializer=tf.truncated_normal_initializer(
#           stddev=depthwise_weights_initializer_stddev),
#       weights_regularizer=None,
#       scope=scope + '_depthwise')
#   return slim.conv2d(
#       outputs,
#       filters,
#       1,
#       weights_initializer=tf.truncated_normal_initializer(
#           stddev=pointwise_weights_initializer_stddev),
#       weights_regularizer=slim.l2_regularizer(weight_decay),
#       scope=scope + '_pointwise')


