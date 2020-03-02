import functools
import tensorflow as tf
slim = tf.contrib.slim

from core import resnet_v1_beta
from core import utils
from tensorflow.contrib.slim.nets import resnet_utils
fc_layer = utils.fc_layer
_MEAN_RGB = [123.15, 115.90, 103.06]


# A map from feature extractor name to the network name scope used in the
# ImageNet pretrained versions of these models.
name_scope = {
    'resnet_v1_50': 'resnet_v1_50',
    'resnet_v1_50_beta': 'resnet_v1_50',
    'resnet_v1_101': 'resnet_v1_101',
    'resnet_v1_101_beta': 'resnet_v1_101',
}

# A map from network name to network arg scope.
arg_scopes_map = {
    'resnet_v1_50': resnet_utils.resnet_arg_scope,
    'resnet_v1_50_beta': resnet_utils.resnet_arg_scope,
    'resnet_v1_101': resnet_utils.resnet_arg_scope,
    'resnet_v1_101_beta': resnet_utils.resnet_arg_scope,
}

# A map from network name to network function.
networks_map = {
    'resnet_v1_50': resnet_v1_beta.resnet_v1_50,
    'resnet_v1_50_beta': resnet_v1_beta.resnet_v1_50_beta,
    'resnet_v1_101': resnet_v1_beta.resnet_v1_101,
    'resnet_v1_101_beta': resnet_v1_beta.resnet_v1_101_beta,
}


# Mean pixel value.
_MEAN_RGB = [123.15, 115.90, 103.06]


def _preprocess_subtract_imagenet_mean(inputs, dtype=tf.float32):
  """Subtract Imagenet mean RGB value."""
  mean_rgb = tf.reshape(_MEAN_RGB, [1, 1, 1, 3])
  num_channels = tf.shape(inputs)[-1]
  # We set mean pixel as 0 for the non-RGB channels.
  mean_rgb_extended = tf.concat(
      [mean_rgb, tf.zeros([1, 1, 1, num_channels - 3])], axis=3)
  return tf.cast(inputs - mean_rgb_extended, dtype=dtype)


def _preprocess_zero_mean_unit_range(inputs, dtype=tf.float32):
  """Map image values from [0, 255] to [-1, 1]."""
  preprocessed_inputs = (2.0 / 255.0) * tf.to_float(inputs) - 1.0
  return tf.cast(preprocessed_inputs, dtype=dtype)


_PREPROCESS_FN = {
    'resnet_v1_50': _preprocess_subtract_imagenet_mean,
    'resnet_v1_50_beta': _preprocess_zero_mean_unit_range,
    'resnet_v1_101': _preprocess_subtract_imagenet_mean,
    'resnet_v1_101_beta': _preprocess_zero_mean_unit_range,
}


def mean_pixel(model_variant=None):
  """Gets mean pixel value.
  This function returns different mean pixel value, depending on the input
  model_variant which adopts different preprocessing functions. We currently
  handle the following preprocessing functions:
  (1) _preprocess_subtract_imagenet_mean. We simply return mean pixel value.
  (2) _preprocess_zero_mean_unit_range. We return [127.5, 127.5, 127.5].
  The return values are used in a way that the padded regions after
  pre-processing will contain value 0.
  Args:
    model_variant: Model variant (string) for feature extraction. For
      backwards compatibility, model_variant=None returns _MEAN_RGB.
  Returns:
    Mean pixel value.
  """
  if model_variant in ['resnet_v1_50',
                       'resnet_v1_101'] or model_variant is None:
    return _MEAN_RGB
  else:
    return [127.5, 127.5, 127.5]
  
  
def extract_features(images,
                    output_stride=8,
                    multi_grid=None,
                  #  depth_multiplier=1.0,
                  #  divisible_by=None,
                  #  final_endpoint=None,
                    model_variant=None,
                    weight_decay=0.0001,
                    reuse=None,
                    is_training=False,
                    fine_tune_batch_norm=False,
                  #  regularize_depthwise=False,
                    preprocess_images=True,
                    preprocessed_images_dtype=tf.float32,
                    num_classes=None,
                    global_pool=False,):
  """Extracts features by the particular model_variant.
  Args:
    images: A tensor of size [batch, height, width, channels].
    output_stride: The ratio of input to output spatial resolution.
    multi_grid: Employ a hierarchy of different atrous rates within network.
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops used in MobileNet.
    divisible_by: None (use default setting) or an integer that ensures all
      layers # channels will be divisible by this number. Used in MobileNet.
    final_endpoint: The MobileNet endpoint to construct the network up to.
    model_variant: Model variant for feature extraction.
    weight_decay: The weight decay for model variables.
    reuse: Reuse the model variables or not.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    regularize_depthwise: Whether or not apply L2-norm regularization on the
      depthwise convolution weights.
    preprocess_images: Performs preprocessing on images or not. Defaults to
      True. Set to False if preprocessing will be done by other functions. We
      supprot two types of preprocessing: (1) Mean pixel substraction and (2)
      Pixel values normalization to be [-1, 1].
    preprocessed_images_dtype: The type after the preprocessing function.
    num_classes: Number of classes for image classification task. Defaults
      to None for dense prediction tasks.
    global_pool: Global pooling for image classification task. Defaults to
      False, since dense prediction tasks do not use this.
    nas_architecture_options: A dictionary storing NAS architecture options.
      It is either None or its kerys are:
      - `nas_stem_output_num_conv_filters`: Number of filters of the NAS stem
        output tensor.
      - `nas_use_classification_head`: Boolean, use image classification head.
    nas_training_hyper_parameters: A dictionary storing hyper-parameters for
      training nas models. It is either None or its keys are:
      - `drop_path_keep_prob`: Probability to keep each path in the cell when
        training.
      - `total_training_steps`: Total training steps to help drop path
        probability calculation.
    use_bounded_activation: Whether or not to use bounded activations. Bounded
      activations better lend themselves to quantized inference. Currently,
      bounded activation is only used in xception model.
  Returns:
    features: A tensor of size [batch, feature_height, feature_width,
      feature_channels], where feature_height/feature_width are determined
      by the images height/width and output_stride.
    end_points: A dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: Unrecognized model variant.
  """
  if 'resnet' in model_variant:
    arg_scope = arg_scopes_map[model_variant](
        weight_decay=weight_decay,
        batch_norm_decay=0.95,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True)
    features, end_points = get_network(
        model_variant, preprocess_images, preprocessed_images_dtype, arg_scope)(
            inputs=images,
            num_classes=num_classes,
            is_training=(is_training and fine_tune_batch_norm),
            global_pool=global_pool,
            output_stride=output_stride,
            multi_grid=multi_grid,
            reuse=reuse,
            scope=name_scope[model_variant])
  else:
    raise ValueError('Unknown model variant %s.' % model_variant)

  return features, end_points


def global_extractor(inputs, 
                       output_dims, 
                       num_layers=3, 
                       decreasing_root=8, 
                       global_pool='average',
                       scope=None):
  """
  Args:
    inputs:
    output_dims
    num_layers:
    decreasing_root:
    global_pool='average:
  Returns:
  Raises:

  """
  # TODO: raise if input dims smaller than output_dims
  with tf.variable_scope(scope, 'non_image_extractor') as sc:
    net = inputs
    # net = slim.conv2d(inputs, output_dims, [1, 1], stride=1, scope='segmentations')
    if global_pool == 'average':
      net = tf.reduce_mean(net, [1, 2], name='global_avg_pool', keep_dims=False)
    elif global_pool == 'max':
      net = tf.reduce_max(net, [1, 2], name='global_max_pool', keepp_dims=False)
    else:
      ValueError("Unkonwn global_pool Keyword")
    # outputs = net
    # TODO: activation??
    # outputs = fc_layer(net, [2048, output_dims], _std=1, reuse=tf.AUTO_REUSE, scope='_'.join(['fc', str(0)]))
    # outputs = tf.reduce_mean(net, axis=1, keep_dims=True)
    for i in range(num_layers):
      # TODO:
      # print(net, 30*'o')
      dims = net.get_shape().as_list()[1]
      if i < num_layers-1:
        net = fc_layer(net, [dims, dims//decreasing_root], _std=1, reuse=tf.AUTO_REUSE, scope='_'.join(['fc', str(i)]))
      else:
        outputs = fc_layer(net, [dims, output_dims], _std=1, reuse=tf.AUTO_REUSE, scope='_'.join(['fc', str(i)]))

  return outputs
  
  
def get_network(network_name, preprocess_images,
                preprocessed_images_dtype=tf.float32, arg_scope=None):
  """Gets the network.
  Args:
    network_name: Network name.
    preprocess_images: Preprocesses the images or not.
    preprocessed_images_dtype: The type after the preprocessing function.
    arg_scope: Optional, arg_scope to build the network. If not provided the
      default arg_scope of the network would be used.
  Returns:
    A network function that is used to extract features.
  Raises:
    ValueError: network is not supported.
  """
  if network_name not in networks_map:
    raise ValueError('Unsupported network %s.' % network_name)
  arg_scope = arg_scope or arg_scopes_map[network_name]()
  def _identity_function(inputs, dtype=preprocessed_images_dtype):
    return tf.cast(inputs, dtype=dtype)
  if preprocess_images:
    preprocess_function = _PREPROCESS_FN[network_name]
  else:
    preprocess_function = _identity_function
  func = networks_map[network_name]
  @functools.wraps(func)
  def network_fn(inputs, *args, **kwargs):
    with slim.arg_scope(arg_scope):
      return func(preprocess_function(inputs, preprocessed_images_dtype),
                  *args, **kwargs)
  return network_fn
