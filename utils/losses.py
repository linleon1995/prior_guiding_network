import six
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
import matplotlib.pyplot as plt

from core import preprocess_utils
from core import utils
_EPSILON = 1e-9
LOSSES_MAP = {"softmax_cross_entropy": add_softmax_cross_entropy_loss_for_each_scale,
              "softmax_dice_loss": add_softmax_dice_loss_for_each_scale,
              "sigmoid_cross_entropy": add_sigmoid_cross_entropy_loss_for_each_scale,
              "sigmoid_dice_loss": add_sigmoid_dice_loss_for_each_scale,
              "softmax_generaled_dice_loss": add_softmax_generaled_dice_loss_for_each_scale,}


def get_label_weight_mask(labels, ignore_label, num_classes, label_weights=1.0, keep_class_dims=False):
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
  if keep_class_dims:
    not_ignore_mask = tf.tile(tf.expand_dims(not_ignore_mask, axis=1), [1, num_classes])
  if isinstance(label_weights, float):
    return not_ignore_mask * label_weights


  label_weights = tf.constant(label_weights, tf.float32)
  if keep_class_dims:
    all_classes_label = tf.tile(tf.expand_dims(tf.ones_like(labels, dtype=tf.float32), axis=1), [1, num_classes])
    weight_mask = label_weights * all_classes_label
  else:
    # Dot product
    weight_mask = tf.einsum('...y,y->...',
                            tf.one_hot(labels, num_classes, dtype=tf.float32),
                            label_weights)

  return tf.multiply(not_ignore_mask, weight_mask)


def _div_maybe_zero(total_loss, num_present):
    """Normalizes the total loss with the number of present pixels."""
    return tf.to_float(num_present > 0) * tf.math.divide(
        total_loss,
        tf.maximum(1e-5, num_present))


def add_sigmoid_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  dilated_kernel=None,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  gt_is_matting_map=False,
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale."""
  add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                labels,
                                                num_classes,
                                                ignore_label,
                                                loss_weight,
                                                upsample_logits,
                                                hard_example_mining_step,
                                                top_k_percent_pixels,
                                                gt_is_matting_map,
                                                activation="sigmoid",
                                                scope=scope)


def add_sigmoid_dice_loss_for_each_scale(scales_to_logits,
                                         labels,
                                         num_classes,
                                         ignore_label,
                                         alpha=0.5,
                                         beta=0.5,
                                         loss_weight=1.0,
                                         scope=None):
  """Adds sigmoid dice loss for logits of each scale."""
  add_softmax_dice_loss_for_each_scale(scales_to_logits,
                                        labels,
                                        num_classes,
                                        ignore_label,
                                        alpha,
                                        beta,
                                        loss_weight,
                                        activation="sigmoid",
                                        scope=scope)


def add_softmax_generaled_dice_loss_for_each_scale(scales_to_logits,
                                                   labels,
                                                   num_classes,
                                                   ignore_label,
                                                   alpha=0.5,
                                                   beta=0.5,
                                                   loss_weight=1.0,
                                                   scope=None):
    """Adds softmax genraled dice loss (GDL) for logits of each scale."""
    if labels is None:
        raise ValueError('No label for softmax dice loss.')

    for scale, logits in scales_to_logits.items():
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        shape = preprocess_utils.resolve_shape(labels, 4)
        logits = tf.image.resize_bilinear(
                logits,
                shape[1:3],
                align_corners=True)
        scaled_labels = labels

        scaled_labels = tf.reshape(scaled_labels, shape=[-1, shape[1]*shape[2]])

        logits = tf.reshape(logits, shape=[-1, shape[1]*shape[2], num_classes])
        train_labels = tf.one_hot(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)

        # The reciprocal of label square for loss weight
        area = tf.reduce_sum(train_labels, axis=1)
        weights = tf.ones_like(area) / (tf.square(area)+_EPSILON)
        weights = tf.where(tf.greater(weights, tf.ones_like(weights)), tf.zeros_like(weights), weights)
        weights = weights * loss_weight
        with tf.name_scope(loss_scope, 'softmax_all_pixel_loss',
                        [logits, train_labels, weights]):
            # Compute the loss for all pixels.
            prediction = tf.nn.softmax(logits, 2)
            train_labels = tf.stop_gradient(
                train_labels, name='train_labels_stop_gradient')

            intersection = tf.reduce_sum(train_labels*prediction, axis=1)
            union = tf.reduce_sum(train_labels, axis=1) + tf.reduce_sum(prediction, axis=1)

            weighted_intersection = tf.reduce_sum(tf.multiply(intersection, weights), axis=1)
            weighted_union = tf.reduce_sum(tf.multiply(union, weights), axis=1)
            loss = 1 - 2*tf.reduce_mean((weighted_intersection+_EPSILON)/(weighted_union+_EPSILON))

            tf.losses.add_loss(loss)


def add_softmax_dice_loss_for_each_scale(scales_to_logits,
                                         labels,
                                         num_classes,
                                         ignore_label,
                                         alpha=0.5,
                                         beta=0.5,
                                         loss_weight=1.0,
                                         activation="softmax",
                                         scope=None):
    """Adds softmax dice loss for logits of each scale."""
    if labels is None:
        raise ValueError('No label for softmax dice loss.')

    for scale, logits in scales_to_logits.items():
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)

        logits = tf.image.resize_bilinear(logits,
                                          preprocess_utils.resolve_shape(labels, 4)[1:3],
                                          align_corners=True)

        labels = tf.reshape(labels, shape=[-1])
        weights = tf.constant(loss_weight, tf.float32)

        logits = tf.reshape(logits, shape=[-1, num_classes])
        train_labels = tf.one_hot(
            labels, num_classes, on_value=1.0, off_value=0.0)

        with tf.name_scope(loss_scope, '%s_all_pixel_loss' %activation,
                        [logits, train_labels, weights]):
            # Compute the loss for all pixels.
            if activation == "softmax":
              prediction = tf.nn.softmax(logits, 1)
            elif activation == "sigmoid":
              prediction = tf.nn.sigmoid(logits)
            else:
              raise ValueError("Unknown activation for prediction")
            train_labels = tf.stop_gradient(
                train_labels, name='train_labels_stop_gradient')

            intersection = tf.reduce_sum(train_labels*prediction, 0)
            union = tf.reduce_sum(train_labels, 0) + tf.reduce_sum(prediction, 0)

            pixel_losses = (2*intersection+_EPSILON) / (union+_EPSILON)
            weighted_pixel_losses = tf.multiply(pixel_losses, weights)
            loss = 1 - tf.reduce_mean(weighted_pixel_losses)

            tf.losses.add_loss(loss)


def add_softmax_cross_entropy_loss_for_each_scale(scales_to_logits,
                                                  labels,
                                                  num_classes,
                                                  ignore_label,
                                                  loss_weight=1.0,
                                                  upsample_logits=True,
                                                  hard_example_mining_step=0,
                                                  top_k_percent_pixels=1.0,
                                                  gt_is_matting_map=False,
                                                  activation="softmax",
                                                  scope=None):
  """Adds softmax cross entropy loss for logits of each scale.
  Args:
    scales_to_logits: A map from logits names for different scales to logits.
      The logits have shape [batch, logits_height, logits_width, num_classes].
    labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
    num_classes: Integer, number of target classes.
    ignore_label: Integer, label to ignore.
    loss_weight: A float or a list of loss weights. If it is a float, it means
      all the labels have the same weight. If it is a list of weights, then each
      element in the list represents the weight for the label of its index, for
      example, loss_weight = [0.1, 0.5] means the weight for label 0 is 0.1 and
      the weight for label 1 is 0.5.
    upsample_logits: Boolean, upsample logits or not.
    hard_example_mining_step: An integer, the training step in which the hard
      exampling mining kicks off. Note that we gradually reduce the mining
      percent to the top_k_percent_pixels. For example, if
      hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
      mining percent will gradually reduce from 100% to 25% until 100K steps
      after which we only mine top 25% pixels.
    top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
      < 1.0, only compute the loss for the top k percent pixels (e.g., the top
      20% pixels). This is useful for hard pixel mining.
    gt_is_matting_map: If true, the groundtruth is a matting map of confidence
      score. If false, the groundtruth is an integer valued class mask.
    scope: String, the scope for the loss.
  Raises:
    ValueError: Label or logits is None, or groundtruth is matting map while
      label is not floating value.
  """
  if labels is None:
    raise ValueError('No label for softmax cross entropy loss.')

  # If input groundtruth is a matting map of confidence, check if the input
  # labels are floating point values.
  if gt_is_matting_map and not labels.dtype.is_floating:
    raise ValueError('Labels must be floats if groundtruth is a matting map.')

  for scale, logits in six.iteritems(scales_to_logits):
    loss_scope = None
    if scope:
      loss_scope = '%s_%s' % (scope, scale)

    if upsample_logits:
      # Label is not downsampled, and instead we upsample logits.
      logits = tf.image.resize_bilinear(
          logits,
          preprocess_utils.resolve_shape(labels, 4)[1:3],
          align_corners=True)
      scaled_labels = labels
    else:
      # Label is downsampled to the same size as logits.
      # When gt_is_matting_map = true, label downsampling with nearest neighbor
      # method may introduce artifacts. However, to avoid ignore_label from
      # being interpolated with other labels, we still perform nearest neighbor
      # interpolation.
      # TODO(huizhongc): Change to bilinear interpolation by processing padded
      # and non-padded label separately.
      if gt_is_matting_map:
        tf.logging.warning(
            'Label downsampling with nearest neighbor may introduce artifacts.')

      scaled_labels = tf.image.resize_nearest_neighbor(
          labels,
          preprocess_utils.resolve_shape(logits, 4)[1:3],
          align_corners=True)

    scaled_labels = tf.reshape(scaled_labels, shape=[-1])
    if activation == "sigmoid":
      keep_class_dims = True
      loss_func = tf.nn.sigmoid_cross_entropy_with_logits
    elif activation == "softmax":
      keep_class_dims = False
      loss_func = tf.nn.softmax_cross_entropy_with_logits_v2
    else:
      raise ValueError("Unknown activation for prediction")
    weights = get_label_weight_mask(
        scaled_labels, ignore_label, num_classes, label_weights=loss_weight, keep_class_dims=keep_class_dims)
    # Dimension of keep_mask is equal to the total number of pixels.
    keep_mask = tf.cast(
        tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)

    train_labels = None
    logits = tf.reshape(logits, shape=[-1, num_classes])

    if gt_is_matting_map:
      # When the groundtruth is integer label mask, we can assign class
      # dependent label weights to the loss. When the groundtruth is image
      # matting confidence, we do not apply class-dependent label weight (i.e.,
      # label_weight = 1.0).
      if loss_weight != 1.0:
        raise ValueError(
            'loss_weight must equal to 1 if groundtruth is matting map.')

      # Assign label value 0 to ignore pixels. The exact label value of ignore
      # pixel does not matter, because those ignore_value pixel losses will be
      # multiplied to 0 weight.
      train_labels = scaled_labels * keep_mask

      train_labels = tf.expand_dims(train_labels, 1)
      train_labels = tf.concat([1 - train_labels, train_labels], axis=1)
    else:
      train_labels = tf.one_hot(
          scaled_labels, num_classes, on_value=1.0, off_value=0.0)

    default_loss_scope = ('softmax_all_pixel_loss'
                          if top_k_percent_pixels == 1.0 else
                          'softmax_hard_example_mining')
    with tf.name_scope(loss_scope, default_loss_scope,
                       [logits, train_labels, weights]):
      # Compute the loss for all pixels.
      pixel_losses = loss_func(
          labels=tf.stop_gradient(
              train_labels, name='train_labels_stop_gradient'),
          logits=logits,
          name='pixel_losses')
      weighted_pixel_losses = tf.multiply(pixel_losses, weights)

      if top_k_percent_pixels == 1.0:
        total_loss = tf.reduce_sum(weighted_pixel_losses)
        num_present = tf.reduce_sum(keep_mask)
        loss = _div_maybe_zero(total_loss, num_present)
        tf.losses.add_loss(loss)
      else:
        num_pixels = tf.to_float(tf.shape(logits)[0])
        # Compute the top_k_percent pixels based on current training step.
        if hard_example_mining_step == 0:
          # Directly focus on the top_k pixels.
          top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
        else:
          # Gradually reduce the mining percent to top_k_percent_pixels.
          global_step = tf.to_float(tf.train.get_or_create_global_step())
          ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
          top_k_pixels = tf.to_int32(
              (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
        top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
                                      k=top_k_pixels,
                                      sorted=True,
                                      name='top_k_percent_pixels')
        total_loss = tf.reduce_sum(top_k_losses)
        num_present = tf.reduce_sum(
            tf.to_float(tf.not_equal(top_k_losses, 0.0)))
        loss = _div_maybe_zero(total_loss, num_present)
        tf.losses.add_loss(loss)


