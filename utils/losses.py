import six
import numpy as np
import nibabel as nib
import glob
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
import matplotlib.pyplot as plt

from core import preprocess_utils
from core import utils
import common
# from utils import loss_utils
_EPSILON = 1e-5


# def binary_focal_sigmoid_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=True):
#     ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_true, labels=y_pred)

#     # If logits are provided then convert the predictions into probabilities
#     if from_logits:
#         pred_prob = tf.sigmoid(y_pred)
#     else:
#         pred_prob = y_pred

#     p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
#     alpha_factor = 1.0
#     modulating_factor = 1.0

#     if alpha:
#         alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

#     if gamma:
#         modulating_factor = tf.pow((1.0 - p_t), gamma)

#     # compute the final loss and return
#     return tf.reduce_sum(alpha_factor * modulating_factor * ce)
                         
#   # p = tf.nn.sigmoid(labels)
#   # q = 1 - p
  
#   # p = tf.math.maximum(p, _EPSILON)
#   # q = tf.math.maximum(q, _EPSILON)
    
#   # pos_loss = -alpha * ((1-p)**gamma) * tf.log(p)
#   # neg_loss = -alpha * ((1-q)**gamma) * tf.log(q)
#   # focal_loss = labels*pos_loss + (1-labels)*neg_loss
#   # focal_loss = tf.reduce_sum(focal_loss)
#   # return focal_loss


# def loss_utils(logits, labels, cost_name, **cost_kwargs):
#     # TODO: unified all loss to using logits instead of using after final activate function
#     """
#     Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
#     Optional arguments are: 
#     class_weights: weights for the different classes in case of multi-class imbalance
#     regularizer: power of the L2 regularizers added to the loss function
#     """
#     if cost_name == "cross_entropy":
#         add_softmax_cross_entropy_loss_for_each_scale(logits,
#                                                     labels,
#                                                     14,
#                                                     -1,
#                                                     loss_weight=[0.04]+13*[1.0])
#         loss = tf.losses.get_losses()[0]
#         # flat_logits = tf.reshape(logits, [-1, logits.get_shape()[-1]])
#         # flat_labels = tf.reshape(labels, [-1, labels.get_shape()[-1]])
        
#         # class_weights = cost_kwargs.pop("class_weights", None)
        
#         # if class_weights is not None:
#         #     class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    
#         #     weight_map = tf.multiply(flat_labels, class_weights)
#         #     weight_map = tf.reduce_sum(weight_map, axis=1)
    
#         #     loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
#         #                                                         labels=flat_labels)
#         #     weighted_loss = tf.multiply(loss_map, weight_map)
    
#         #     loss = tf.reduce_mean(weighted_loss)
            
#         # else:
#         #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
#         #                                                                     labels=flat_labels))

            
#     elif cost_name == "KL_divergence":
#         eps = 1e-5
#         labels = tf.exp(labels)
#         loss = tf.reduce_mean(tf.reduce_sum(labels * tf.log((eps+labels)/(eps+logits)), axis=3))
        
#     elif cost_name == "cross_entropy_sigmoid":
#         flat_logits = tf.reshape(logits, [-1, logits.get_shape()[-1]])
#         flat_labels = tf.reshape(labels, [-1, labels.get_shape()[-1]])
        
#         class_weights = cost_kwargs.pop("class_weights", None)
        
#         if class_weights is not None:
#             class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    
#             weight_map = tf.multiply(flat_labels, class_weights)
#             weight_map = tf.reduce_sum(weight_map, axis=1)
    
#             loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits,
#                                                                 labels=flat_labels)
#             weighted_loss = tf.multiply(loss_map, weight_map)
    
#             loss = tf.reduce_mean(weighted_loss)
            
#         else:
#             loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, 
#                                                                             labels=flat_labels))
            
#     elif cost_name == "cross_entropy_zlabel":
#         class_weights = cost_kwargs.pop("class_weights", None)
#         z_class = cost_kwargs.pop('z_class', None)
        
#         labels = tf.one_hot(indices=labels,
#                             depth=int(z_class),
#                             on_value=1,
#                             off_value=0,
#                             axis=-1,
#                             )
#         if class_weights is not None:
#             class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    
#             weight_map = tf.multiply(labels, class_weights)
#             weight_map = tf.reduce_sum(weight_map, axis=1)
    
#             loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
#                                                                 labels=labels)
#             weighted_loss = tf.multiply(loss_map, weight_map)
    
#             loss = tf.reduce_mean(weighted_loss)
            
#         else:
#             loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
#                                                                             labels=labels))
            
#     elif cost_name == "mean_dice_coefficient":       
#         eps = 1e-5
#         batch_size = cost_kwargs.pop("batch_size", None)
#         # TODO: num_class variables or change in datagenerator
#         is_onehot = cost_kwargs.pop("is_onehot", True)
#         if is_onehot:
#           labels = tf.one_hot(indices=labels,
#                               depth=14,
#                               on_value=1,
#                               off_value=0,
#                               axis=-1,
#                               )
#         gt = tf.reshape(labels, [-1, labels.get_shape()[-1]])
#         gt = tf.cast(gt, tf.float32)
#         prediction = tf.nn.softmax(logits)
#         prediction = tf.reshape(prediction, [-1, logits.get_shape()[-1]])
        
#         intersection = tf.reduce_sum(gt*prediction, 0)
#         union = tf.reduce_sum(gt, 0) + tf.reduce_sum(prediction, 0)

#         loss = (2*intersection+eps) / (union+eps)
#         loss = 1 - tf.reduce_mean(loss)

#         # if is_onehot:
#         #   labels = tf.one_hot(indices=labels,
#         #                       depth=14,
#         #                       on_value=1,
#         #                       off_value=0,
#         #                       axis=-1,
#         #                       )
#         # gt = tf.reshape(labels, [batch_size, -1, labels.get_shape()[-1]])
#         # gt = tf.cast(gt, tf.float32)
#         # prediction = tf.nn.softmax(logits)
#         # prediction = tf.reshape(prediction, [batch_size, -1, logits.get_shape()[-1]])
        
#         # intersection = tf.reduce_sum(gt*prediction, axis=1)
#         # union = tf.reduce_sum(gt, axis=1) + tf.reduce_sum(prediction, axis=1)
#         # loss = 1 - tf.reduce_mean((2*intersection+eps)/(union+eps), axis=1)
#         # loss = tf.reduce_mean(loss)

#     elif cost_name == "MSE":
#         loss = tf.losses.mean_squared_error(
#                                         labels,
#                                         logits,
#                                         )
#     elif cost_name == "binary_focal_sigmoid":
#       alpha = cost_kwargs.pop("alpha", 0.25)
#       gamma = cost_kwargs.pop("gamma", 2.0)
#       flat_logits = tf.reshape(logits, [-1, ])
#       flat_labels = tf.reshape(labels, [-1, ])
#       loss = binary_focal_sigmoid_loss(flat_labels, flat_logits, alpha, gamma)   
      
#     else:
#         raise ValueError("Unknown cost function: "%cost_name)

#     # regularizer = cost_kwargs.pop("regularizer", None)
#     # if regularizer is not None:
#     #     regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
#     #     loss += (regularizer * regularizers)
#     return loss
    
    
# def get_losses(output_dict, 
#                layers_dict, 
#                samples, 
#                loss_dict, 
#                batch_size=None):
#     # TODO: auxlarity loss in latent
#     losses = []
#     ys = tf.one_hot(samples[common.LABEL][...,0], 14, on_value=1.0, off_value=0.0)

#     # Calculate segmentation loss
#     seg_loss = loss_utils(output_dict[common.OUTPUT_TYPE], samples[common.LABEL], 
#                           cost_name=loss_dict[common.OUTPUT_TYPE]["loss"],
#                           batch_size=batch_size)
#     seg_loss = tf.identity(seg_loss, name='/'.join(['segmentation_loss', loss_dict[common.OUTPUT_TYPE]["loss"]]))
#     losses.append(seg_loss)
    
#     # Calculate z loss
#     if common.OUTPUT_Z in loss_dict:
#         z_loss = loss_utils(output_dict[common.OUTPUT_Z], samples[common.Z_LABEL], cost_name=loss_dict[common.OUTPUT_Z]["loss"])
#         z_loss = tf.multiply(loss_dict[common.OUTPUT_Z]["decay"], z_loss, 
# 							 name='/'.join(['z_loss', loss_dict[common.OUTPUT_Z]["loss"]]))
#         losses.append(z_loss)

#     # Calculate guidance loss
#     if common.GUIDANCE in loss_dict:
#         guidance_loss = 0
        
# 		# Upsample logits in each stage with tf loss func# Upsample logits in each stage with tf loss func
#         ny = samples[common.LABEL].get_shape()[1]
#         nx = samples[common.LABEL].get_shape()[2]
        
        
#         for name, value in layers_dict.items():
#             if 'guidance' in name:
#                 value = tf.compat.v2.image.resize(value, [ny, nx])
#                 guidance_loss += loss_utils(value, samples[common.LABEL], cost_name=loss_dict[common.GUIDANCE]["loss"])
#         guidance_loss = tf.multiply(loss_dict[common.GUIDANCE]["decay"], guidance_loss,
#                                     name='/'.join(['guidance_loss', loss_dict[common.GUIDANCE]["loss"]]))

#         losses.append(guidance_loss)  

#     # Calculate transformation loss  
#     if "transform" in loss_dict:
#         guid = tf.compat.v2.image.resize(output_dict[common.GUIDANCE], [256,256])
#         transform_loss = loss_utils(guid, ys, cost_name=loss_dict["transform"]["loss"])
#         transform_loss = tf.multiply(loss_dict["transform"]["decay"], transform_loss, 
#                                      name='/'.join(['transform_loss', loss_dict["transform"]["loss"]]))                             
#         losses.append(transform_loss)
    
#     return losses



def _div_maybe_zero(total_loss, num_present):
    """Normalizes the total loss with the number of present pixels."""
    return tf.to_float(num_present > 0) * tf.math.divide(
        total_loss,
        tf.maximum(1e-5, num_present))


# def add_sigmoid_cross_entropy_loss_for_each_scale(scales_to_logits,
#                                                   labels,
#                                                   num_classes,
#                                                   ignore_label,
#                                                   loss_weight=1.0,
#                                                   upsample_logits=True,
#                                                   hard_example_mining_step=0,
#                                                   top_k_percent_pixels=1.0,
#                                                   gt_is_matting_map=False,
#                                                   scope=None):
#   """Adds softmax cross entropy loss for logits of each scale.
#   Args:
#     scales_to_logits: A map from logits names for different scales to logits.
#       The logits have shape [batch, logits_height, logits_width, num_classes].
#     labels: Groundtruth labels with shape [batch, image_height, image_width, 1].
#     num_classes: Integer, number of target classes.
#     ignore_label: Integer, label to ignore.
#     loss_weight: A float or a list of loss weights. If it is a float, it means
#       all the labels have the same weight. If it is a list of weights, then each
#       element in the list represents the weight for the label of its index, for
#       example, loss_weight = [0.1, 0.5] means the weight for label 0 is 0.1 and
#       the weight for label 1 is 0.5.
#     upsample_logits: Boolean, upsample logits or not.
#     hard_example_mining_step: An integer, the training step in which the hard
#       exampling mining kicks off. Note that we gradually reduce the mining
#       percent to the top_k_percent_pixels. For example, if
#       hard_example_mining_step = 100K and top_k_percent_pixels = 0.25, then
#       mining percent will gradually reduce from 100% to 25% until 100K steps
#       after which we only mine top 25% pixels.
#     top_k_percent_pixels: A float, the value lies in [0.0, 1.0]. When its value
#       < 1.0, only compute the loss for the top k percent pixels (e.g., the top
#       20% pixels). This is useful for hard pixel mining.
#     gt_is_matting_map: If true, the groundtruth is a matting map of confidence
#       score. If false, the groundtruth is an integer valued class mask.
#     scope: String, the scope for the loss.
#   Raises:
#     ValueError: Label or logits is None, or groundtruth is matting map while
#       label is not floating value.
#   """
#   if labels is None:
#     raise ValueError('No label for softmax cross entropy loss.')

#   # If input groundtruth is a matting map of confidence, check if the input
#   # labels are floating point values.
#   if gt_is_matting_map and not labels.dtype.is_floating:
#     raise ValueError('Labels must be floats if groundtruth is a matting map.')

#   for scale, logits in six.iteritems(scales_to_logits):
#     loss_scope = None
#     if scope:
#       loss_scope = '%s_%s' % (scope, scale)

#     if upsample_logits:
#       # Label is not downsampled, and instead we upsample logits.
#       logits = tf.image.resize_bilinear(
#           logits,
#           preprocess_utils.resolve_shape(labels, 4)[1:3],
#           align_corners=True)
#       scaled_labels = labels
#     else:
#       # Label is downsampled to the same size as logits.
#       # When gt_is_matting_map = true, label downsampling with nearest neighbor
#       # method may introduce artifacts. However, to avoid ignore_label from
#       # being interpolated with other labels, we still perform nearest neighbor
#       # interpolation.
#       # TODO(huizhongc): Change to bilinear interpolation by processing padded
#       # and non-padded label separately.
#       if gt_is_matting_map:
#         tf.logging.warning(
#             'Label downsampling with nearest neighbor may introduce artifacts.')

#       scaled_labels = tf.image.resize_nearest_neighbor(
#           labels,
#           preprocess_utils.resolve_shape(logits, 4)[1:3],
#           align_corners=True)

#     scaled_labels = tf.reshape(scaled_labels, shape=[-1])
#     weights = utils.get_label_weight_mask(
#         scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
#     # Dimension of keep_mask is equal to the total number of pixels.
#     keep_mask = tf.cast(
#         tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)

#     train_labels = None
#     logits = tf.reshape(logits, shape=[-1, num_classes])

#     if gt_is_matting_map:
#       # When the groundtruth is integer label mask, we can assign class
#       # dependent label weights to the loss. When the groundtruth is image
#       # matting confidence, we do not apply class-dependent label weight (i.e.,
#       # label_weight = 1.0).
#       if loss_weight != 1.0:
#         raise ValueError(
#             'loss_weight must equal to 1 if groundtruth is matting map.')

#       # Assign label value 0 to ignore pixels. The exact label value of ignore
#       # pixel does not matter, because those ignore_value pixel losses will be
#       # multiplied to 0 weight.
#       train_labels = scaled_labels * keep_mask

#       train_labels = tf.expand_dims(train_labels, 1)
#       train_labels = tf.concat([1 - train_labels, train_labels], axis=1)
#     else:
#       train_labels = tf.one_hot(
#           scaled_labels, num_classes, on_value=1.0, off_value=0.0)

#     default_loss_scope = ('softmax_all_pixel_loss'
#                           if top_k_percent_pixels == 1.0 else
#                           'softmax_hard_example_mining')
#     with tf.name_scope(loss_scope, default_loss_scope,
#                        [logits, train_labels, weights]):
#       # Compute the loss for all pixels.
#       pixel_losses = tf.nn.sigmoid_cross_entropy_with_logits(
#           labels=tf.stop_gradient(
#               train_labels, name='train_labels_stop_gradient'),
#           logits=logits,
#           name='pixel_losses')
#       # weighted_pixel_losses = tf.multiply(pixel_losses, weights)
#       weighted_pixel_losses = pixel_losses
#       if top_k_percent_pixels == 1.0:
#         total_loss = tf.reduce_sum(weighted_pixel_losses)
#         num_present = tf.reduce_sum(keep_mask)
#         loss = _div_maybe_zero(total_loss, num_present)
#         tf.losses.add_loss(loss)
#       else:
#         num_pixels = tf.to_float(tf.shape(logits)[0])
#         # Compute the top_k_percent pixels based on current training step.
#         if hard_example_mining_step == 0:
#           # Directly focus on the top_k pixels.
#           top_k_pixels = tf.to_int32(top_k_percent_pixels * num_pixels)
#         else:
#           # Gradually reduce the mining percent to top_k_percent_pixels.
#           global_step = tf.to_float(tf.train.get_or_create_global_step())
#           ratio = tf.minimum(1.0, global_step / hard_example_mining_step)
#           top_k_pixels = tf.to_int32(
#               (ratio * top_k_percent_pixels + (1.0 - ratio)) * num_pixels)
#         top_k_losses, _ = tf.nn.top_k(weighted_pixel_losses,
#                                       k=top_k_pixels,
#                                       sorted=True,
#                                       name='top_k_percent_pixels')
#         total_loss = tf.reduce_sum(top_k_losses)
#         num_present = tf.reduce_sum(
#             tf.to_float(tf.not_equal(top_k_losses, 0.0)))
#         loss = _div_maybe_zero(total_loss, num_present)
#         tf.losses.add_loss(loss)


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

    # TODO: tf.rank
    if scaled_labels.get_shape().as_list()[3] != num_classes:
      scaled_labels = tf.one_hot(
            scaled_labels[...,0], num_classes, on_value=1.0, off_value=0.0)

    if dilated_kernel is not None:
      scaled_labels = tf.nn.dilation2d(scaled_labels, filter=dilated_kernel, strides=(1,1,1,1), 
                                      rates=(1,1,1,1), padding="SAME")
      scaled_labels = scaled_labels - tf.ones_like(scaled_labels)
    train_labels = tf.reshape(scaled_labels, shape=[-1, num_classes])
    # weights = utils.get_label_weight_mask(
    #     scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
    # # Dimension of keep_mask is equal to the total number of pixels.
    keep_mask = tf.cast(
        tf.not_equal(train_labels, ignore_label), dtype=tf.float32)

    # train_labels = None
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
    
    default_loss_scope = ('softmax_all_pixel_loss'
                          if top_k_percent_pixels == 1.0 else
                          'softmax_hard_example_mining')
    with tf.name_scope(loss_scope, default_loss_scope,
                       [logits, train_labels]):
      # Compute the loss for all pixels.
      pixel_losses = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.stop_gradient(
              train_labels, name='train_labels_stop_gradient'),
          logits=logits,
          name='pixel_losses')
      # weighted_pixel_losses = tf.multiply(pixel_losses, weights)
      weighted_pixel_losses = pixel_losses
      if top_k_percent_pixels == 1.0:
        total_loss = tf.reduce_sum(weighted_pixel_losses)
        num_present = tf.reduce_sum(keep_mask)
        loss = _div_maybe_zero(total_loss, num_present//num_classes)
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
        
            
            
def add_softmax_dice_loss_for_each_scale(scales_to_logits,
                                         labels,
                                         num_classes,
                                         ignore_label,
                                         alpha=0.5,
                                         beta=0.5,
                                         loss_weight=1.0,
                                         scope=None):
    """TODO"""
    if labels is None:
        raise ValueError('No label for softmax dice loss.')
    
    for scale, logits in scales_to_logits.items():
        loss_scope = None
        if scope:
            loss_scope = '%s_%s' % (scope, scale)
        
        logits = tf.image.resize_bilinear(
                logits,
                preprocess_utils.resolve_shape(labels, 4)[1:3],
                align_corners=True)
        scaled_labels = labels

        scaled_labels = tf.reshape(scaled_labels, shape=[-1])
        weights = tf.constant(loss_weight, tf.float32)
        # weights = utils.get_label_weight_mask(
        #     scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
        
        # Dimension of keep_mask is equal to the total number of pixels.
        # keep_mask = tf.cast(
        #     tf.not_equal(scaled_labels, ignore_label), dtype=tf.float32)

        logits = tf.reshape(logits, shape=[-1, num_classes])
    
        train_labels = tf.one_hot(
            scaled_labels, num_classes, on_value=1.0, off_value=0.0)
        
        with tf.name_scope(loss_scope, 'softmax_all_pixel_loss',
                        [logits, train_labels, weights]):
            # Compute the loss for all pixels.
            prediction = tf.nn.softmax(logits, 1)
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
    weights = utils.get_label_weight_mask(
        scaled_labels, ignore_label, num_classes, label_weights=loss_weight)
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
      pixel_losses = tf.nn.softmax_cross_entropy_with_logits_v2(
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
