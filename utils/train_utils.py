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


def binary_focal_sigmoid_loss(y_true, y_pred, alpha=0.25, gamma=2.0, from_logits=True):
    ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_true, labels=y_pred)

    # If logits are provided then convert the predictions into probabilities
    if from_logits:
        pred_prob = tf.sigmoid(y_pred)
    else:
        pred_prob = y_pred

    p_t = (y_true * pred_prob) + ((1 - y_true) * (1 - pred_prob))
    alpha_factor = 1.0
    modulating_factor = 1.0

    if alpha:
        alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)

    if gamma:
        modulating_factor = tf.pow((1.0 - p_t), gamma)

    # compute the final loss and return
    return tf.reduce_sum(alpha_factor * modulating_factor * ce)
                         
  # p = tf.nn.sigmoid(labels)
  # q = 1 - p
  
  # p = tf.math.maximum(p, _EPSILON)
  # q = tf.math.maximum(q, _EPSILON)
    
  # pos_loss = -alpha * ((1-p)**gamma) * tf.log(p)
  # neg_loss = -alpha * ((1-q)**gamma) * tf.log(q)
  # focal_loss = labels*pos_loss + (1-labels)*neg_loss
  # focal_loss = tf.reduce_sum(focal_loss)
  # return focal_loss


def loss_utils(logits, labels, cost_name, **cost_kwargs):
    # TODO: unified all loss to using logits instead of using after final activate function
    """
    Constructs the cost function, either cross_entropy, weighted cross_entropy or dice_coefficient.
    Optional arguments are: 
    class_weights: weights for the different classes in case of multi-class imbalance
    regularizer: power of the L2 regularizers added to the loss function
    """
    if cost_name == "cross_entropy":
        add_softmax_cross_entropy_loss_for_each_scale(logits,
                                                    labels,
                                                    14,
                                                    -1)
        loss = tf.losses.get_losses()[0]
        # flat_logits = tf.reshape(logits, [-1, logits.get_shape()[-1]])
        # flat_labels = tf.reshape(labels, [-1, labels.get_shape()[-1]])
        
        # class_weights = cost_kwargs.pop("class_weights", None)
        
        # if class_weights is not None:
        #     class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    
        #     weight_map = tf.multiply(flat_labels, class_weights)
        #     weight_map = tf.reduce_sum(weight_map, axis=1)
    
        #     loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
        #                                                         labels=flat_labels)
        #     weighted_loss = tf.multiply(loss_map, weight_map)
    
        #     loss = tf.reduce_mean(weighted_loss)
            
        # else:
        #     loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits, 
        #                                                                     labels=flat_labels))

            
    elif cost_name == "KL_divergence":
        eps = 1e-5
        labels = tf.exp(labels)
        loss = tf.reduce_mean(tf.reduce_sum(labels * tf.log((eps+labels)/(eps+logits)), axis=3))
        
    elif cost_name == "cross_entropy_sigmoid":
        flat_logits = tf.reshape(logits, [-1, logits.get_shape()[-1]])
        flat_labels = tf.reshape(labels, [-1, labels.get_shape()[-1]])
        
        class_weights = cost_kwargs.pop("class_weights", None)
        
        if class_weights is not None:
            class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    
            weight_map = tf.multiply(flat_labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)
    
            loss_map = tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits,
                                                                labels=flat_labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
    
            loss = tf.reduce_mean(weighted_loss)
            
        else:
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=flat_logits, 
                                                                            labels=flat_labels))
            
    elif cost_name == "cross_entropy_zlabel":
        class_weights = cost_kwargs.pop("class_weights", None)
        z_class = cost_kwargs.pop('z_class', None)
        
        labels = tf.one_hot(indices=labels,
                            depth=int(z_class),
                            on_value=1,
                            off_value=0,
                            axis=-1,
                            )
        if class_weights is not None:
            class_weights = tf.constant(np.array(class_weights, dtype=np.float32))
    
            weight_map = tf.multiply(labels, class_weights)
            weight_map = tf.reduce_sum(weight_map, axis=1)
    
            loss_map = tf.nn.softmax_cross_entropy_with_logits(logits=logits,
                                                                labels=labels)
            weighted_loss = tf.multiply(loss_map, weight_map)
    
            loss = tf.reduce_mean(weighted_loss)
            
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, 
                                                                            labels=labels))
            
    elif cost_name == "mean_dice_coefficient":       
        eps = 1e-5
        batch_size = cost_kwargs.pop("batch_size", None)
        # TODO: num_class variables or change in datagenerator
        is_onehot = cost_kwargs.pop("is_onehot", True)
        if is_onehot:
          labels = tf.one_hot(indices=labels,
                              depth=14,
                              on_value=1,
                              off_value=0,
                              axis=-1,
                              )
        gt = tf.reshape(labels, [-1, labels.get_shape()[-1]])
        gt = tf.cast(gt, tf.float32)
        prediction = tf.nn.softmax(logits)
        prediction = tf.reshape(prediction, [-1, logits.get_shape()[-1]])
        
        intersection = tf.reduce_sum(gt*prediction, 0)
        union = tf.reduce_sum(gt, 0) + tf.reduce_sum(prediction, 0)

        loss = (2*intersection+eps) / (union+eps)
        loss = 1 - tf.reduce_mean(loss)

        # if is_onehot:
        #   labels = tf.one_hot(indices=labels,
        #                       depth=14,
        #                       on_value=1,
        #                       off_value=0,
        #                       axis=-1,
        #                       )
        # gt = tf.reshape(labels, [batch_size, -1, labels.get_shape()[-1]])
        # gt = tf.cast(gt, tf.float32)
        # prediction = tf.nn.softmax(logits)
        # prediction = tf.reshape(prediction, [batch_size, -1, logits.get_shape()[-1]])
        
        # intersection = tf.reduce_sum(gt*prediction, axis=1)
        # union = tf.reduce_sum(gt, axis=1) + tf.reduce_sum(prediction, axis=1)
        # loss = 1 - tf.reduce_mean((2*intersection+eps)/(union+eps), axis=1)
        # loss = tf.reduce_mean(loss)

    elif cost_name == "MSE":
        loss = tf.losses.mean_squared_error(
                                        labels,
                                        logits,
                                        )
    elif cost_name == "binary_focal_sigmoid":
      alpha = cost_kwargs.pop("alpha", 0.25)
      gamma = cost_kwargs.pop("gamma", 2.0)
      flat_logits = tf.reshape(logits, [-1, ])
      flat_labels = tf.reshape(labels, [-1, ])
      loss = binary_focal_sigmoid_loss(flat_labels, flat_logits, alpha, gamma)   
      
    else:
        raise ValueError("Unknown cost function: "%cost_name)

    # regularizer = cost_kwargs.pop("regularizer", None)
    # if regularizer is not None:
    #     regularizers = sum([tf.nn.l2_loss(variable) for variable in self.variables])
    #     loss += (regularizer * regularizers)
    return loss
    
    
def get_losses(output_dict, 
               layers_dict, 
               samples, 
               loss_dict, 
               z_loss_decay=None, 
               transformed_loss_decay=None, 
               guidance_loss_decay=None,
               transform_loss_decay=None,
               batch_size=None):
    # TODO: auxlarity loss in latent
    losses = []
    seg_loss = loss_utils(output_dict[common.OUTPUT_TYPE], samples[common.LABEL], 
                          cost_name=loss_dict[common.OUTPUT_TYPE],
                          batch_size=batch_size)
    seg_loss = tf.identity(seg_loss, name='/'.join(['segmentation_loss', loss_dict[common.OUTPUT_TYPE]]))
    losses.append(seg_loss)
    
    # Calculate z loss
    if common.OUTPUT_Z in output_dict:
        z_loss = loss_utils(output_dict[common.OUTPUT_Z], samples[common.Z_LABEL], cost_name=loss_dict[common.OUTPUT_Z])
        z_loss = tf.multiply(z_loss_decay, z_loss, name='/'.join(['z_loss', loss_dict[common.OUTPUT_Z]]))
        losses.append(z_loss)

    # Calculate guidance loss
    if common.GUIDANCE in output_dict:
      guidance_loss = 0
      
      # Upsample logits in each stage with tf loss func
      ny = samples[common.LABEL].get_shape()[1]
      nx = samples[common.LABEL].get_shape()[2]
      ys = tf.one_hot(
          samples[common.LABEL][...,0], 14, on_value=1.0, off_value=0.0)
      
      for name, value in layers_dict.items():
          if 'guidance' in name:
              # value = tf.nn.sigmoid(value)
              value = tf.compat.v2.image.resize(value, [ny, nx])
              
              guidance_loss += loss_utils(value, ys, cost_name=loss_dict[common.GUIDANCE])

      
      # # Upsample logits in each stage
      # ny = samples[common.LABEL].get_shape()[1]
      # nx = samples[common.LABEL].get_shape()[2]
      # ys = tf.one_hot(
      #     samples[common.LABEL][...,0], 14, on_value=1.0, off_value=0.0)
      # ys = tf.reshape(ys, [-1,14])
      # ys = tf.cast(ys, tf.float32)
      # for name, value in layers_dict.items():
      #     if 'guidance' in name:
      #         value = tf.nn.sigmoid(value)
      #         value = tf.compat.v2.image.resize(value, [ny, nx])
              
      #         value = tf.reshape(value, [-1,14])
      #         loss = -tf.reduce_mean(ys * tf.log(tf.clip_by_value(value,1e-10,1.0)))
      #         guidance_loss += loss
              
              
      #         # guidance_loss += loss_utils(value, samples[common.LABEL], cost_name=loss_dict[common.OUTPUT_TYPE])
              
      # # Downsample Ground Truth       
      # for name, value in layers_dict.items():
      #     if 'guidance' in name:
      #         ny_g = value.get_shape()[1]
      #         nx_g = value.get_shape()[2]
      #         ys = tf.compat.v2.image.resize(samples[common.LABEL], [ny_g, nx_g])
      #         ys = tf.cast(ys, tf.int32)
      #         guidance_loss += loss_utils(value, ys, cost_name=loss_dict[common.OUTPUT_TYPE])
      
    
      guidance_loss = tf.multiply(guidance_loss_decay, 
                                  guidance_loss, 
                                  name='/'.join(['guidance_loss', loss_dict[common.GUIDANCE]]))

      losses.append(guidance_loss)  
      
    if transform_loss_decay is not None:
      ys2 = tf.compat.v2.image.resize(ys, [32,32])
      transform_loss = loss_utils(output_dict[common.GUIDANCE], ys2, cost_name=loss_dict[common.GUIDANCE])
      transform_loss = tf.multiply(transform_loss_decay, 
                                transform_loss, 
                                name='/'.join(['transform_loss', loss_dict[common.GUIDANCE]]))                             
      losses.append(transform_loss)
    
    # if common.PRIOR_IMGS in loss_dict:
    #   # TODO: do in the correct way
    #   transformed_loss = loss_utils(samples[common.IMAGE], output_dict[common.PRIOR_IMGS], 
    #                                 cost_name=loss_dict[common.PRIOR_IMGS])
    #   # image_gradient  = tf.image.image_gradients(output_dict[common.PRIOR_IMGS])
    #   # transformed_loss += image_gradient
    #   transformed_loss = tf.multiply(transformed_loss_decay, 
    #                               transformed_loss, 
    #                               name='/'.join(['transformed_loss', loss_dict[common.PRIOR_IMGS]]))
      # losses.append(transformed_loss)
      
    return losses

def get_files_name(path, data_suffix='*.jpg'):
    subject = glob.glob(path + data_suffix)
    if not subject:
        raise IOError("No such file data suffix exist")
    subject.sort()
    return subject


def load_nibabel_data(path, num_of_class=None, processing_list=None, onehot_label=False):
    # get file list
    subject = get_files_name(path, data_suffix='*.nii.gz')
    
    # select processing subject by subject_list
    if processing_list is not None:
        subject = [subject[s] for s in processing_list]
    
    # preprocessing
    imgs = []
    for i in subject:
        sample = nib.load(i).get_data()
        sample = np.flip(np.swapaxes(sample, 0, -1), 1)
        if onehot_label:
            if num_of_class is None:
                raise ValueError('TODO!!')
            sample = np.eye(num_of_class)[sample]
            sample = np.uint8(sample)
        imgs.append(sample)
    return imgs


def _div_maybe_zero(total_loss, num_present):
    """Normalizes the total loss with the number of present pixels."""
    return tf.to_float(num_present > 0) * tf.math.divide(
        total_loss,
        tf.maximum(1e-5, num_present))


def colorize(value, vmin=None, vmax=None, cmap=None):
    """
    A utility function for TensorFlow that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: 'gray')
    Example usage:
    ```
    output = tf.random_uniform(shape=[256, 256, 1])
    output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='viridis')
    tf.summary.image('output', output_color)
    ```
    
    Returns a 3D tensor of shape [height, width, 3].
    """

    # normalize
    vmin = tf.reduce_min(value) if vmin is None else vmin
    vmax = tf.reduce_max(value) if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin) # vmin..vmax

    # squeeze last dim if it exists
    value = tf.squeeze(value, axis=-1)

    # quantize
    indices = tf.to_int32(tf.round(value * 255))

    # gather
    cm = plt.cm.get_cmap(cmap if cmap is not None else 'gray')
    colors = tf.constant(cm.colors, dtype=tf.float32)
    value = tf.gather(colors, indices)
  
    value = tf.cast(value*255, tf.uint8)
    return value


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

  # TODO:
  scales_to_logits = {"1": scales_to_logits}
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


def get_model_init_fn(train_logdir,
                      tf_initial_checkpoint,
                      initialize_last_layer,
                      last_layers=None,
                      ignore_missing_vars=False):
  """Gets the function initializing model variables from a checkpoint.
  Args:
    train_logdir: Log directory for training.
    tf_initial_checkpoint: TensorFlow checkpoint for initialization.
    initialize_last_layer: Initialize last layer or not.
    last_layers: Last layers of the model.
    ignore_missing_vars: Ignore missing variables in the checkpoint.
  Returns:
    Initialization function.
  """
  if tf_initial_checkpoint is None:
    tf.logging.info('Not initializing the model from a checkpoint.')
    return None

  if tf.train.latest_checkpoint(train_logdir):
    tf.logging.info('Ignoring initialization; other checkpoint exists')
    return None

  tf.logging.info('Initializing model from path: %s', tf_initial_checkpoint)

  # Variables that will not be restored.
  exclude_list = ['global_step']
  if not initialize_last_layer:
    exclude_list.extend(last_layers)

  variables_to_restore = contrib_framework.get_variables_to_restore(
      exclude=exclude_list)

  if variables_to_restore:
    init_op, init_feed_dict = contrib_framework.assign_from_checkpoint(
        tf_initial_checkpoint,
        variables_to_restore,
        ignore_missing_vars=ignore_missing_vars)
    
    global_step = tf.train.get_or_create_global_step()

    def restore_fn(scaffold, sess):
      sess.run(init_op, init_feed_dict)
      sess.run([global_step])

    return restore_fn

  return None


def get_model_gradient_multipliers(last_layers, last_layer_gradient_multiplier):
  """Gets the gradient multipliers.
  The gradient multipliers will adjust the learning rates for model
  variables. For the task of semantic segmentation, the models are
  usually fine-tuned from the models trained on the task of image
  classification. To fine-tune the models, we usually set larger (e.g.,
  10 times larger) learning rate for the parameters of last layer.
  Args:
    last_layers: Scopes of last layers.
    last_layer_gradient_multiplier: The gradient multiplier for last layers.
  Returns:
    The gradient multiplier map with variables as key, and multipliers as value.
  """
  gradient_multipliers = {}

  for var in tf.model_variables():
    # Double the learning rate for biases.
    if 'biases' in var.op.name:
      gradient_multipliers[var.op.name] = 2.

    # Use larger learning rate for last layer variables.
    for layer in last_layers:
      if layer in var.op.name and 'biases' in var.op.name:
        gradient_multipliers[var.op.name] = 2 * last_layer_gradient_multiplier
        break
      elif layer in var.op.name:
        gradient_multipliers[var.op.name] = last_layer_gradient_multiplier
        break

  return gradient_multipliers


def get_model_learning_rate(learning_policy,
                            base_learning_rate,
                            learning_rate_decay_step,
                            learning_rate_decay_factor,
                            training_number_of_steps,
                            learning_power,
                            slow_start_step,
                            slow_start_learning_rate,
                            slow_start_burnin_type='none',
                            decay_steps=0.0,
                            end_learning_rate=1e-5,
                            boundaries=None,
                            boundary_learning_rates=None):
  """Gets model's learning rate.
  Computes the model's learning rate for different learning policy.
  Right now, only "step" and "poly" are supported.
  (1) The learning policy for "step" is computed as follows:
    current_learning_rate = base_learning_rate *
      learning_rate_decay_factor ^ (global_step / learning_rate_decay_step)
  See tf.train.exponential_decay for details.
  (2) The learning policy for "poly" is computed as follows:
    current_learning_rate = base_learning_rate *
      (1 - global_step / training_number_of_steps) ^ learning_power
  Args:
    learning_policy: Learning rate policy for training.
    base_learning_rate: The base learning rate for model training.
    learning_rate_decay_step: Decay the base learning rate at a fixed step.
    learning_rate_decay_factor: The rate to decay the base learning rate.
    training_number_of_steps: Number of steps for training.
    learning_power: Power used for 'poly' learning policy.
    slow_start_step: Training model with small learning rate for the first
      few steps.
    slow_start_learning_rate: The learning rate employed during slow start.
    slow_start_burnin_type: The burnin type for the slow start stage. Can be
      `none` which means no burnin or `linear` which means the learning rate
      increases linearly from slow_start_learning_rate and reaches
      base_learning_rate after slow_start_steps.
    decay_steps: Float, `decay_steps` for polynomial learning rate.
    end_learning_rate: Float, `end_learning_rate` for polynomial learning rate.
    boundaries: A list of `Tensor`s or `int`s or `float`s with strictly
      increasing entries.
    boundary_learning_rates: A list of `Tensor`s or `float`s or `int`s that
      specifies the values for the intervals defined by `boundaries`. It should
      have one more element than `boundaries`, and all elements should have the
      same type.
  Returns:
    Learning rate for the specified learning policy.
  Raises:
    ValueError: If learning policy or slow start burnin type is not recognized.
    ValueError: If `boundaries` and `boundary_learning_rates` are not set for
      multi_steps learning rate decay.
  """
  global_step = tf.train.get_or_create_global_step()
  adjusted_global_step = tf.maximum(global_step - slow_start_step, 0)
  if decay_steps == 0.0:
    tf.logging.info('Setting decay_steps to total training steps.')
    decay_steps = training_number_of_steps - slow_start_step
  if learning_policy == 'step':
    learning_rate = tf.train.exponential_decay(
        base_learning_rate,
        adjusted_global_step,
        learning_rate_decay_step,
        learning_rate_decay_factor,
        staircase=True)
  elif learning_policy == 'poly':
    learning_rate = tf.train.polynomial_decay(
        base_learning_rate,
        adjusted_global_step,
        decay_steps=decay_steps,
        end_learning_rate=end_learning_rate,
        power=learning_power)
  elif learning_policy == 'cosine':
    learning_rate = tf.train.cosine_decay(
        base_learning_rate,
        adjusted_global_step,
        training_number_of_steps - slow_start_step)
  elif learning_policy == 'multi_steps':
    if boundaries is None or boundary_learning_rates is None:
      raise ValueError('Must set `boundaries` and `boundary_learning_rates` '
                       'for multi_steps learning rate decay.')
    learning_rate = tf.train.piecewise_constant_decay(
        adjusted_global_step,
        boundaries,
        boundary_learning_rates)
  else:
    raise ValueError('Unknown learning policy.')

  adjusted_slow_start_learning_rate = slow_start_learning_rate
  if slow_start_burnin_type == 'linear':
    # Do linear burnin. Increase linearly from slow_start_learning_rate and
    # reach base_learning_rate after (global_step >= slow_start_steps).
    adjusted_slow_start_learning_rate = (
        slow_start_learning_rate +
        (base_learning_rate - slow_start_learning_rate) *
        tf.to_float(global_step) / slow_start_step)
  elif slow_start_burnin_type != 'none':
    raise ValueError('Unknown burnin type.')

  # Employ small learning rate at the first few steps for warm start.
  return tf.where(global_step < slow_start_step,
                  adjusted_slow_start_learning_rate, learning_rate)