"""Prepares the data used for DeepLab training/evaluation."""
import tensorflow as tf
from core import features_extractor
from core import preprocess_utils


# The probability of flipping the images and labels
# left-right during training
_PROB_OF_FLIP = 0.5
_PROB_OF_ROT = 0.5

def preprocess_image_and_label_seq(image,
                               label,
                               prior_segs,
                               crop_height,
                               crop_width,
                               channel,
                               seq_length,
                               label_for_each_frame,
                               num_class=None,
                               HU_window=None,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               rotate_angle=None,
                               is_training=True,
                               model_variant=None,):
  """Preprocesses the image and label.
  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.
  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.
  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  # if (prior_num_slice is not None) != (prior_imgs is not None or prior_segs is not None):
  #   raise ValueError('prior_num_slice should exist when import prior and vice versa')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image
  original_label = label

  # sample prior if exist
  # TODO: sample problem (consider z gt)
  # TODO: check HU
  # data type and value convert
  if HU_window is not None:
    image = preprocess_utils.HU_to_pixelvalue(image, HU_window)
  processed_image = tf.cast(image, tf.float32)
  
  if label is not None:
    label = tf.cast(label, tf.int32)

  pp = {}
      
  # Resize image and label to the desired range.
  # TODO: interface for this func.
  if min_resize_value or max_resize_value:
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

    if prior_segs is not None:
      # prior_segs = tf.cast(prior_segs, tf.int32)
      prior_segs, _ = (
          preprocess_utils.resize_to_range(
              image=prior_segs,
              min_size=min_resize_value,
              max_size=max_resize_value,
              factor=resize_factor,
              align_corners=True))
      pp["a"] = prior_segs
      
  # Data augmentation by randomly scaling the inputs.
  if is_training:
    # TODO: indepedent scale parameters
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, scale)
    processed_image.set_shape([None, None, seq_length*channel])
    
    if prior_segs is not None:
      prior_segs = preprocess_utils.scale_image_data(prior_segs, scale)
      pp["b"] = prior_segs
      
  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  # image_shape = processed_image.get_shape().as_list()
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Pad image with mean pixel value.
  # TODO: check padding value
  # mean_pixel = tf.reshape(
  #     features_extractor.mean_pixel(model_variant), [1, 1, 3])
  mean_pixel = 0.0
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, 0)
    
  if prior_segs is not None:
    prior_segs = preprocess_utils.pad_to_bounding_box(
      prior_segs, 0, 0, target_height, target_width, 0)
    pp["c"] = prior_segs
    
  # Randomly crop the image and label.
  # TODO: Do it in the right way
  # TODO: offset_height, offset_width for input
  # processed_image, label = preprocess_utils.random_crop(
  #         [processed_image, label], crop_height, crop_width)
  
  if is_training and label is not None:
    if prior_segs is not None:
      processed_image, label, prior_segs = preprocess_utils.random_crop(
        [processed_image, label, prior_segs], crop_height, crop_width)
    else:
      processed_image, label = preprocess_utils.random_crop(
        [processed_image, label], crop_height, crop_width)
    
  processed_image.set_shape([crop_height, crop_width, seq_length*channel])
  if label is not None:
    if label_for_each_frame:
      label.set_shape([crop_height, crop_width, seq_length*channel])
    else:
      label.set_shape([crop_height, crop_width, 1])
  if prior_segs is not None:
    prior_segs = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(prior_segs,axis=0), [crop_height,crop_width]), axis=0)
    prior_segs.set_shape([crop_height,crop_width,num_class])
    
  if is_training:
    # Randomly left-right flip the image and label.
    if prior_segs is not None:
      processed_image, label, prior_segs, _ = preprocess_utils.flip_dim(
        [processed_image, label, prior_segs], _PROB_OF_FLIP, dim=1)
    else:
      processed_image, label, _ = preprocess_utils.flip_dim(
        [processed_image, label], _PROB_OF_FLIP, dim=1)     
      
    # TODO: coplete random rotate method
    # Randomly rotate the image and label.
    if rotate_angle is not None:
        pass
        # processed_image, label, _ = preprocess_utils.random_rotate([processed_image, label], _PROB_OF_ROT, rotate_angle)

  return original_image, processed_image, label, original_label, pp, prior_segs



def preprocess_image_and_label(image,
                               label,
                               depth,
                               prior_imgs,
                               prior_segs,
                               num_slices,
                               crop_height,
                               crop_width,
                               z_label_method=None,
                               z_class=None,
                               HU_window=None,
                               min_resize_value=None,
                               max_resize_value=None,
                               resize_factor=None,
                               min_scale_factor=1.,
                               max_scale_factor=1.,
                               scale_factor_step_size=0,
                               ignore_label=255,
                               rotate_angle=None,
                               is_training=True,
                               model_variant=None,
                               prior_num_slice=None):
  """Preprocesses the image and label.
  Args:
    image: Input image.
    label: Ground truth annotation label.
    crop_height: The height value used to crop the image and label.
    crop_width: The width value used to crop the image and label.
    min_resize_value: Desired size of the smaller image side.
    max_resize_value: Maximum allowed size of the larger image side.
    resize_factor: Resized dimensions are multiple of factor plus one.
    min_scale_factor: Minimum scale factor value.
    max_scale_factor: Maximum scale factor value.
    scale_factor_step_size: The step size from min scale factor to max scale
      factor. The input is randomly scaled based on the value of
      (min_scale_factor, max_scale_factor, scale_factor_step_size).
    ignore_label: The label value which will be ignored for training and
      evaluation.
    is_training: If the preprocessing is used for training or not.
    model_variant: Model variant (string) for choosing how to mean-subtract the
      images. See feature_extractor.network_map for supported model variants.
  Returns:
    original_image: Original image (could be resized).
    processed_image: Preprocessed image.
    label: Preprocessed ground truth segmentation label.
  Raises:
    ValueError: Ground truth label not provided during training.
  """
  if is_training and label is None:
    raise ValueError('During training, label must be provided.')
  # if (prior_num_slice is not None) != (prior_imgs is not None or prior_segs is not None):
  #   raise ValueError('prior_num_slice should exist when import prior and vice versa')
  if model_variant is None:
    tf.logging.warning('Default mean-subtraction is performed. Please specify '
                       'a model_variant. See feature_extractor.network_map for '
                       'supported model variants.')

  # Keep reference to original image.
  original_image = image
  original_label = label

  # sample prior if exist
  # TODO: sample problem (consider z gt)
  
  # data type and value convert
  if HU_window is not None:
    image = preprocess_utils.HU_to_pixelvalue(image, HU_window)
    if prior_imgs is not None:
      prior_imgs = preprocess_utils.HU_to_pixelvalue(prior_imgs, HU_window)
  processed_image = tf.cast(image, tf.float32)
  
  if prior_imgs is not None:
    prior_imgs = tf.cast(prior_imgs, tf.float32)
    prior_num_slices = prior_imgs.get_shape().as_list()[2]
  
  if label is not None:
    label = tf.cast(label, tf.int32)

  pp = {}
  if min_resize_value or max_resize_value:
    if prior_segs is not None:
      # prior_segs = tf.cast(prior_segs, tf.int32)
      prior_segs, _ = (
          preprocess_utils.resize_to_range(
              image=prior_segs,
              min_size=min_resize_value,
              max_size=max_resize_value,
              factor=resize_factor,
              align_corners=True))
      pp["a"] = prior_segs
      
  # Resize image and label to the desired range.
  # TODO: interface for this func.
  if min_resize_value or max_resize_value:
    # processed_image = tf.expand_dims(processed_image, axis=0)
    # processed_image = tf.image.resize_bilinear(processed_image, [256,256])
    # processed_image = tf.squeeze(processed_image, axis=0)
    # label = tf.expand_dims(label, axis=0)
    # label = tf.image.resize_nearest_neighbor(label, [256,256])
    # label = tf.squeeze(label, axis=0)
    [processed_image, label] = (
        preprocess_utils.resize_to_range(
            image=processed_image,
            label=label,
            min_size=min_resize_value,
            max_size=max_resize_value,
            factor=resize_factor,
            align_corners=True))
    # The `original_image` becomes the resized image.
    original_image = tf.identity(processed_image)

  # Data augmentation by randomly scaling the inputs.
  if is_training:
    # TODO: indepedent scale parameters
    scale = preprocess_utils.get_random_scale(
        min_scale_factor, max_scale_factor, scale_factor_step_size)
    processed_image, label, prior_imgs, prior_segs = preprocess_utils.randomly_scale_image_and_label(
        processed_image, label, prior_imgs, prior_segs, scale)
    processed_image.set_shape([None, None, 1])
    pp["b"] = prior_segs
  # Pad image and label to have dimensions >= [crop_height, crop_width]
  image_shape = tf.shape(processed_image)
  # image_shape = processed_image.get_shape().as_list()
  image_height = image_shape[0]
  image_width = image_shape[1]

  target_height = image_height + tf.maximum(crop_height - image_height, 0)
  target_width = image_width + tf.maximum(crop_width - image_width, 0)

  # Pad image with mean pixel value.
  # TODO: check padding value
  # mean_pixel = tf.reshape(
  #     features_extractor.mean_pixel(model_variant), [1, 1, 3])
  mean_pixel = 0.0
  processed_image = preprocess_utils.pad_to_bounding_box(
      processed_image, 0, 0, target_height, target_width, mean_pixel)

  if label is not None:
    label = preprocess_utils.pad_to_bounding_box(
        label, 0, 0, target_height, target_width, 0)

  # TODO: [?,?,147] --> [?,?,?]
  if prior_imgs is not None:
    prior_imgs = preprocess_utils.pad_to_bounding_box(
      prior_imgs, 0, 0, target_height, target_width, mean_pixel)
    
  if prior_segs is not None:
    prior_segs = preprocess_utils.pad_to_bounding_box(
      prior_segs, 0, 0, target_height, target_width, 0)
    pp["c"] = prior_segs
    
  # Randomly crop the image and label.
  # TODO: Do it in the right way
  # TODO: offset_height, offset_width for input
  # processed_image, label = preprocess_utils.random_crop(
  #         [processed_image, label], crop_height, crop_width)
  if is_training and label is not None:
    if prior_imgs is not None and prior_segs is not None:
      processed_image, label, prior_imgs, prior_segs = preprocess_utils.random_crop(
          [processed_image, label, prior_imgs, prior_segs], crop_height, crop_width)
    if prior_imgs is not None and prior_segs is None:
      processed_image, label, prior_imgs = preprocess_utils.random_crop(
          [processed_image, label, prior_imgs], crop_height, crop_width)
    if prior_imgs is None and prior_segs is not None:
      processed_image, label, prior_segs = preprocess_utils.random_crop(
          [processed_image, label, prior_segs], crop_height, crop_width)
    if prior_imgs is None and prior_segs is None:
      processed_image, label = preprocess_utils.random_crop(
          [processed_image, label], crop_height, crop_width)
                

  processed_image.set_shape([crop_height, crop_width, 1])

  if label is not None:
    label.set_shape([crop_height, crop_width, 1])

  if prior_imgs is not None:
    prior_imgs.set_shape([crop_height, crop_width, prior_num_slices])
    
  if prior_segs is not None:
    
    prior_segs = tf.squeeze(tf.image.resize_bilinear(tf.expand_dims(prior_segs,axis=0), [crop_height,crop_width]), axis=0)
    prior_segs.set_shape([crop_height,crop_width,14])
    
  if is_training:
    # Randomly left-right flip the image and label.
    # TODO: Do it in the right way
    # TODO: input isfliped
    if prior_imgs is not None and prior_segs is not None:
      processed_image, label, prior_imgs, prior_segs, _ = preprocess_utils.flip_dim(
          [processed_image, label, prior_imgs, prior_segs], _PROB_OF_FLIP, dim=1)
    if prior_imgs is None and prior_segs is not None:
      processed_image, label, prior_segs, _ = preprocess_utils.flip_dim(
          [processed_image, label, prior_segs], _PROB_OF_FLIP, dim=1)
    if prior_imgs is not None and prior_segs is None:
      processed_image, label, prior_imgs, _ = preprocess_utils.flip_dim(
          [processed_image, label, prior_imgs], _PROB_OF_FLIP, dim=1)
    if prior_imgs is None and prior_segs is None:
      processed_image, label, _ = preprocess_utils.flip_dim(
          [processed_image, label], _PROB_OF_FLIP, dim=1)
          
    # TODO: coplete random rotate method
    # Randomly rotate the image and label.
    if rotate_angle is not None:
        pass
        # processed_image, label, _ = preprocess_utils.random_rotate([processed_image, label], _PROB_OF_ROT, rotate_angle)

  if z_label_method is not None:
      z_label = preprocess_utils.get_z_label(method=z_label_method, num_slices=num_slices, depth=depth, z_class=z_class) 
  else:
      z_label = None
  return original_image, processed_image, label, original_label, z_label, pp, prior_segs

