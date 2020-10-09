
import copy
import collections
import argparse
import tensorflow as tf

# TODO: complete parameters which seldom using

parser = argparse.ArgumentParser()

parser.add_argument('--fine_tune_batch_norm', type=bool, default=True,
                    help='')

parser.add_argument('--model_variant', type=str, default='resnet_v1_50_beta',
                    help='')

parser.add_argument('--batch_norm_decay', type=float, default=0.9997,
                    help='')

# Provide three different decoder types, including 'refinement_network', 'unet_structure', 'upsample'
parser.add_argument('--decoder_type', type=str, default='refine',
                    help='')

FLAGS, unparsed = parser.parse_known_args()

# The path for saving tensorflow checkpoint and tensorboard event
LOGGING_PATH = '/home/user/DISK/data/Jing/model/Thesis/thesis_trained/'

# The path for dataset directory. Each directory should contain raw data, 
# and tfrrecord or prior if the converting process is run
BASE_DATA_DIR = "/home/user/DISK/data/Jing/data/"

# Semantic segmentation item names.
LABELS_CLASS = 'labels_class'
IMAGE = 'image'
HEIGHT = 'height'
WIDTH = 'width'
IMAGE_NAME = 'image_name'
LABEL = 'label'
ORIGINAL_IMAGE = 'original_image'
NUM_SLICES = 'num_slices'
DEPTH = 'depth'
OUTPUT_TYPE = 'semantic'
OUTPUT_Z = 'z_pred'
GUIDANCE = 'guidance'
Z_LABEL = 'z_label'
PRIOR_IMGS = 'prior_imgs'
PRIOR_SEGS = 'prior_segs'
MULTI_GRID = None

class ModelOptions(
    collections.namedtuple('ModelOptions', [
        'outputs_to_num_classes',
        'crop_size',
        'output_stride',
        'preprocessed_images_dtype',
        'multi_grid',
        'model_variant',
        'fine_tune_batch_norm',
        'batch_norm_decay',
        'decoder_type'
    ])):
  """Immutable class to hold model options."""

  __slots__ = ()

  def __new__(cls,
              outputs_to_num_classes,
              crop_size=None,
              output_stride=8,
              preprocessed_images_dtype=tf.float32):
    """Constructor to set default values.
    Args:
      outputs_to_num_classes: A dictionary from output type to the number of
        classes. For example, for the task of semantic segmentation with 21
        semantic classes, we would have outputs_to_num_classes['semantic'] = 21.
      crop_size: A tuple [crop_height, crop_width].
      atrous_rates: A list of atrous convolution rates for ASPP.
      output_stride: The ratio of input to output spatial resolution.
      preprocessed_images_dtype: The type after the preprocessing function.
    Returns:
      A new ModelOptions instance.
    """
    # dense_prediction_cell_config = None
    # if FLAGS.dense_prediction_cell_json:
    #   with tf.gfile.Open(FLAGS.dense_prediction_cell_json, 'r') as f:
    #     dense_prediction_cell_config = json.load(f)
    # decoder_output_stride = None
    # if FLAGS.decoder_output_stride:
    #   decoder_output_stride = [
    #       int(x) for x in FLAGS.decoder_output_stride]
    #   if sorted(decoder_output_stride, reverse=True) != decoder_output_stride:
    #     raise ValueError('Decoder output stride need to be sorted in the '
    #                      'descending order.')
    # image_pooling_crop_size = None
    # if FLAGS.image_pooling_crop_size:
    #   image_pooling_crop_size = [int(x) for x in FLAGS.image_pooling_crop_size]
    # image_pooling_stride = [1, 1]
    # if FLAGS.image_pooling_stride:
    #   image_pooling_stride = [int(x) for x in FLAGS.image_pooling_stride]
    # label_weights = FLAGS.label_weights
    # if label_weights is None:
    #   label_weights = 1.0
    # nas_architecture_options = {
    #     'nas_stem_output_num_conv_filters': (
    #         FLAGS.nas_stem_output_num_conv_filters),
    #     'nas_use_classification_head': FLAGS.nas_use_classification_head,
    #     'nas_remove_os32_stride': FLAGS.nas_remove_os32_stride,
    # }
    return super(ModelOptions, cls).__new__(
        cls, outputs_to_num_classes, crop_size, output_stride, preprocessed_images_dtype,
        MULTI_GRID,
        FLAGS.model_variant,
        FLAGS.fine_tune_batch_norm,
        FLAGS.batch_norm_decay,
        FLAGS.decoder_type)
    
    def __deepcopy__(self, memo):
        return ModelOptions(copy.deepcopy(self.outputs_to_num_classes),
                            self.crop_size,
                            self.output_stride,
                            self.preprocessed_images_dtype)


    