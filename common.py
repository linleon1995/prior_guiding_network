
import copy
import collections
import argparse
import tensorflow as tf


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


    