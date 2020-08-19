import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import nibabel as nib
import argparse
import six
import time

import model
import common
import eval
import experiments
from model import voxelmorph
from datasets import data_generator
from utils import train_utils, eval_utils
from core import features_extractor
import input_preprocess
from tensorflow.python.ops import math_ops
import math
colorize = train_utils.colorize
spatial_transfom_exp = experiments.spatial_transfom_exp

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

PRIOR_PATH = '/home/user/DISK/data/Jing/model/Thesis/priors/'
LOGGING_PATH = '/home/user/DISK/data/Jing/model/Thesis/thesis_trained/'
PRETRAINED_PATH = '/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_019/model.ckpt-200000'
PRETRAINED_PATH = None
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/pretrained_weight/resnet/resnet_v1_50/model.ckpt'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_042/model.ckpt-40000'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-50000'
# DATASET_DIR = '/home/user/DISK/data/Jing/data/Training/tfrecord/'
# DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord_seq/'

FUSIONS = 5*["sum"]
FUSIONS = ["concat"] + 4*["guid_uni"]
FUSIONS = 5*["guid_uni"]

#TRAIN_SPLIT = ["train"]
#SEG_WEIGHT = 1.0

#DATASET_NAME = ['2013_MICCAI_Abdominal']
#DATASET_NAME = ['2019_ISBI_CHAOS_MR_T1', '2019_ISBI_CHAOS_MR_T2']
#DATASET_NAME = ['2019_ISBI_CHAOS_CT']

# TODO: shouldn't just select the first dataset pre_crop_size
#DATA_INFO = data_generator._DATASETS_INFORMATION[DATASET_NAME[0]]

# TODO: tf argparse
# TODO: dropout
# TODO: Multi-Scale Training
# TODO: flags.DEFINE_multi_integer
# TODO: Solve Warning in new tensorflow version
# TODO: tf.gather problem


def create_training_path(train_logdir):
    # TODO: Check whether empty of last folder before creating new one
    idx = 0
    path = os.path.join(train_logdir, "run_{:03d}".format(idx))
    while os.path.exists(path):
        # if len(os.listdir(path)) == 0:
        #     break
        idx += 1
        path = os.path.join(train_logdir, "run_{:03d}".format(idx))

    os.makedirs(path)
    return path



parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name', nargs='+', required=True,
                    help='')

parser.add_argument('--train_split', nargs='+', required=True,
                    help='')

parser.add_argument('--seq_length', type=int, default=3,
                    help='')

parser.add_argument('--cell_type', type=str, default="ConvGRU",
                    help='')

parser.add_argument('--guid_fuse', type=str, default="sum_wo_back",
                    help='')

parser.add_argument('--guid_feature_only', type=bool, default=False,
                    help='')

parser.add_argument('--stage_pred_ks', type=int, default=1,
                    help='')

parser.add_argument('--add_feature', type=bool, default=True,
                    help='')

parser.add_argument('--apply_sram2', type=bool, default=True,
                    help='')

parser.add_argument('--fuse_flag', type=bool, default=True,
                    help='')

parser.add_argument('--predict_without_background', type=bool, default=False,
                    help='')

parser.add_argument('--guid_encoder', type=str, default="early",
                    help='')

parser.add_argument('--out_node', type=int, default=32,
                    help='')

parser.add_argument('--guid_conv_type', type=str, default="conv",
                    help='')

parser.add_argument('--guid_conv_nums', type=int, default=2,
                    help='')

parser.add_argument('--share', type=bool, default=True,
                    help='')

parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='')

# Training configuration
parser.add_argument('--train_logdir', type=str, default=create_training_path(LOGGING_PATH),
                    help='')

parser.add_argument('--batch_size', type=int, default=16,
                    help='')

parser.add_argument('--tf_initial_checkpoint', type=str, default=PRETRAINED_PATH,
                    help='')

parser.add_argument('--initialize_last_layer', type=bool, default=True,
                    help='')

parser.add_argument('--training_number_of_steps', type=int, default=30000,
                    help='')

parser.add_argument('--profile_logdir', type=str, default='',
                    help='')

parser.add_argument('--master', type=str, default='',
                    help='')

parser.add_argument('--task', type=int, default=0,
                    help='')

parser.add_argument('--log_steps', type=int, default=10,
                    help='')

parser.add_argument('--save_summaries_secs', type=int, default=None,
                    help='')

parser.add_argument('--save_summaries_images', type=bool, default=True,
                    help='')

parser.add_argument('--save_checkpoint_steps', type=int, default=1000,
                    help='')

parser.add_argument('--validation_steps', type=int, default=1000,
                    help='')

parser.add_argument('--num_ps_tasks', type=int, default=0,
                    help='')

parser.add_argument('--last_layers_contain_logits_only', type=bool, default=True,
                    help='')

parser.add_argument('--drop_prob', type=float, default=None,
                    help='')

# Model configuration
parser.add_argument('--model_variant', type=str, default=None,
                    help='')

parser.add_argument('--z_model', type=str, default=None,
                    help='')

parser.add_argument('--z_label_method', type=str, default=None,
                    help='')

parser.add_argument('--z_class', type=int, default=None,
                    help='')
# Input prior could be "zeros", "ground_truth", "training_data_fusion" (fixed)
# , "adaptive" witch means decide adaptively by learning parameters or "come_from_featrue"
# TODO: Check conflict of guidance
parser.add_argument('--guidance_type', type=str, default="training_data_fusion",
                    help='')

parser.add_argument('--prior_num_slice', type=int, default=1,
                    help='')

parser.add_argument('--prior_num_subject', type=int, default=16,
                    help='')

parser.add_argument('--fusion_slice', type=float, default=3,
                    help='')

parser.add_argument('--z_loss_decay', type=float, default=None,
                    help='')

parser.add_argument('--stage_pred_loss', type=bool, default=True,
                    help='')

parser.add_argument('--guidance_loss', type=bool, default=True,
                    help='')

parser.add_argument('--regularization_weight', type=float, default=None,
                    help='')

parser.add_argument('--seg_loss_name', type=str, default="softmax_dice_loss",
                    help='')

parser.add_argument('--guid_loss_name', type=str, default="sigmoid_cross_entropy",
                    help='')

parser.add_argument('--stage_pred_loss_name', type=str, default="sigmoid_cross_entropy",
                    help='')


# Learning rate configuration
parser.add_argument('--learning_policy', type=str, default='poly',
                    help='')

parser.add_argument('--base_learning_rate', type=float, default=7.5e-3,
                    help='')

parser.add_argument('--learning_rate_decay_step', type=float, default=0,
                    help='')

parser.add_argument('--learning_rate_decay_factor', type=float, default=1e-1,
                    help='')

parser.add_argument('--learning_power', type=float, default=0.9,
                    help='')

parser.add_argument('--slow_start_step', type=int, default=0,
                    help='')

parser.add_argument('--slow_start_learning_rate', type=float, default=1e-4,
                    help='')

parser.add_argument('--momentum', type=float, default=0.9,
                    help='')

# Dataset settings.
# '2019_ISBI_CHAOS_CT', '2019_ISBI_CHAOS_MR', '2013_MICCAI_Abdominal'
# parser.add_argument('--dataset', type=str, default=DATASET_NAME,
#                     help='')

# parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
#                     help='')

parser.add_argument('--output_stride', type=int, default=None,
                    help='')

parser.add_argument('--num_clones', type=int, default=1,
                    help='')

# parser.add_argument('--crop_size', type=int, default=256,
#                     help='')

parser.add_argument('--min_scale_factor', type=float, default=0.625,
                    help='')

parser.add_argument('--max_scale_factor', type=float, default=1.25,
                    help='')

parser.add_argument('--scale_factor_step_size', type=float, default=0.125,
                    help='')

parser.add_argument('--min_resize_value', type=int, default=None,
                    help='')

parser.add_argument('--max_resize_value', type=int, default=None,
                    help='')

parser.add_argument('--resize_factor', type=int, default=None,
                    help='')

parser.add_argument('--pre_crop_flag', type=bool, default=True,
                    help='')

def check_model_conflict(model_options):
  pass
#     if not model_options.decoder_type == "refinement_network":
#         assert FLAGS.guida

    # if FLAGS.deformable_transform:
    #     assert common.PRIOR_SEGS in dataset and common.PRIOR_IMGS in dataset


class DSHandleHook(tf.train.SessionRunHook):
    def __init__(self, train_str, valid_str):
        self.train_str = train_str
        self.valid_str = valid_str
        self.train_handle = None
        self.valid_handle = None

    def after_create_session(self, session, coord):
        del coord
        if self.train_str is not None:
            self.train_handle, self.valid_handle = session.run([self.train_str,
                                                                self.valid_str])
        print('session run ds string-handle done....')

def _build_network(samples, outputs_to_num_classes, model_options, ignore_label, is_training):
  """Builds a clone of DeepLab.
  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    outputs_to_num_classes: A map from output type to the number of classes. For
      example, for the task of semantic segmentation with 21 semantic classes,
      we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.
  """

  # Add name to input and label nodes so we can add to summary.
  print(60*"SAMPLES", samples)
  samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

  summary_img = samples[common.IMAGE]
  summary_label = samples[common.LABEL]
  if FLAGS.seq_length > 1:
    summary_img = summary_img[:,1]
    summary_label = summary_label[:,1]

  if 'prior_slices' in samples:
    prior_slices = samples['prior_slices']
  else:
    prior_slices = None

  # if common.PRIOR_IMGS in samples:
  #   samples[common.PRIOR_IMGS] = tf.identity(
  #       samples[common.PRIOR_IMGS], name=common.PRIOR_IMGS)
  # else:
  #   samples[common.PRIOR_IMGS] = None

  if common.PRIOR_SEGS in samples:
    samples[common.PRIOR_SEGS] = tf.identity(
        samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
  else:
    samples[common.PRIOR_SEGS] = None

  # if common.GUIDANCE in samples:
  #   samples[common.GUIDANCE] = tf.identity(samples[common.GUIDANCE], name=common.GUIDANCE)
  # else:
  #   samples[common.GUIDANCE] = None

  # TODO: moving_segs
  clone_batch_size = FLAGS.batch_size // FLAGS.num_clones

  # translations = tf.random_uniform([], minval=-15,maxval=15,dtype=tf.float32)
  # angle = tf.random_uniform([], minval=-20,maxval=20,dtype=tf.float32)
  # angle = angle * math.pi / 180
  # labels = samples[common.LABEL]
  # samples[common.IMAGE] = spatial_transfom_exp(samples[common.IMAGE], angle,
  #                                             [translations,0], "BILINEAR")
  # samples[common.LABEL] = spatial_transfom_exp(samples[common.LABEL], angle,
  #                                             [translations,0], "NEAREST")


  num_class = outputs_to_num_classes['semantic']
  output_dict, layers_dict = model.pgb_network(
                samples[common.IMAGE],
                samples[common.HEIGHT],
                samples[common.WIDTH],
                model_options=model_options,
                # labels=samples[common.LABEL],
                # samples=samples["organ_label"],
                # prior_imgs=samples[common.PRIOR_IMGS],
                prior_segs=samples[common.PRIOR_SEGS],
                num_class=num_class,
                # num_slices=samples[common.NUM_SLICES],
                prior_slice=prior_slices,
                batch_size=clone_batch_size,
                guidance_type=FLAGS.guidance_type,
                fusion_slice=FLAGS.fusion_slice,
                # prior_dir=FLAGS.prior_dir,
                drop_prob=FLAGS.drop_prob,
                stn_in_each_class=True,
                # prior_num_slice=FLAGS.prior_num_slice,
                reuse=tf.AUTO_REUSE,
                is_training=is_training,
                weight_decay=FLAGS.weight_decay,
                # fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                share=FLAGS.share,
                fusions=FUSIONS,
                out_node=FLAGS.out_node,
                guid_encoder=FLAGS.guid_encoder,
                z_label_method=FLAGS.z_label_method,
                z_model=FLAGS.z_model,
                z_class=FLAGS.z_class,
                guidance_loss=FLAGS.guidance_loss,
                stage_pred_loss=FLAGS.stage_pred_loss,
                guidance_loss_name=FLAGS.guid_loss_name,
                stage_pred_loss_name=FLAGS.stage_pred_loss_name,
                guid_conv_nums=FLAGS.guid_conv_nums,
                guid_conv_type=FLAGS.guid_conv_type,
                fuse_flag=FLAGS.fuse_flag,
                predict_without_background=FLAGS.predict_without_background,
                apply_sram2=FLAGS.apply_sram2,
                add_feature=FLAGS.add_feature,
                guid_feature_only=FLAGS.guid_feature_only,
                stage_pred_ks=FLAGS.stage_pred_ks,
                guid_fuse=FLAGS.guid_fuse,
                seq_length=FLAGS.seq_length,
                cell_type=FLAGS.cell_type
                )

  # Add name to graph node so we can add to summary.
  output = output_dict[common.OUTPUT_TYPE]
  output = tf.identity(output, name=common.OUTPUT_TYPE)

  if common.Z_LABEL in samples:
    z_label = tf.identity(samples[common.Z_LABEL], name=common.Z_LABEL)
  else:
    z_label = None

  # if common.PRIOR_IMGS in output_dict:
  #   prior_img = output_dict[common.PRIOR_IMGS]
  # else:
  #   prior_img = None

  if common.PRIOR_SEGS in output_dict:
    prior_seg = output_dict[common.PRIOR_SEGS]
  else:
    prior_seg = None

  if common.OUTPUT_Z in output_dict:
    z_pred = output_dict[common.OUTPUT_Z]
  else:
    z_pred = None

  if 'original_guidance' in output_dict:
    guidance_original = output_dict['original_guidance']
  else:
    guidance_original = None

  # guidance_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'guidance' in dict_key}
  # if len(guidance_dict) == 0:
  #   guidance_dict = None

  # TODO validation phase can use log_summaries?
  # Log the summary
  _log_summaries(summary_img,
                 summary_label,
                 outputs_to_num_classes['semantic'],
                 output_dict[common.OUTPUT_TYPE],
                 z_label=z_label,
                 z_pred=z_pred,
                 prior_segs=prior_seg,
                 guidance=layers_dict,
                 guidance_original=guidance_original)
  return output_dict, layers_dict


def _tower_loss(iterator, num_of_classes, model_options, ignore_label, scope, reuse_variable):
    """Calculates the total loss on a single tower running the deeplab model.
    Args:
        iterator: An iterator of type tf.data.Iterator for images and labels.
        num_of_classes: Number of classes for the dataset.
        ignore_label: Ignore label for the dataset.
        scope: Unique prefix string identifying the deeplab tower.
        reuse_variable: If the variable should be reused.
    Returns:
        The total loss for a batch of data.
    """
    with tf.variable_scope(
        tf.get_variable_scope(), reuse=True if reuse_variable else None):
        samples = iterator
        output_dict, layers_dict = _build_network(samples, {common.OUTPUT_TYPE: num_of_classes},
                                                model_options, ignore_label, is_training=True)

    loss_dict = {}
    seg_weight = 1.0
    # guidance_loss_weight = FLAGS.guidance_loss
    # stage_pred_loss_weight = FLAGS.stage_pred_loss
    # stage_pred_loss_weight = [0.04] + 13*[1.0]
    # TODO: stage_pred_loss should be weight or flag
    # TODO: predict_w.o_bg case
    if FLAGS.stage_pred_loss:
      if FLAGS.stage_pred_loss_name == "softmax_generaled_dice_loss":
        num_label_pixels = tf.reduce_sum(tf.one_hot(
          samples[common.LABEL][...,0], num_of_classes, on_value=1.0, off_value=0.0), axis=[1,2])
        stage_pred_loss_weight = (tf.ones_like(num_label_pixels) + 1e-10) / (tf.pow(num_label_pixels, 2) + 1e-10)
      elif FLAGS.stage_pred_loss_name == "sigmoid_cross_entropy":
        num_label_pixels = tf.reduce_sum(tf.nn.sigmoid(output_dict[common.OUTPUT_TYPE]))
        stage_pred_loss_weight = (tf.ones_like(num_label_pixels) + 1e-10) / (num_label_pixels + 1e-10)
    else:
      stage_pred_loss_weight = 1.0
    if FLAGS.guidance_loss:
      guidance_loss_weight = stage_pred_loss_weight
    else:
      guidance_loss_weight = 1.0
    # seg_weight = train_utils.get_loss_weight(samples[common.LABEL], loss_name, loss_weight=SEG_WEIGHT_FLAG)

    loss_dict[common.OUTPUT_TYPE] = {"loss": FLAGS.seg_loss_name, "decay": None, "weights": seg_weight, "scope": "segmenation"}
    if FLAGS.guidance_loss:
      # guidance_loss_weight = train_utils.get_loss_weight(samples[common.LABEL], FLAGS.guid_loss_name,
      #                                                    decay=FLAGS.guidance_loss)
      loss_dict[common.GUIDANCE] = {"loss": FLAGS.guid_loss_name, "decay": None, "weights": guidance_loss_weight, "scope": "guidance"}
    if FLAGS.stage_pred_loss:
      # stage_pred_loss_weight = train_utils.get_loss_weight(samples[common.LABEL], FLAGS.stage_pred_loss_name,
      #                                                      decay=FLAGS.stage_pred_loss)
      loss_dict["stage_pred"] = {"loss": FLAGS.stage_pred_loss_name, "decay": None, "weights": stage_pred_loss_weight, "scope": "stage_pred"}

    if common.OUTPUT_Z in output_dict:
      loss_dict[common.OUTPUT_Z] = {"loss": "softmax_cross_entropy", "decay": None, "weights": 1.0, "scope": "z_pred"}
      z_class = FLAGS.z_class
    else:
      z_class = None

    train_utils.get_losses(output_dict,
                           layers_dict,
                           samples,
                           loss_dict,
                           num_of_classes,
                           FLAGS.seq_length,
                           FLAGS.batch_size,
                           predict_without_background=FLAGS.predict_without_background,
                           z_class=z_class)

    # if FLAGS.z_loss_decay is not None:
    #     if FLAGS.z_label_method.split("_")[1] == 'regression':
    #         loss_dict[common.OUTPUT_Z] = {"loss": "MSE", "decay": FLAGS.z_loss_decay}
    #     elif FLAGS.z_label_method.split("_")[1] == 'classification':
    #         loss_dict[common.OUTPUT_Z] = {"loss": "cross_entropy_zlabel", "decay": FLAGS.z_loss_decay}

    # if FLAGS.guidance_loss is not None:
    #     loss_dict[common.GUIDANCE] = {"loss": "mean_dice_coefficient", "decay": FLAGS.guidance_loss}

    # if FLAGS.transform_loss_decay is not None:
    #     loss_dict["transform"] = {"loss": "cross_entropy_sigmoid", "decay": FLAGS.transform_loss_decay}

    # clone_batch_size = FLAGS.batch_size // FLAGS.num_clones
    # losses = train_utils.get_losses(output_dict, layers_dict, samples,
    #                                 loss_dict=loss_dict,
    #                                 batch_size=clone_batch_size)
    # seg_loss = losses[0]

    losses = tf.compat.v1.losses.get_losses(scope=scope)
    seg_loss = losses[0]
    for loss in losses:
        tf.summary.scalar('Losses/%s' % loss.op.name, loss)

    if FLAGS.regularization_weight is not None:
        regularization_loss = tf.losses.get_regularization_loss(scope=scope)
        regularization_loss = FLAGS.regularization_weight * regularization_loss
        regularization_loss = tf.identity(regularization_loss, name='regularization_loss_with_decay')
        tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                            regularization_loss)
        total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
    else:
        total_loss = tf.add_n(losses)
    return total_loss, seg_loss


def _log_summaries(input_image, label, num_of_classes, output, z_pred, prior_segs,
                   guidance, guidance_original, **kwargs):
  """Logs the summaries for the model.
  Args:
    input_image: Input image of the model. Its shape is [batch_size, height,
      width, channel].
    label: Label of the image. Its shape is [batch_size, height, width].
    num_of_classes: The number of classes of the dataset.
    output: Output of the model. Its shape is [batch_size, height, width].
  """
  z_label = kwargs.pop("z_label", None)
  # Add summaries for model variables.
  for model_var in tf.model_variables():
    tf.summary.histogram(model_var.op.name, model_var)

  # Add summaries for images, labels, semantic predictions.
  if FLAGS.save_summaries_images:
    tf.summary.image('samples/%s' % common.IMAGE, colorize(input_image, cmap='viridis'))

    # Scale up summary image pixel values for better visualization.
    pixel_scaling = max(1, 255 // num_of_classes)

    summary_label = tf.cast(label * pixel_scaling, tf.uint8)
    tf.summary.image('samples/%s' % common.LABEL, colorize(summary_label, cmap='viridis'))

    predictions = tf.expand_dims(tf.argmax(output, 3), -1)
    summary_predictions = tf.cast(predictions * pixel_scaling, tf.uint8)
    tf.summary.image('samples/%s' % common.OUTPUT_TYPE, colorize(summary_predictions, cmap='viridis'))

  # TODO: parameterization
  # if guidance is not None:
  #   # tf.summary.image('reg_field/%s' % 'field_x', colorize(field[...,0:1], cmap='viridis'))
  #   # tf.summary.image('reg_field/%s' % 'field_y', colorize(field[...,1:2], cmap='viridis'))

  #   tf.summary.image('guidance/%s' % 'guidance1_6', colorize(guidance['guidance4'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance1_7', colorize(guidance['guidance4'][...,7:8], cmap='viridis'))

  #   tf.summary.image('guidance/%s' % 'guidance2_6', colorize(guidance['guidance3'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance2_7', colorize(guidance['guidance3'][...,7:8], cmap='viridis'))

  #   tf.summary.image('guidance/%s' % 'guidance3_6', colorize(guidance['guidance2'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance3_7', colorize(guidance['guidance2'][...,7:8], cmap='viridis'))

  #   tf.summary.image('guidance/%s' % 'guidance4_6', colorize(guidance['guidance1'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance4_7', colorize(guidance['guidance1'][...,7:8], cmap='viridis'))

  #   tf.summary.image('guidance/%s' % 'guidance5/logits_6', colorize(output[...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance5/logits_7', colorize(output[...,7:8], cmap='viridis'))

    guid_avg = tf.get_collection("guid_avg")
    tf.summary.image('guidance/%s' % 'guid_avg0', colorize(guid_avg[0][...,0:1], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guid_avg1', colorize(guid_avg[1][...,0:1], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guid_avg2', colorize(guid_avg[2][...,0:1], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guid_avg3', colorize(guid_avg[3][...,0:1], cmap='viridis'))
    # tf.summary.image('guidance/%s' % 'guid_avg4', colorize(guid_avg[4][...,0:1], cmap='viridis'))

  # # if z_label is not None and z_pred is not None:
  # #   clone_batch_size = FLAGS.batch_size // FLAGS.num_clones

  # #   z_label = tf.reshape(z_label, [1,clone_batch_size,1,1])
  # #   z_label = tf.tile(z_label, [1,1,clone_batch_size,1])
  # #   tf.summary.image('samples/%s' % 'z_label', colorize(z_label, cmap='viridis'))

  # #   z_pred = tf.reshape(z_pred, [1,clone_batch_size,1,1])
  # #   z_pred = tf.tile(z_pred, [1,1,clone_batch_size,1])
  # #   tf.summary.image('samples/%s' % 'z_pred', colorize(z_pred, cmap='viridis'))


def _average_gradients(tower_grads):
  """Calculates average of gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list is
      over individual gradients. The inner list is over the gradient calculation
      for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been summed
       across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads, variables = zip(*grad_and_vars)

    grad = tf.reduce_mean(tf.stack(grads, axis=0), axis=0)

    # All vars are of the same value, using the first tower here.
    average_grads.append((grad, variables[0]))

  return average_grads


# def _train_deeplab_model(iterator, num_of_classes, model_options, ignore_label, handle, reuse=None):
#   def train_loss(total_loss, total_seg_loss, tower_grads, global_step, learning_rate, optimizer):
#     """Trains the deeplab model.
#     Args:
#       iterator: An iterator of type tf.data.Iterator for images and labels.
#       num_of_classes: Number of classes for the dataset.
#       ignore_label: Ignore label for the dataset.
#     Returns:
#       train_tensor: A tensor to update the model variables.
#       summary_op: An operation to log the summaries.
#     """

#     # optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

#       # TODO: understand and modify
#       # # Modify the gradients for biases and last layer variables.
#       # last_layers = model.get_extra_layer_scopes(
#       #     FLAGS.last_layers_contain_logits_only)
#       # grad_mult = train_utils.get_model_gradient_multipliers(
#       #     last_layers, FLAGS.last_layer_gradient_multiplier)
#       # if grad_mult:
#       #   grads_and_vars = tf.contrib.training.multiply_gradients(
#       #       grads_and_vars, grad_mult)

#     with tf.device('/cpu:0'):
#         grads_and_vars = _average_gradients(tower_grads)

#         # Create gradient update op.
#         grad_updates = optimizer.apply_gradients(
#             grads_and_vars, global_step=global_step)

#         # Gather update_ops. These contain, for example,
#         # the updates for the batch_norm variables created by model_fn.
#         update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#         update_ops.append(grad_updates)
#         update_op = tf.group(*update_ops)

#         # total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
#         # total_loss = loss + tf.losses.get_regularization_loss
#         # Print total loss to the terminal.
#         # This implementation is mirrored from tf.slim.summaries.
#         should_log = tf.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)
#         total_loss = tf.cond(
#             should_log,
#             lambda: tf.Print(total_loss, [total_loss, total_seg_loss, global_step], 'Total loss, Segmentation loss and Global step:'),
#             lambda: total_loss)

#         with tf.control_dependencies([update_op]):
#           train_tensor = tf.identity(total_loss, name='train_op')
#     return train_tensor

#   def valid_loss(train_tensor):
#     return train_tensor

#   summaries = []
#   global_step = tf.train.get_or_create_global_step()
#   learning_rate = train_utils.get_model_learning_rate(
#       FLAGS.learning_policy, FLAGS.base_learning_rate,
#       FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
#       FLAGS.training_number_of_steps, FLAGS.learning_power,
#       FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)
#   optimizer = tf.train.AdamOptimizer(learning_rate)

#   total_loss, total_seg_loss = 0, 0
#   tower_grads = []
#   for i in range(FLAGS.num_clones):
#     with tf.device('/gpu:%d' % i):
#       with tf.name_scope('clone_%d' % i) as scope:
#         loss, seg_loss = _tower_loss(
#             iterator=iterator,
#             num_of_classes=num_of_classes,
#             model_options=model_options,
#             ignore_label=ignore_label,
#             scope=scope,
#             reuse_variable=(i != 0)
#             # reuse_variable=reuse
#             )
#         total_loss += loss
#         total_seg_loss += seg_loss

#         grads = optimizer.compute_gradients(loss)
#         tower_grads.append(grads)

#   tower_summaries = tf.summary.merge_all()
#   if tower_summaries is not None:
#       summaries.append(tower_summaries)



#   summaries.append(tf.summary.scalar('learning_rate', learning_rate))
#   summary_op = tf.summary.merge(summaries)
#   # handle = tf.Print(handle [handle], 'gg'),
#   train_tensor = tf.cond(tf.equal(handle.name, "train:0"),

#                          lambda: train_loss(total_loss, total_seg_loss, tower_grads, global_step, learning_rate, optimizer),
#                          lambda: valid_loss(total_loss),
#                         #  lambda: valid_loss(total_loss)
#                          )
#   train_tensor = tf.identity(train_tensor, name='train_op')
#   return train_tensor, summary_op


def _train_deeplab_model(iterator, num_of_classes, model_options, ignore_label, reuse=None):
  """Trains the deeplab model.
  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_of_classes: Number of classes for the dataset.
    ignore_label: Ignore label for the dataset.
  Returns:
    train_tensor: A tensor to update the model variables.
    summary_op: An operation to log the summaries.
  """
  global_step = tf.train.get_or_create_global_step()
  summaries = []

  learning_rate = train_utils.get_model_learning_rate(
      FLAGS.learning_policy, FLAGS.base_learning_rate,
      FLAGS.learning_rate_decay_step, FLAGS.learning_rate_decay_factor,
      FLAGS.training_number_of_steps, FLAGS.learning_power,
      FLAGS.slow_start_step, FLAGS.slow_start_learning_rate)

  optimizer = tf.train.AdamOptimizer(learning_rate)
  # optimizer = tf.train.MomentumOptimizer(learning_rate, FLAGS.momentum)

  tower_grads = []
  total_loss, total_seg_loss = 0, 0
  tower_summaries = None
  for i in range(FLAGS.num_clones):
    with tf.device('/gpu:%d' % i):
      with tf.name_scope('clone_%d' % i) as scope:
        loss, seg_loss = _tower_loss(
            iterator=iterator,
            num_of_classes=num_of_classes,
            model_options=model_options,
            ignore_label=ignore_label,
            scope=scope,
            reuse_variable=(i != 0)
            # reuse_variable=reuse
            )
        total_loss += loss
        total_seg_loss += seg_loss

        grads = optimizer.compute_gradients(loss)
        tower_grads.append(grads)

        # Retain the summaries from the first tower.
        # if not i:
          # TODO:
  tower_summaries = tf.summary.merge_all()
          # tower_summaries = tf.summary.merge_all(scope=scope)

  summaries.append(tf.summary.scalar('learning_rate', learning_rate))

  with tf.device('/cpu:0'):
    grads_and_vars = _average_gradients(tower_grads)
    if tower_summaries is not None:
      summaries.append(tower_summaries)

    # TODO: understand and modify
    # # Modify the gradients for biases and last layer variables.
    # last_layers = model.get_extra_layer_scopes(
    #     FLAGS.last_layers_contain_logits_only)
    # grad_mult = train_utils.get_model_gradient_multipliers(
    #     last_layers, FLAGS.last_layer_gradient_multiplier)
    # if grad_mult:
    #   grads_and_vars = tf.contrib.training.multiply_gradients(
    #       grads_and_vars, grad_mult)


    # Create gradient update op.
    grad_updates = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)

    # Gather update_ops. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)


    # update_op = tf.cond(tf.equal(handle.name, "train:0"), get_update_op, no_update_op)

    # total_loss = tf.losses.get_total_loss(add_regularization_losses=True)
    # total_loss = loss + tf.losses.get_regularization_loss
    # Print total loss to the terminal.
    # This implementation is mirrored from tf.slim.summaries.
    should_log = tf.equal(math_ops.mod(global_step, FLAGS.log_steps), 0)
    total_loss = tf.cond(
        should_log,
        lambda: tf.Print(total_loss, [total_loss, total_seg_loss, global_step], 'Total loss, Segmentation loss and Global step:'),
        lambda: total_loss)

    summaries.append(tf.summary.scalar('total_loss', total_loss))

    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')
    summary_op = tf.summary.merge(summaries)

  return train_tensor, summary_op


def _val_deeplab_model(iterator, num_of_classes, model_options, ignore_label, steps, reuse=None):
  """Trains the deeplab model.
  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    num_of_classes: Number of classes for the dataset.
    ignore_label: Ignore label for the dataset.
  Returns:
    train_tensor: A tensor to update the model variables.
    summary_op: An operation to log the summaries.
  """
  with tf.variable_scope(
      tf.get_variable_scope(), reuse=True):
      samples = iterator
      output_dict, layers_dict = _build_network(samples, {common.OUTPUT_TYPE: num_of_classes},
                                              model_options, ignore_label, is_training=False)


  logits = output_dict[common.OUTPUT_TYPE]
  # preds = tf.nn.softmax(logits)
  # predictions = tf.identity(preds, name=common.OUTPUT_TYPE)
  # predictions = tf.argmax(predictions, axis=3)
  # predictions = tf.cast(predictions, tf.int32)
  prediction = eval_utils.inference_segmentation(logits, dim=3)
  pred_flat = tf.reshape(prediction, shape=[-1,])

  # labels = tf.squeeze(samples[common.LABEL], axis=3)

  if FLAGS.seq_length > 1:
      label = samples[common.LABEL][:,FLAGS.seq_length//2]
  else:
      label = samples[common.LABEL]
  labels_flat = tf.reshape(label, shape=[-1,])
  print(60*"Q", samples)
  # print(samples, predictions, logits, 30*"s")
  # Define Confusion Maxtrix
  cm = tf.confusion_matrix(labels_flat, pred_flat, num_classes=num_of_classes)

  summary_op = 0
  return cm, summary_op


def main(unused_argv):
    # TODO: single data information --> multiple
    data_inforamtion = data_generator._DATASETS_INFORMATION[FLAGS.dataset_name[0]]
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    # tf.gfile.MakeDirs(FLAGS.train_logdir+"/train_envs/")
    # tf.gfile.MakeDirs(FLAGS.train_logdir+"/val_envs/")
    for split in FLAGS.train_split:
      tf.logging.info('Training on %s set', split)

    path = FLAGS.train_logdir
    parameters_dict = vars(FLAGS)
    with open(os.path.join(path, 'logging.txt'), 'w') as f:
        for key in parameters_dict:
            f.write( "{}: {}".format(str(key), str(parameters_dict[key])))
            f.write("\n")
        f.write("\nStart time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        f.write("\n")

    graph = tf.Graph()
    with graph.as_default():
        with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.num_ps_tasks)):
            assert FLAGS.batch_size % FLAGS.num_clones == 0, (
                'Training batch size not divisble by number of clones (GPUs).')
            clone_batch_size = FLAGS.batch_size // FLAGS.num_clones

            if '2019_ISBI_CHAOS_MR_T1' in FLAGS.dataset_name or '2019_ISBI_CHAOS_MR_T2' in FLAGS.dataset_name:
              min_resize_value = data_inforamtion.height
              max_resize_value = data_inforamtion.height
            else:
              if FLAGS.min_resize_value is not None:
                min_resize_value = FLAGS.min_resize_value
              else:
                min_resize_value = data_inforamtion.height

              if FLAGS.max_resize_value is not None:
                max_resize_value = FLAGS.max_resize_value
              else:
                max_resize_value = data_inforamtion.height

            train_generator = data_generator.Dataset(
                dataset_name=FLAGS.dataset_name,
                split_name=FLAGS.train_split,
                # dataset_dir=FLAGS.dataset_dir,
                batch_size=clone_batch_size,
                # HU_window=DATA_INFO.HU_window,
                pre_crop_flag=FLAGS.pre_crop_flag,
                mt_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                mt_class=FLAGS.z_class,
                mt_label_type="z_label",
                crop_size=data_inforamtion.train["train_crop_size"],
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                resize_factor=FLAGS.resize_factor,
                min_scale_factor=FLAGS.min_scale_factor,
                max_scale_factor=FLAGS.max_scale_factor,
                scale_factor_step_size=FLAGS.scale_factor_step_size,
                # model_variant=FLAGS.model_variant,
                num_readers=2,
                is_training=True,
                shuffle_data=True,
                repeat_data=True,
                prior_num_slice=FLAGS.prior_num_slice,
                prior_num_subject=FLAGS.prior_num_subject,
                # prior_dir=FLAGS.prior_dir,
                seq_length=FLAGS.seq_length,
                seq_type="bidirection")

            # TODO: no validation option
            val_generator = data_generator.Dataset(
                dataset_name=FLAGS.dataset_name,
                split_name=["val"],
                # dataset_dir=FLAGS.dataset_dir,
                batch_size=1,
                # HU_window=DATA_INFO.HU_window,
                mt_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                mt_class=FLAGS.z_class,
                mt_label_type="z_label",
                crop_size=[data_inforamtion.height, data_inforamtion.width],
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                # resize_factor=FLAGS.resize_factor,
                # min_scale_factor=FLAGS.min_scale_factor,
                # max_scale_factor=FLAGS.max_scale_factor,
                # scale_factor_step_size=FLAGS.scale_factor_step_size,
                # model_variant=FLAGS.model_variant,
                num_readers=2,
                is_training=False,
                shuffle_data=False,
                repeat_data=True,
                prior_num_slice=FLAGS.prior_num_slice,
                prior_num_subject=FLAGS.prior_num_subject,
                # prior_dir=FLAGS.prior_dir,
                seq_length=FLAGS.seq_length,
                seq_type="bidirection")

            model_options = common.ModelOptions(
              outputs_to_num_classes=train_generator.num_of_classes,
              crop_size=data_inforamtion.train["train_crop_size"],
              output_stride=FLAGS.output_stride)
            check_model_conflict(model_options)

            dataset1 = train_generator.get_one_shot_iterator()
            dataset2 = val_generator.get_one_shot_iterator()
            iter1 = dataset1.make_one_shot_iterator()
            iter2 = dataset2.make_one_shot_iterator()

            train_samples = iter1.get_next()
            val_samples = iter2.get_next()
            steps = tf.compat.v1.placeholder(tf.int32, shape=[])
            # handle = tf.compat.v1.placeholder(tf.string, shape=[])
            # iterator = tf.data.Iterator.from_string_handle(
            #   handle, iter1.output_types, iter1.output_shapes)
            # samples = iterator.get_next()

            train_tensor, summary_op = _train_deeplab_model(
                train_samples, train_generator.num_of_classes, model_options,
                train_generator.ignore_label)

            val_tensor, _ = _val_deeplab_model(
                val_samples, val_generator.num_of_classes, model_options,
                val_generator.ignore_label, steps)

            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)

            # TODO:
            # last_layers = model.get_extra_layer_scopes(
            #     FLAGS.last_layers_contain_logits_only)
            init_fn = None
            if FLAGS.tf_initial_checkpoint:
                init_fn = train_utils.get_model_init_fn(
                    train_logdir=FLAGS.train_logdir,
                    tf_initial_checkpoint=FLAGS.tf_initial_checkpoint,
                    initialize_last_layer=FLAGS.initialize_last_layer,
                    # last_layers,
                    ignore_missing_vars=True)

            scaffold = tf.train.Scaffold(
                init_fn=init_fn,
                summary_op=summary_op,
            )

            stop_hook = tf.train.StopAtStepHook(FLAGS.training_number_of_steps)
            # # save_hook = tf.train.CheckpointSaverHook(checkpoint_dir=FLAGS.train_logdir,
            # #                                          save_steps=FLAGS.save_checkpoint_steps,
            # #                                          saver=tf.train.Saver(var_list=tf.trainable_variables()))

            # train_handle = iter1.string_handle("train")
            # test_handle = iter2.string_handle("val")
            # ds_handle_hook = DSHandleHook(train_handle, test_handle)

            # # Define summary writer for saving "training" logs
            # writer = tf.summary.FileWriter(FLAGS.train_logdir+"train_envs/",
            #                                 graph=tf.get_default_graph())
            # writer.add_summary(t_summaries, step)
            saver = tf.train.Saver()
            with tf.train.MonitoredTrainingSession(
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                config=session_config,
                scaffold=scaffold,
                checkpoint_dir=FLAGS.train_logdir,
                log_step_count_steps=FLAGS.log_steps,
                save_summaries_steps=20,
                save_checkpoint_steps=FLAGS.save_checkpoint_steps,
                hooks=[stop_hook]) as sess:

                # step=0
                total_val_loss, total_val_steps = [], []
                best_model_performance = 0.0
                while not sess.should_stop():
                    _, global_step = sess.run([train_tensor, tf.train.get_global_step()])
                    if global_step%FLAGS.validation_steps == 0:
                      cm_total = 0
                      for sub_split in val_generator.splits_to_sizes.values():
                        for j in range(sub_split["val"]):
                          cm_total += sess.run(val_tensor, feed_dict={steps: j})

                      mean_dice_score, _ = eval_utils.compute_mean_dsc(cm_total)


                      total_val_loss.append(mean_dice_score)
                      total_val_steps.append(global_step)
                      plt.legend(["validation loss"])
                      plt.xlabel("global step")
                      plt.ylabel("loss")
                      plt.plot(total_val_steps, total_val_loss, "bo-")
                      plt.grid(True)
                      plt.savefig(FLAGS.train_logdir+"/losses.png")



                      # _, steps = sess.run([train_tensor, tf.train.get_or_create_global_step()], feed_dict={handle: ds_handle_hook.train_handle})
                      # total_steps.append(steps)

                      # loss, t_summaries, steps = sess.run([train_tensor, summary_op, tf.train.get_or_create_global_step()],
                      #                              feed_dict={handle: ds_handle_hook.train_handle})
                      # print(30*"-", steps)

                      # if steps%2  == 0:
                      #   train_loss.append(loss)
                      #   loss, v_summaries = sess.run([train_tensor, summary_op],
                      #                                feed_dict={handle: ds_handle_hook.valid_handle})
                      #   valid_loss.append(loss)

                      # if steps%200 == 0:
                      #   plt.plot(train_loss)
                      #   plt.hold(True)
                      #   plt.plot(valid_loss)
                      #   plt.savefig(FLAGS.train_logdir+"/losses.png")
            with open(os.path.join(path, 'logging.txt'), 'a') as f:
              f.write("\nEnd time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
              f.write("\n")
if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)