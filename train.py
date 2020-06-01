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
from utils import train_utils
from core import features_extractor
import input_preprocess
from tensorflow.python.ops import math_ops
import math
colorize = train_utils.colorize
spatial_transfom_exp = experiments.spatial_transfom_exp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PRIOR_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/'
LOGGING_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/'
PRETRAINED_PATH = None
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/pretrained_weight/resnet/resnet_v1_50/model.ckpt'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_042/model.ckpt-40000'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-50000'
DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord/'
# DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord_seq/'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_104/model.ckpt-50000'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-40000'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_000/model.ckpt-5000'
# LOGGING_PATH = '/mnt/md0/home/applyACC/EE_ACM528/EE_ACM528_04/project/tf_thesis/thesis_trained/'
# DATASET_DIR = '/mnt/md0/home/applyACC/EE_ACM528/EE_ACM528_04/project/data/tfrecord/'

HU_WINDOW = [-125, 275]
TRAIN_CROP_SIZE = [256, 256]
# TRAIN_CROP_SIZE = [512, 512]

FUSIONS = 5*["concat"]
# FUSIONS = ["guid", "sum", "sum", "sum", "sum"]
FUSIONS = 5*["slim_guid"]
FUSIONS = ["slim_guid_plain", "sum", "sum", "sum", "sum"]
FUSIONS = ["slim_guid", "sum", "sum", "sum", "sum"]
FUSIONS = 5*["sum"]
FUSIONS = 5*["guid_uni"]

SEG_LOSS = "softmax_dice_loss"
GUID_LOSS = "softmax_dice_loss"
GUID_LOSS = "sigmoid_cross_entropy"
STAGE_PRED_LOSS = "softmax_dice_loss"
STAGE_PRED_LOSS = "sigmoid_cross_entropy"
# STAGE_PRED_LOSS = "softmax_cross_entropy"
SEG_WEIGHT_FLAG = False

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

parser.add_argument('--predict_without_background', type=bool, default=False,
                    help='')

parser.add_argument('--guid_encoder', type=str, default="last_stage_feature",
                    help='')

parser.add_argument('--guid_method', type=str, default=None,
                    help='')

parser.add_argument('--out_node', type=int, default=32,
                    help='')

parser.add_argument('--guid_conv_type', type=str, default="conv",
                    help='')
                    
parser.add_argument('--guid_conv_nums', type=int, default=1,
                    help='')

parser.add_argument('--share', type=bool, default=True,
                    help='')

parser.add_argument('--guidance_acc', type=str, default=None,
                    help='')

parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='')

# Training configuration
parser.add_argument('--train_logdir', type=str, default=create_training_path(LOGGING_PATH),
                    help='')

parser.add_argument('--prior_dir', type=str, default=PRIOR_PATH,
                    help='')

parser.add_argument('--train_split', type=str, default='train',
                    help='')

parser.add_argument('--batch_size', type=int, default=16,
                    help='')

parser.add_argument('--tf_initial_checkpoint', type=str, default=PRETRAINED_PATH,
                    help='')

parser.add_argument('--initialize_last_layer', type=bool, default=True,
                    help='')

parser.add_argument('--training_number_of_steps', type=int, default=140000,
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

parser.add_argument('--save_checkpoint_steps', type=int, default=5000,
                    help='')

parser.add_argument('--save_interval_secs', type=int, default=1800,
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

parser.add_argument('--prior_num_subject', type=int, default=20,
                    help='')

parser.add_argument('--fusion_slice', type=float, default=3,
                    help='')

parser.add_argument('--affine_transform', type=bool, default=False,
                    help='')

parser.add_argument('--deformable_transform', type=bool, default=False,
                    help='')

parser.add_argument('--z_loss_decay', type=float, default=None,
                    help='')

parser.add_argument('--stage_pred_loss_decay', type=float, default=1.0,
                    help='')

parser.add_argument('--guidance_loss_decay', type=float, default=1.0,
                    help='')

parser.add_argument('--regularization_weight', type=float, default=None,
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
parser.add_argument('--dataset', type=str, default='2013_MICCAI_Abdominal',
                    help='')

parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                    help='')

parser.add_argument('--output_stride', type=int, default=None,
                    help='')

parser.add_argument('--num_clones', type=int, default=1,
                    help='')

parser.add_argument('--min_scale_factor', type=float, default=0.75,
                    help='')

parser.add_argument('--max_scale_factor', type=float, default=1.25,
                    help='')

parser.add_argument('--scale_factor_step_size', type=float, default=0.125,
                    help='')

parser.add_argument('--min_resize_value', type=int, default=256,
                    help='')

parser.add_argument('--max_resize_value', type=int, default=256,
                    help='')

parser.add_argument('--resize_factor', type=int, default=None,
                    help='')


def check_model_conflict(model_options):
  pass
#     if not model_options.decoder_type == "refinement_network":
#         assert FLAGS.guidance_type is None

    # if FLAGS.affine_transform:
    #     assert FLAGS.transform_loss_decay is not None

    # if FLAGS.affine_transform:
    #     assert common.PRIOR_SEGS in dataset

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

def _build_network(samples, outputs_to_num_classes, model_options, ignore_label):
  """Builds a clone of DeepLab.
  Args:
    iterator: An iterator of type tf.data.Iterator for images and labels.
    outputs_to_num_classes: A map from output type to the number of classes. For
      example, for the task of semantic segmentation with 21 semantic classes,
      we would have outputs_to_num_classes['semantic'] = 21.
    ignore_label: Ignore label.
  """

  # Add name to input and label nodes so we can add to summary.
  samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
  samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

  if 'prior_slices' in samples:
    prior_slices = samples['prior_slices']
  else:
    prior_slices = None

  if common.PRIOR_IMGS in samples:
    samples[common.PRIOR_IMGS] = tf.identity(
        samples[common.PRIOR_IMGS], name=common.PRIOR_IMGS)
  else:
    samples[common.PRIOR_IMGS] = None

  if common.PRIOR_SEGS in samples:
    samples[common.PRIOR_SEGS] = tf.identity(
        samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
  else:
    samples[common.PRIOR_SEGS] = None

  if common.GUIDANCE in samples:
    samples[common.GUIDANCE] = tf.identity(samples[common.GUIDANCE], name=common.GUIDANCE)
  else:
    samples[common.GUIDANCE] = None

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
  
  
  if FLAGS.guid_method is not None:
    if FLAGS.guidance_loss_decay is None:
      raise ValueError("guidance loss")
    FUSIONS[0] = FLAGS.guid_method

  num_class = outputs_to_num_classes['semantic']  
  if FLAGS.predict_without_background:
    num_class -= 1
    
  output_dict, layers_dict = model.pgb_network(
                samples[common.IMAGE],
                model_options=model_options,
                affine_transform=FLAGS.affine_transform,
                # deformable_transform=FLAGS.deformable_transform,
                # labels=samples[common.LABEL],
                samples=samples["organ_label"],
                # prior_imgs=samples[common.PRIOR_IMGS],
                prior_segs=samples[common.PRIOR_SEGS],
                num_class=num_class,
                # num_slices=samples[common.NUM_SLICES],
                prior_slice=prior_slices,
                batch_size=clone_batch_size,
                z_label_method=FLAGS.z_label_method,
                # z_label=samples[common.Z_LABEL],
                guidance_type=FLAGS.guidance_type,
                fusion_slice=FLAGS.fusion_slice,
                prior_dir=FLAGS.prior_dir,
                drop_prob=FLAGS.drop_prob,
                stn_in_each_class=True,
                # prior_num_slice=FLAGS.prior_num_slice,
                reuse=tf.AUTO_REUSE,
                is_training=True,
                weight_decay=FLAGS.weight_decay,
                # fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                guidance_acc=FLAGS.guidance_acc,
                share=FLAGS.share,
                fusions=FUSIONS,
                out_node=FLAGS.out_node,
                guid_encoder=FLAGS.guid_encoder,
                z_model=FLAGS.z_model,
                z_class=FLAGS.z_class,
                guidance_loss=GUID_LOSS,
                stage_pred_loss=STAGE_PRED_LOSS,
                guid_conv_nums=FLAGS.guid_conv_nums,
                guid_conv_type=FLAGS.guid_conv_type,
                )

  # Add name to graph node so we can add to summary.
  output = output_dict[common.OUTPUT_TYPE]
  output = tf.identity(output, name=common.OUTPUT_TYPE)

  if common.Z_LABEL in samples:
    samples[common.Z_LABEL] = tf.identity(samples[common.Z_LABEL], name=common.Z_LABEL)
  else:
    samples[common.Z_LABEL] = None

  if common.PRIOR_IMGS in output_dict:
    prior_img = output_dict[common.PRIOR_IMGS]
  else:
    prior_img = None

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

  # Log the summary
  _log_summaries(samples[common.IMAGE],
                 samples[common.LABEL],
                 outputs_to_num_classes['semantic'],
                 output_dict[common.OUTPUT_TYPE],
                 z_label=samples[common.Z_LABEL],
                 z_pred=z_pred,
                 prior_imgs=prior_img,
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
                                                model_options, ignore_label)

    loss_dict = {}
    seg_weight = 1.0
    guidance_loss_weight = FLAGS.guidance_loss_decay
    stage_pred_loss_weight = FLAGS.stage_pred_loss_decay
    # stage_pred_loss_weight = [0.04] + 13*[1.0]
    
    # seg_weight = train_utils.get_loss_weight(samples[common.LABEL], loss_name, loss_weight=SEG_WEIGHT_FLAG)
      
    loss_dict[common.OUTPUT_TYPE] = {"loss": SEG_LOSS, "decay": None, "weights": seg_weight, "scope": "segmenation"}
    if FLAGS.guidance_loss_decay is not None:
      # guidance_loss_weight = train_utils.get_loss_weight(samples[common.LABEL], GUID_LOSS, 
      #                                                    decay=FLAGS.guidance_loss_decay)
      loss_dict[common.GUIDANCE] = {"loss": GUID_LOSS, "decay": None, "weights": guidance_loss_weight, "scope": "guidance"}
    if FLAGS.stage_pred_loss_decay is not None:
      # stage_pred_loss_weight = train_utils.get_loss_weight(samples[common.LABEL], STAGE_PRED_LOSS, 
      #                                                      decay=FLAGS.stage_pred_loss_decay)
      loss_dict["stage_pred"] = {"loss": STAGE_PRED_LOSS, "decay": None, "weights": stage_pred_loss_weight, "scope": "stage_pred"}

    train_utils.get_losses(output_dict, 
                           layers_dict, 
                           samples, 
                           loss_dict,
                           num_of_classes,
                           predict_without_background=FLAGS.predict_without_background)
    
    # if FLAGS.z_loss_decay is not None:
    #     if FLAGS.z_label_method.split("_")[1] == 'regression':
    #         loss_dict[common.OUTPUT_Z] = {"loss": "MSE", "decay": FLAGS.z_loss_decay}
    #     elif FLAGS.z_label_method.split("_")[1] == 'classification':
    #         loss_dict[common.OUTPUT_Z] = {"loss": "cross_entropy_zlabel", "decay": FLAGS.z_loss_decay}

    # if FLAGS.guidance_loss_decay is not None:
    #     loss_dict[common.GUIDANCE] = {"loss": "mean_dice_coefficient", "decay": FLAGS.guidance_loss_decay}

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


def _log_summaries(input_image, label, num_of_classes, output, z_label, z_pred, prior_imgs, prior_segs,
                   guidance, guidance_original):
  """Logs the summaries for the model.
  Args:
    input_image: Input image of the model. Its shape is [batch_size, height,
      width, channel].
    label: Label of the image. Its shape is [batch_size, height, width].
    num_of_classes: The number of classes of the dataset.
    output: Output of the model. Its shape is [batch_size, height, width].
  """
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
  if prior_imgs is not None:
    tf.summary.image('samples/%s' % 'prior_imgs', colorize(prior_imgs, cmap='viridis'))

  if guidance is not None:
    # tf.summary.image('reg_field/%s' % 'field_x', colorize(field[...,0:1], cmap='viridis'))
    # tf.summary.image('reg_field/%s' % 'field_y', colorize(field[...,1:2], cmap='viridis'))

    if FLAGS.affine_transform:
      tf.summary.image('guidance/%s' % 'guidance_original0_6', colorize(guidance_original[...,6:7], cmap='viridis'))
      tf.summary.image('guidance/%s' % 'guidance_original0_7', colorize(guidance_original[...,7:8], cmap='viridis'))

    tf.summary.image('guidance/%s' % 'guidance1_6', colorize(guidance['guidance4'][...,6:7], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guidance1_7', colorize(guidance['guidance4'][...,7:8], cmap='viridis'))

    tf.summary.image('guidance/%s' % 'guidance2_6', colorize(guidance['guidance3'][...,6:7], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guidance2_7', colorize(guidance['guidance3'][...,7:8], cmap='viridis'))

    tf.summary.image('guidance/%s' % 'guidance3_6', colorize(guidance['guidance2'][...,6:7], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guidance3_7', colorize(guidance['guidance2'][...,7:8], cmap='viridis'))

    tf.summary.image('guidance/%s' % 'guidance4_6', colorize(guidance['guidance1'][...,6:7], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guidance4_7', colorize(guidance['guidance1'][...,7:8], cmap='viridis'))

    tf.summary.image('guidance/%s' % 'guidance5/logits_6', colorize(output[...,6:7], cmap='viridis'))
    tf.summary.image('guidance/%s' % 'guidance5/logits_7', colorize(output[...,7:8], cmap='viridis'))

  if z_label is not None and z_pred is not None:
    clone_batch_size = FLAGS.batch_size // FLAGS.num_clones

    z_label = tf.reshape(z_label, [1,clone_batch_size,1,1])
    z_label = tf.tile(z_label, [1,1,clone_batch_size,1])
    tf.summary.image('samples/%s' % 'z_label', colorize(z_label, cmap='viridis'))

    z_pred = tf.reshape(z_pred, [1,clone_batch_size,1,1])
    z_pred = tf.tile(z_pred, [1,1,clone_batch_size,1])
    tf.summary.image('samples/%s' % 'z_pred', colorize(z_pred, cmap='viridis'))


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


def _train_deeplab_model(iterator, num_of_classes, model_options, ignore_label, r=None):
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
  total_loss = total_seg_loss = 0
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
            # reuse_variable=r
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


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.train_logdir)
    # tf.gfile.MakeDirs(FLAGS.train_logdir+"/train_envs/")
    # tf.gfile.MakeDirs(FLAGS.train_logdir+"/val_envs/")
    tf.logging.info('Training on %s set', FLAGS.train_split)

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

            dataset = data_generator.Dataset(
                dataset_name=FLAGS.dataset,
                split_name=FLAGS.train_split,
                dataset_dir=FLAGS.dataset_dir,
                affine_transform=FLAGS.affine_transform,
                deformable_transform=FLAGS.deformable_transform,
                batch_size=clone_batch_size,
                HU_window=HU_WINDOW,
                z_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                z_class=FLAGS.z_class,
                crop_size=TRAIN_CROP_SIZE,
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
                prior_dir=FLAGS.prior_dir)


            dataset2 = data_generator.Dataset(
                dataset_name=FLAGS.dataset,
                split_name="val",
                dataset_dir=FLAGS.dataset_dir,
                affine_transform=FLAGS.affine_transform,
                deformable_transform=FLAGS.deformable_transform,
                batch_size=clone_batch_size,
                HU_window=HU_WINDOW,
                z_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                z_class=FLAGS.z_class,
                crop_size=TRAIN_CROP_SIZE,
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
                prior_dir=FLAGS.prior_dir)
            
            model_options = common.ModelOptions(
              outputs_to_num_classes=dataset.num_of_classes,
              crop_size=TRAIN_CROP_SIZE,
              output_stride=FLAGS.output_stride)
            check_model_conflict(model_options)
            
            d1 = dataset.get_one_shot_iterator()
            d2 = dataset2.get_one_shot_iterator()
            iter1 = d1.make_one_shot_iterator()
            iter2 = d2.make_one_shot_iterator()
            
            handle = tf.compat.v1.placeholder(tf.string, shape=[])
            iterator = tf.data.Iterator.from_string_handle(
              handle, iter1.output_types, iter1.output_shapes)
            samples = iterator.get_next()
            
            train_tensor, summary_op = _train_deeplab_model(
                samples, dataset.num_of_classes, model_options,
                dataset.ignore_label)

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
            
            train_handle = iter1.string_handle("train")
            test_handle = iter2.string_handle("val")
            ds_handle_hook = DSHandleHook(train_handle, test_handle)

            # # Define summary writer for saving "training" logs
            # writer = tf.summary.FileWriter(FLAGS.train_logdir+"train_envs/",
            #                                 graph=tf.get_default_graph())
            # writer.add_summary(t_summaries, step)
            
            with tf.train.MonitoredTrainingSession(
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                config=session_config,
                scaffold=scaffold,
                checkpoint_dir=FLAGS.train_logdir,
                log_step_count_steps=FLAGS.log_steps,
                save_summaries_steps=20,
                # save_checkpoint_secs=FLAGS.save_interval_secs,
                save_checkpoint_steps=FLAGS.save_checkpoint_steps,
                hooks=[stop_hook, ds_handle_hook]) as sess:
                
                step=0
                while not sess.should_stop():
                  
                      sess.run([train_tensor], feed_dict={handle: ds_handle_hook.train_handle})

                      # loss, t_summaries = sess.run([train_tensor, summary_op], feed_dict={handle: ds_handle_hook.train_handle})
                      
                      # if step%2  == 0:
                      #   loss, v_summaries = sess.run([train_tensor, summary_op], feed_dict={handle: ds_handle_hook.train_handle})
                        
                      # step+=1

            with open(os.path.join(path, 'logging.txt'), 'a') as f:
              f.write("\nEnd time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
              f.write("\n")
if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
