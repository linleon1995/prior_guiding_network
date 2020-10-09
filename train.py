import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
import six
import time
from tensorflow.python.ops import math_ops

import model
import common
from datasets import data_generator
from utils import train_utils, eval_utils
from core import features_extractor
import input_preprocess
colorize = train_utils.colorize
create_training_path = train_utils.create_training_path
str2bool = train_utils.str2bool

LOGGING_PATH = common.LOGGING_PATH


parser = argparse.ArgumentParser()
parser.add_argument('--fusions', nargs='+', required=True,
                    help='')

parser.add_argument('--dataset_name', nargs='+', required=True,
                    help='The list contains all the datasets name in training')

parser.add_argument('--train_split', nargs='+', required=True,
                    help='The list contains the splitting apply in training process')

parser.add_argument('--seq_length', type=int, default=1,
                    help='The slice number in single sequence sample')

parser.add_argument('--cell_type', type=str, default="ConvGRU",
                    help='')

parser.add_argument('--guid_fuse', type=str, default="sum_wo_back",
                    help='')

parser.add_argument('--apply_sram2', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--guid_encoder', type=str, default="early",
                    help='')

parser.add_argument('--out_node', type=int, default=32,
                    help='')

parser.add_argument('--guid_conv_type', type=str, default="conv",
                    help='')

parser.add_argument('--guid_conv_nums', type=int, default=2,
                    help='')

parser.add_argument('--share', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--weight_decay', type=float, default=1e-3,
                    help='')

# Training configuration
parser.add_argument('--train_logdir', type=str, default=create_training_path(LOGGING_PATH),
                    help='')

parser.add_argument('--batch_size', type=int, default=16,
                    help='')

parser.add_argument('--tf_initial_checkpoint', type=str, default=None,
                    help='')

parser.add_argument('--initialize_last_layer', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--training_number_of_steps', type=int, default=30000,
                    help='')

# TODO: test profile function
parser.add_argument('--profile_logdir', type=str, default='',
                    help='Where the profile files are stored.')

parser.add_argument('--master', type=str, default='',
                    help='')

parser.add_argument('--task', type=int, default=0,
                    help='')

parser.add_argument('--log_steps', type=int, default=10,
                    help='')

parser.add_argument('--save_summaries_secs', type=int, default=None,
                    help='')

parser.add_argument('--save_summaries_images', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--save_checkpoint_steps', type=int, default=1000,
                    help='')

parser.add_argument('--validation_steps', type=int, default=1000,
                    help='')

parser.add_argument('--num_ps_tasks', type=int, default=0,
                    help='')

parser.add_argument('--drop_prob', type=float, default=None,
                    help='')

# Model configuration
#tbc
parser.add_argument('--model_variant', type=str, default=None,
                    help='')

parser.add_argument('--z_model', type=str, default=None,
                    help='')

parser.add_argument('--z_label_method', type=str, default=None,
                    help='')

parser.add_argument('--z_class', type=int, default=None,
                    help='')

# Input prior could be "zeros", "ground_truth", "training_data_fusion" (fixed)
# ,or "adaptive" witch means decide adaptively by learning
parser.add_argument('--guidance_type', type=str, default="training_data_fusion",
                    help='')

parser.add_argument('--prior_num_slice', type=int, default=1,
                    help='')

parser.add_argument('--prior_num_subject', type=int, default=None,
                    help='')

#tbc
parser.add_argument('--fusion_slice', type=float, default=3,
                    help='')

parser.add_argument('--z_loss_weight', type=float, default=None,
                    help='')

parser.add_argument('--seg_loss_weight', nargs='+', type=float, default=1.0,
                    help="")
                    
parser.add_argument('--guid_loss_weight', nargs='+', type=float, default=1.0,
                    help="")

parser.add_argument('--stage_pred_loss_weight', nargs='+', type=float, default=1.0,
                    help="")

parser.add_argument('--seg_loss_name', type=str, default="softmax_dice_loss",
                    help='')
                    
parser.add_argument('--guid_loss_name', type=str, default=None,
                    help='')

parser.add_argument('--stage_pred_loss_name', type=str, default=None,
                    help='')

parser.add_argument('--z_loss_name', type=str, default=None,
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

#tbc
parser.add_argument('--output_stride', type=int, default=None,
                    help='')

parser.add_argument('--num_clones', type=int, default=1,
                    help='')

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

parser.add_argument('--pre_crop_flag', type=str2bool, nargs='?', const=True, default=True,
                    help='')


def check_model_conflict():
  # Different predict class amond datasets.
  pass


def get_session(sess):
  session = sess
  while type(session).__name__ != 'Session':
    #pylint: disable=W0212
    session = session._sess
  return session


def _build_network(samples, outputs_to_num_classes, model_options, ignore_label, is_training):
  """Builds a clone of pgn.
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

  summary_img = samples[common.IMAGE]
  summary_label = samples[common.LABEL]
  if FLAGS.seq_length > 1:
    summary_img = summary_img[:,1]
    summary_label = summary_label[:,1]

  if 'prior_slices' in samples:
    prior_slices = samples['prior_slices']
  else:
    prior_slices = None

  if common.PRIOR_SEGS in samples:
    samples[common.PRIOR_SEGS] = tf.identity(
        samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
  else:
    samples[common.PRIOR_SEGS] = None

  clone_batch_size = FLAGS.batch_size // FLAGS.num_clones

  num_class = outputs_to_num_classes['semantic']
  output_dict, layers_dict = model.pgb_network(
                samples[common.IMAGE],
                samples[common.HEIGHT],
                samples[common.WIDTH],
                model_options=model_options,
                prior_segs=samples[common.PRIOR_SEGS],
                num_class=num_class,
                prior_slice=prior_slices,
                batch_size=clone_batch_size,
                fusion_slice=FLAGS.fusion_slice,
                drop_prob=FLAGS.drop_prob,
                stn_in_each_class=True,
                reuse=tf.AUTO_REUSE,
                is_training=is_training,
                weight_decay=FLAGS.weight_decay,
                share=FLAGS.share,
                fusions=FLAGS.fusions,
                out_node=FLAGS.out_node,
                guid_encoder=FLAGS.guid_encoder,
                z_label_method=FLAGS.z_label_method,
                z_model=FLAGS.z_model,
                z_class=FLAGS.z_class,
                guidance_loss_name=FLAGS.guid_loss_name,
                stage_pred_loss_name=FLAGS.stage_pred_loss_name,
                guid_conv_nums=FLAGS.guid_conv_nums,
                guid_conv_type=FLAGS.guid_conv_type,
                apply_sram2=FLAGS.apply_sram2,
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
                 layers_dict=layers_dict,
                 guidance_original=guidance_original)
  return output_dict, layers_dict


def _tower_loss(iterator, num_of_classes, model_options, ignore_label, scope, reuse_variable):
    """Calculates the total loss on a single tower running the pgn model.
    Args:
        iterator: An iterator of type tf.data.Iterator for images and labels.
        num_of_classes: Number of classes for the dataset.
        ignore_label: Ignore label for the dataset.
        scope: Unique prefix string identifying the pgn tower.
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
    seg_loss_weight = FLAGS.seg_loss_weight
    if isinstance(FLAGS.seg_loss_weight, list):
      if len(FLAGS.seg_loss_weight) == 1:
        seg_loss_weight = FLAGS.seg_loss_weight[0]
        
    loss_dict[common.OUTPUT_TYPE] = {"loss": FLAGS.seg_loss_name, "weights": seg_loss_weight, "scope": "segmenation"}

    if FLAGS.guid_loss_name is not None:
      guid_loss_weight = FLAGS.guid_loss_weight
      if isinstance(FLAGS.guid_loss_weight, list):
        if len(FLAGS.guid_loss_weight) == 1:
          guid_loss_weight = FLAGS.guid_loss_weight[0]
      loss_dict[common.GUIDANCE] = {"loss": FLAGS.guid_loss_name, "weights": guid_loss_weight, "scope": "guidance"}

    if FLAGS.stage_pred_loss_name is not None:
      stage_pred_loss_weight = FLAGS.stage_pred_loss_weight
      if isinstance(FLAGS.stage_pred_loss_weight, list):
        if len(FLAGS.stage_pred_loss_weight) == 1:
          stage_pred_loss_weight = FLAGS.stage_pred_loss_weight[0]
      loss_dict["stage_pred"] = {"loss": FLAGS.stage_pred_loss_name, "weights": stage_pred_loss_weight, "scope": "stage_pred"}

    if FLAGS.z_loss_name is not None:
      z_loss_weight = FLAGS.z_loss_weight
      if isinstance(FLAGS.z_loss_weight, list):
        if len(FLAGS.z_loss_weight) == 1:
          z_loss_weight = FLAGS.z_loss_weight[0]
      loss_dict[common.OUTPUT_Z] = {"loss": "softmax_cross_entropy", "weights": z_loss_weight, "scope": "z_pred"}
      z_class = FLAGS.z_class
    else:
      z_class = None

    train_utils.get_losses(output_dict,
                           samples,
                           loss_dict,
                           num_of_classes,
                           FLAGS.seq_length,
                           FLAGS.batch_size,
                           z_class=z_class)

    losses = tf.compat.v1.losses.get_losses(scope=scope)
    seg_loss = losses[0]
    for loss in losses:
        tf.summary.scalar('Losses/%s' % loss.op.name, loss)
    total_loss = tf.add_n(losses)
    return total_loss, seg_loss


def _log_summaries(input_image, label, num_of_classes, output, z_pred, prior_segs,
                   layers_dict, guidance_original, **kwargs):
  """Logs the summaries for the model.
  The easiest way to add the summarirs for interesting feature is call
  tf.add_to_collections() during model building, then call 
  tf.get_collections() in this function.
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

    summary_feature_node = [0,1,2]
    for n in summary_feature_node:
      tf.summary.image('low_level1/node%d' %n, colorize(layers_dict["low_level1"][...,n:n+1], cmap='viridis'))
      tf.summary.image('low_level2/node%d' %n, colorize(layers_dict["low_level2"][...,n:n+1], cmap='viridis'))
      tf.summary.image('low_level3/node%d' %n, colorize(layers_dict["low_level3"][...,n:n+1], cmap='viridis'))
      tf.summary.image('low_level4/node%d' %n, colorize(layers_dict["low_level4"][...,n:n+1], cmap='viridis'))
      tf.summary.image('low_level5/node%d' %n, colorize(layers_dict["low_level5"][...,n:n+1], cmap='viridis'))

    # guid = tf.get_collection("guidance")
    # TODO: validation issue
    # print(guid)
    # tf.summary.image('guidance/guid0', colorize(guid[0][...,0:1], cmap='viridis'))
    # tf.summary.image('guidance/guid1', colorize(guid[1][...,0:1], cmap='viridis'))
    # tf.summary.image('guidance/guid2', colorize(guid[2][...,0:1], cmap='viridis'))
    # tf.summary.image('guidance/guid3', colorize(guid[3][...,0:1], cmap='viridis'))
    # tf.summary.image('guidance/guid4', colorize(guid[4][...,0:1], cmap='viridis'))
    # tf.summary.image('guidance/guid5', colorize(guid[5][...,0:1], cmap='viridis'))

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

def _train_pgn_model(iterator, num_of_classes, model_options, ignore_label, reuse=None):
  """Trains the pgn model.
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

  tower_summaries = tf.summary.merge_all()
  summaries.append(tf.summary.scalar('learning_rate', learning_rate))

  with tf.device('/cpu:0'):
    grads_and_vars = _average_gradients(tower_grads)
    if tower_summaries is not None:
      summaries.append(tower_summaries)

    # Create gradient update op.
    grad_updates = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)

    # Gather update_ops. These contain, for example,
    # the updates for the batch_norm variables created by model_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    update_ops.append(grad_updates)
    update_op = tf.group(*update_ops)

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


def _val_pgn_model(iterator, num_of_classes, model_options, ignore_label, steps, reuse=None):
  """Trains the pgn model.
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
  prediction = eval_utils.inference_segmentation(logits, dim=3)
  pred_flat = tf.reshape(prediction, shape=[-1,])
  if FLAGS.seq_length > 1:
      label = samples[common.LABEL][:,FLAGS.seq_length//2]
  else:
      label = samples[common.LABEL]
  labels_flat = tf.reshape(label, shape=[-1,])

  # Define Confusion Maxtrix
  cm = tf.confusion_matrix(labels_flat, pred_flat, num_classes=num_of_classes)

  summary_op = 0
  return cm, summary_op


def main(unused_argv):
    # TODO: single data information --> multiple
    data_inforamtion = data_generator._DATASETS_INFORMATION[FLAGS.dataset_name[0]]
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.gfile.MakeDirs(FLAGS.train_logdir)
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
                guidance_type=FLAGS.guidance_type,
                batch_size=clone_batch_size,
                pre_crop_flag=FLAGS.pre_crop_flag,
                mt_label_method=FLAGS.z_label_method,
                mt_class=FLAGS.z_class,
                mt_label_type="z_label",
                crop_size=data_inforamtion.train["train_crop_size"],
                min_resize_value=FLAGS.min_resize_value,
                max_resize_value=FLAGS.max_resize_value,
                resize_factor=FLAGS.resize_factor,
                min_scale_factor=FLAGS.min_scale_factor,
                max_scale_factor=FLAGS.max_scale_factor,
                scale_factor_step_size=FLAGS.scale_factor_step_size,
                num_readers=2,
                is_training=True,
                shuffle_data=True,
                repeat_data=True,
                prior_num_slice=FLAGS.prior_num_slice,
                prior_num_subject=FLAGS.prior_num_subject,
                seq_length=FLAGS.seq_length,
                seq_type="bidirection")

            if "val" not in FLAGS.train_split:
              val_generator = data_generator.Dataset(
                  dataset_name=FLAGS.dataset_name,
                  split_name=["val"],
                  guidance_type=FLAGS.guidance_type,
                  batch_size=1,
                  mt_label_method=FLAGS.z_label_method,
                  mt_class=FLAGS.z_class,
                  mt_label_type="z_label",
                  crop_size=[data_inforamtion.height, data_inforamtion.width],
                  min_resize_value=FLAGS.min_resize_value,
                  max_resize_value=FLAGS.max_resize_value,
                  num_readers=2,
                  is_training=False,
                  shuffle_data=False,
                  repeat_data=True,
                  prior_num_slice=FLAGS.prior_num_slice,
                  prior_num_subject=FLAGS.prior_num_subject,
                  seq_length=FLAGS.seq_length,
                  seq_type="bidirection")

            model_options = common.ModelOptions(
              outputs_to_num_classes=train_generator.num_of_classes,
              crop_size=data_inforamtion.train["train_crop_size"],
              output_stride=FLAGS.output_stride)

            steps = tf.compat.v1.placeholder(tf.int32, shape=[])

            dataset1 = train_generator.get_dataset()
            iter1 = dataset1.make_one_shot_iterator()
            train_samples = iter1.get_next()

            train_tensor, summary_op = _train_pgn_model(
                train_samples, train_generator.num_of_classes, model_options,
                train_generator.ignore_label)

            if "val" not in FLAGS.train_split:
              dataset2 = val_generator.get_dataset()
              iter2 = dataset2.make_one_shot_iterator()
              val_samples = iter2.get_next()

              val_tensor, _ = _val_pgn_model(
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
            saver = tf.train.Saver()
            best_dice = 0
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
                    if "val" not in FLAGS.train_split:
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

                        if mean_dice_score > best_dice:
                          best_dice = mean_dice_score
                          saver.save(get_session(sess), os.path.join(FLAGS.train_logdir, 'model.ckpt-best'))
                          # saver.save(get_session(sess), os.path.join(FLAGS.train_logdir, 'model.ckpt-best-%d' %global_step))
                          txt = 20*">" + " saving best mdoel model.ckpt-best-%d with DSC: %f" %(global_step,best_dice)
                          print(txt)
                          with open(os.path.join(path, 'logging.txt'), 'a') as f:
                            f.write(txt)
                            f.write("\n")

            with open(os.path.join(path, 'logging.txt'), 'a') as f:
              f.write("\nEnd time: {}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
              f.write("\n")
if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)