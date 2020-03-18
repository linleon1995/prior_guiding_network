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
from model import voxelmorph
from datasets import data_generator
from utils import train_utils
from core import features_extractor
import input_preprocess
from tensorflow.python.ops import math_ops
colorize = train_utils.colorize

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

PRIOR_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_tesis2/prior/'
LOGGING_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/'
PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/pretrained_weight/resnet/resnet_v1_50/model.ckpt'
DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord/'
# PRETRAINED_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_003/model.ckpt-40000'

# LOGGING_PATH = '/mnt/md0/home/applyACC/EE_ACM528/EE_ACM528_04/project/tf_thesis/thesis_trained/'
# DATASET_DIR = '/mnt/md0/home/applyACC/EE_ACM528/EE_ACM528_04/project/data/tfrecord/'

HU_WINDOW = [-125, 275]
TRAIN_CROP_SIZE = [256,256]
# TRAIN_CROP_SIZE = [512, 512]

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
        if len(os.listdir(path)) == 0:
            break
        idx += 1
        path = os.path.join(train_logdir, "run_{:03d}".format(idx))

    os.makedirs(path)
    return path

parser = argparse.ArgumentParser()

# Training configuration
parser.add_argument('--train_logdir', type=str, default=create_training_path(LOGGING_PATH),
                    help='')

parser.add_argument('--prior-dir', type=str, default=PRIOR_PATH,
                    help='')

parser.add_argument('--train_split', type=str, default='train',
                    help='')

parser.add_argument('--batch_size', type=int, default=16,
                    help='')

parser.add_argument('--tf_initial_checkpoint', type=str, default=PRETRAINED_PATH,
                    help='')

parser.add_argument('--initialize_last_layer', type=bool, default=True,
                    help='')

parser.add_argument('--training_number_of_steps', type=int, default=80000,
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

parser.add_argument('--model_variant', type=str, default=None,
                    help='')

# Model configuration
parser.add_argument('--z_label_method', type=str, default='regression',
                    help='')

parser.add_argument('--z_class', type=int, default=60,
                    help='')

# Input prior could be "zeros", "ground_truth", "training_data_fusion" (fixed) 
# or "adaptive" witch means decide adaptively
parser.add_argument('--guidance_type', type=str, default="training_fusion",
                    help='')

parser.add_argument('--num_prior_samples', type=int, default=None,
                    help='')

parser.add_argument('--fusion_rate', type=float, default=0.2,
                    help='')

parser.add_argument('--affine_transform', type=bool, default=False,
                    help='')

parser.add_argument('--deformable_transform', type=bool, default=False,
                    help='')


parser.add_argument('--z_loss_decay', type=float, default=1.0,
                    help='')

parser.add_argument('--transformed_loss_decay', type=float, default=1e-10,
                    help='')

parser.add_argument('--guidance_loss_decay', type=float, default=1e-1,
                    help='')

# Learning rate configuration
parser.add_argument('--learning_policy', type=str, default='poly',
                    help='')

parser.add_argument('--base_learning_rate', type=float, default=5e-3,
                    help='')

parser.add_argument('--learning_rate_decay_step', type=float, default=0,
                    help='')

parser.add_argument('--learning_rate_decay_factor', type=float, default=1e-1,
                    help='')

parser.add_argument('--learning_power', type=float, default=0.7,
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

parser.add_argument('--output_stride', type=int, default=8,
                    help='')

parser.add_argument('--num_clones', type=int, default=2,
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


def _build_network(samples, outputs_to_num_classes, model_options, ignore_label, s2=None):
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
    samples[common.PRIOR_IMGS] = tf.identity(samples[common.PRIOR_IMGS], name=common.PRIOR_IMGS)
  else:
    samples[common.PRIOR_IMGS] = None

  if common.PRIOR_SEGS in samples:
    samples[common.PRIOR_SEGS] = tf.identity(samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
  else:
    samples[common.PRIOR_SEGS] = None

  if common.GUIDANCE in samples:
    samples[common.GUIDANCE] = tf.identity(samples[common.GUIDANCE], name=common.GUIDANCE)
  else:
    samples[common.GUIDANCE] = None

  # TODO: moving_segs
  clone_batch_size = FLAGS.batch_size // FLAGS.num_clones
  output_dict, layers_dict = model.pgb_network(
                samples[common.IMAGE],
                model_options=model_options,
                affine_transform=FLAGS.affine_transform,
                deformable_transform=FLAGS.deformable_transform,
                labels=samples[common.LABEL],
                prior_imgs=samples[common.PRIOR_IMGS],
                prior_segs=samples[common.PRIOR_SEGS],
                num_classes=outputs_to_num_classes['semantic'],
                num_slices=samples[common.NUM_SLICES],
                prior_slices=prior_slices,
                batch_size=clone_batch_size,
                z_label_method=FLAGS.z_label_method,
                z_label=samples[common.Z_LABEL],
                fusion_rate=FLAGS.fusion_rate,
                guidance_type=FLAGS.guidance_type,
                # weight_decay=FLAGS.weight_decay,
                is_training=True,
                # fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
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
    
  guidance_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'guidance' in dict_key}
  if len(guidance_dict) == 0:
    guidance_dict = None

  # Log the summary
  _log_summaries(samples[common.IMAGE],
                 samples[common.LABEL],
                 outputs_to_num_classes['semantic'],
                 output_dict[common.OUTPUT_TYPE],
                 z_label=samples[common.Z_LABEL],
                 z_pred=z_pred,
                 prior_imgs=prior_img,
                 prior_segs=prior_seg,
                 guidance=guidance_dict,
                 guidance_original=output_dict['original_guidance'])
  return output_dict, layers_dict


def _tower_loss(iterator, num_of_classes, model_options, ignore_label, scope, reuse_variable, it2=None):
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
    samples = iterator.get_next()
    if it2 is not None:
      s2 = it2.get_next()
    else:
      s2 = None
    output_dict, layers_dict = _build_network(samples, {common.OUTPUT_TYPE: num_of_classes}, 
                                              model_options, ignore_label, s2)

  loss_dict = {common.OUTPUT_TYPE: 'mean_dice_coefficient'}
  if common.OUTPUT_Z in output_dict:
    if FLAGS.z_label_method == 'regression':
      loss_dict[common.OUTPUT_Z] = 'MSE'
    elif FLAGS.z_label_method == 'classification':
      loss_dict[common.OUTPUT_Z] = 'cross_entropy_zlabel'

  # TODO: Other voxel_morph loss
  # TODO: Repeated clone name --> clone_0/Losses/clone_0/guidance_loss/mean_dice_coefficient
  # TODO: loss summaries without decay
  losses = train_utils.get_losses(output_dict, layers_dict,
                                  samples,
                                  loss_dict=loss_dict,
                                  z_loss_decay=FLAGS.z_loss_decay,
                                  transformed_loss_decay=FLAGS.transformed_loss_decay,
                                  guidance_loss_decay=FLAGS.guidance_loss_decay)
  seg_loss = losses[0]
  
  for loss in losses:
    tf.summary.scalar('Losses/%s' % loss.op.name, loss)

  regularization_loss = tf.losses.get_regularization_loss(scope=scope)
  regularization_loss = tf.identity(regularization_loss, name='regularization_loss_with_decay')
  tf.summary.scalar('Losses/%s' % regularization_loss.op.name,
                    regularization_loss)
  total_loss = tf.add_n([tf.add_n(losses), regularization_loss])
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

  if prior_segs is not None:
    tf.summary.image('samples/%s' % 'prior_segs6', colorize(prior_segs[...,6:7], cmap='viridis'))
    tf.summary.image('samples/%s' % 'prior_segs7', colorize(prior_segs[...,7:8], cmap='viridis'))

  # if guidance is not None:
  #   tf.summary.image('guidance/%s' % 'guidance_original0_6', colorize(guidance_original[...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance_original0_7', colorize(guidance_original[...,7:8], cmap='viridis'))
    
  #   tf.summary.image('guidance/%s' % 'guidance0_6', colorize(guidance['guidance_in'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance0_7', colorize(guidance['guidance_in'][...,7:8], cmap='viridis'))
    
  #   tf.summary.image('guidance/%s' % 'guidance1_6', colorize(guidance['guidance1'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance1_7', colorize(guidance['guidance1'][...,7:8], cmap='viridis'))
    
  #   tf.summary.image('guidance/%s' % 'guidance2_6', colorize(guidance['guidance2'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance2_7', colorize(guidance['guidance2'][...,7:8], cmap='viridis'))
    
  #   tf.summary.image('guidance/%s' % 'guidance3_6', colorize(guidance['guidance3'][...,6:7], cmap='viridis'))
  #   tf.summary.image('guidance/%s' % 'guidance3_7', colorize(guidance['guidance3'][...,7:8], cmap='viridis'))
  
   
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


def _train_deeplab_model(iterator, num_of_classes, model_options, ignore_label, it2=None):
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
            reuse_variable=(i != 0),
            it2=it2)
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

            path =  '/home/acm528_02/Jing_Siang/data/Synpase_raw/'

            dataset = data_generator.Dataset(
                dataset_name=FLAGS.dataset,
                split_name=FLAGS.train_split,
                dataset_dir=FLAGS.dataset_dir,
                affine_transform=FLAGS.affine_transform,
                deformable_transform=FLAGS.deformable_transform,
                batch_size=clone_batch_size,
                HU_window=HU_WINDOW,
                z_label_method=FLAGS.z_label_method,
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
                num_prior_samples=FLAGS.num_prior_samples)

            model_options = common.ModelOptions(
              outputs_to_num_classes=dataset.num_of_classes,
              crop_size=TRAIN_CROP_SIZE,
              output_stride=FLAGS.output_stride)
            
            # TODO: it2
            # if import_prior_imgs or import_prior_segs:
            #   iterator, it2 = dataset.get_one_shot_iterator()
            #   train_tensor, summary_op = _train_deeplab_model(
            #     iterator, dataset.num_of_classes,
            #     dataset.ignore_label, it2)
            # else:
            iterator = dataset.get_one_shot_iterator()
            train_tensor, summary_op = _train_deeplab_model(
                iterator, dataset.num_of_classes, model_options, 
                dataset.ignore_label)


            # train_tensor, summary_op = _train_deeplab_model(
            #     dataset.get_one_shot_iterator(), dataset.num_of_classes,
            #     dataset.ignore_label, prior_segs_set.get_one_shot_iterator())

            # Soft placement allows placing on CPU ops without GPU implementation.
            session_config = tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=False)

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
            save_summaries_hook = tf.train.SummarySaverHook(save_steps=10,
                                                            output_dir=FLAGS.train_logdir,
                                                            summary_op=summary_op)
            
            
            with tf.train.MonitoredTrainingSession(
                master=FLAGS.master,
                is_chief=(FLAGS.task == 0),
                config=session_config,
                scaffold=scaffold,
                checkpoint_dir=FLAGS.train_logdir,
                log_step_count_steps=FLAGS.log_steps,
                save_summaries_steps=100,
                # save_checkpoint_secs=FLAGS.save_interval_secs,
                save_checkpoint_steps=FLAGS.save_checkpoint_steps,
                hooks=[stop_hook]) as sess:
                while not sess.should_stop():
                    sess.run([train_tensor])


if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
