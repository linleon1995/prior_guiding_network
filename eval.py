# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Evaluation script for the DeepLab model.
See model.py for more details and usage.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import common
import model
from datasets import data_generator
from utils import eval_utils

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EVAL_CROP_SIZE = [256,256]
# EVAL_CROP_SIZE = [512,512]
ATROUS_RATES = None
# TODO: Multi-Scale Test
# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
EVAL_SCALES = [1.0]
HU_WINDOW = [-125, 275]
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_012/model.ckpt-40000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_007/model.ckpt-40000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_005/model.ckpt-36235'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_008/model.ckpt-40000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_009/model.ckpt-29382'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_000/model.ckpt-40000'
DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord/'

parser = argparse.ArgumentParser()

parser.add_argument('--master', type=str, default='',
                    help='')

# Settings for log directories.
parser.add_argument('--eval_logdir', type=str, default=CHECKPOINT+'eval/',
                    help='')

parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT,
                    help='')

# Settings for evaluating the model.
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='')

parser.add_argument('--eval_interval_secs', type=int, default=5,
                    help='')

parser.add_argument('--output_stride', type=int, default=8,
                    help='')

# Change to True for adding flipped images during test.
parser.add_argument('--add_flipped_images', type=bool, default=False,
                    help='')

parser.add_argument('--fusion_rate', type=float, default=0.2,
                    help='')

parser.add_argument('--z_label_method', type=str, default='regression',
                    help='')

parser.add_argument('--zero_guidance', type=bool, default=True,
                    help='')

parser.add_argument('--vis_guidance', type=bool, default=True,
                    help='')

parser.add_argument('--vis_features', type=bool, default=False,
                    help='')

parser.add_argument('--display_box_plot', type=bool, default=False,
                    help='')

# Dataset settings.

parser.add_argument('--dataset', type=str, default='2013_MICCAI_Abdominal',
                    help='')

parser.add_argument('--eval_split', type=str, default='val',
                    help='')

parser.add_argument('--dataset_dir', type=str, default=DATASET_DIR,
                    help='')

parser.add_argument('--max_number_of_evaluations', type=int, default=1,
                    help='')
# TODO: boxplot
# TODO: MSE for z information
# TODO: Automatically output --> mean_iou, mean_dsc, pixel_acc as text file,
# TODO: image, label, prediciton for subplots and full image

# TODO: Warning for dataset_split, guidance and procedure of guidance, MSE for z information
# TODO: testing mode for online eval

def load_model(saver, sess, ckpt_path):
    '''Load trained weights.

    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    '''
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # TODO:
  model_options = common.ModelOptions(
              outputs_to_num_classes=14,
              crop_size=EVAL_CROP_SIZE,
              output_stride=FLAGS.output_stride)

  dataset = data_generator.Dataset(
                dataset_name=FLAGS.dataset,
                split_name=FLAGS.eval_split,
                dataset_dir=FLAGS.dataset_dir,
                model_options=model_options,
                batch_size=FLAGS.eval_batch_size,
                HU_window=HU_WINDOW,
                z_label_method=FLAGS.z_label_method,
                z_class=60,
                crop_size=EVAL_CROP_SIZE,
                min_resize_value=EVAL_CROP_SIZE[0],
                max_resize_value=EVAL_CROP_SIZE[0],
                # resize_factor=FLAGS.resize_factor,
                min_scale_factor=0.75,
                max_scale_factor=1.25,
                scale_factor_step_size=0.25,
                # model_variant=FLAGS.model_variant,
                num_readers=2,
                is_training=False,
                shuffle_data=False,
                repeat_data=False,
                num_prior_samples=None)

  # TODO: make dirs?
  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default():
    samples = dataset.get_one_shot_iterator().get_next()

    # Add name to input and label nodes so we can add to summary.
    samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
    samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

    model_options = common.ModelOptions(
      outputs_to_num_classes=dataset.num_of_classes,
      crop_size=EVAL_CROP_SIZE,
      output_stride=FLAGS.output_stride)

    # Set shape in order for tf.contrib.tfprof.model_analyzer to work properly.
    samples[common.IMAGE].set_shape(
        [FLAGS.eval_batch_size,
         EVAL_CROP_SIZE[0],
         EVAL_CROP_SIZE[1],
         1])

    image_placeholder = tf.placeholder(tf.float32, shape=[1,EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1],1])
    label_placeholder = tf.placeholder(tf.int32, shape=[None,EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1],1])
    num_slices_placeholder = tf.placeholder(tf.int64, shape=[None])
    
    placeholder_dict = {common.IMAGE: image_placeholder,
                        common.LABEL: label_placeholder,
                        common.NUM_SLICES: num_slices_placeholder}

    if common.Z_LABEL in samples:
      samples[common.Z_LABEL] = tf.identity(samples[common.Z_LABEL], name=common.Z_LABEL)
      z_label_placeholder = tf.placeholder(tf.float32, shape=[None])
      placeholder_dict[common.Z_LABEL] = z_label_placeholder
    else:
      placeholder_dict[common.Z_LABEL] = None

    if common.PRIOR_IMGS in samples:
      samples[common.PRIOR_IMGS] = tf.identity(samples[common.PRIOR_IMGS], name=common.PRIOR_IMGS)
      prior_img_placeholder = tf.placeholder(tf.float32,
                                           shape=[None, EVAL_CROP_SIZE[0]//FLAGS.output_stride,
                                                  EVAL_CROP_SIZE[1]//FLAGS.output_stride, None])
      placeholder_dict[common.PRIOR_IMGS] = prior_img_placeholder
    else:
      placeholder_dict[common.PRIOR_IMGS] = None

    if common.PRIOR_SEGS in samples:
      samples[common.PRIOR_SEGS] = tf.identity(samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
      prior_seg_placeholder = tf.placeholder(tf.float32,
                                           shape=[None, EVAL_CROP_SIZE[0]//FLAGS.output_stride,
                                                  EVAL_CROP_SIZE[1]//FLAGS.output_stride, None])
      placeholder_dict[common.PRIOR_SEGS] = prior_seg_placeholder
    else:
      placeholder_dict[common.PRIOR_SEGS] = None

    if 'prior_slices' in samples:
      prior_slices_placeholder = tf.placeholder(tf.int64, shape=[None])
      placeholder_dict['prior_slices'] = prior_slices_placeholder
    else:
      placeholder_dict['prior_slices'] = None


    output_dict, layers_dict = model.pgb_network(
                  placeholder_dict[common.IMAGE],
                  model_options=model_options,
                  labels=placeholder_dict[common.LABEL],
                  prior_imgs=placeholder_dict[common.PRIOR_IMGS],
                  prior_segs=placeholder_dict[common.PRIOR_SEGS],
                  num_classes=dataset.num_of_classes,
                  num_slices=placeholder_dict[common.NUM_SLICES],
                  prior_slices=placeholder_dict['prior_slices'],
                  batch_size=FLAGS.eval_batch_size,
                  z_label_method=FLAGS.z_label_method,
                  zero_guidance=FLAGS.zero_guidance,
                  fusion_rate=FLAGS.fusion_rate,
                  # weight_decay=FLAGS.weight_decay,
                  is_training=False,
                  # fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                  )
    guidance_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'guid' in dict_key}
    feature_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'feature' in dict_key}
    sram_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'sram' in dict_key}

    # Add name to graph node so we can add to summary.
    logits = output_dict[common.OUTPUT_TYPE]
    predictions = tf.nn.softmax(logits)
    predictions = tf.identity(predictions, name=common.OUTPUT_TYPE)
    predictions = tf.argmax(predictions, axis=3)
    predictions = tf.cast(predictions, tf.int32)
    pred_flat = tf.reshape(predictions, shape=[-1,])

    labels = tf.squeeze(placeholder_dict[common.LABEL], axis=3)
    label_onehot = tf.one_hot(indices=labels,
                              depth=dataset.num_of_classes,
                              on_value=1,
                              off_value=0,
                              axis=3)
    num_fg_pixel = tf.reduce_sum(label_onehot, axis=[1,2]) 
    labels_flat = tf.reshape(labels, shape=[-1,])

    guidance = output_dict[common.GUIDANCE]
    if common.OUTPUT_Z in output_dict:
      z_mse = tf.losses.mean_squared_error(placeholder_dict[common.Z_LABEL], output_dict[common.OUTPUT_Z])
      
    # Define the evaluation metric.
    predictions_tag = 'miou'
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_flat, labels_flat, num_classes=dataset.num_of_classes,
                                                            )
    tf.summary.scalar(predictions_tag, mIoU)

    # Define Confusion Maxtrix
    cm = tf.confusion_matrix(labels_flat, pred_flat, num_classes=dataset.num_of_classes)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.contrib.training.SummaryAtEndHook(
        log_dir=FLAGS.eval_logdir, summary_op=summary_op)
    hooks = [summary_hook]

    num_eval_iters = None
    if FLAGS.max_number_of_evaluations > 0:
      num_eval_iters = FLAGS.max_number_of_evaluations

    # Set up tf session and initialize variables.
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # Load weights.
    loader = tf.train.Saver()
    if FLAGS.checkpoint_dir is not None:
        load_model(loader, sess, FLAGS.checkpoint_dir)
    else:
        raise ValueError("model checkpoint not exist")


    cm_total = 0
    total_eval_z = 0
    DSC_slice = []
    total_resnet_activate = {}
    z_pred_label = []

    # Build up Pyplot displaying tool
    show_seg_results = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir,
                                                        is_showfig=False,
                                                        is_savefig=False,
                                                        subplot_split=(1,3),
                                                        type_list=3*['img'])
    # Start Evaluate
    # TODO: The order of subject
    for i in range(dataset.splits_to_sizes):
        data = sess.run(samples)
        _feed_dict = {placeholder_dict[k]: v for k, v in data.items() if k in placeholder_dict}
        print('Sample {} Slice {}'.format(i, data[common.DEPTH][0]))
          
        # Segmentation Evaluation
        cm_slice, pred = sess.run([cm, predictions], feed_dict=_feed_dict)
        _, dscs = eval_utils.compute_mean_dsc(cm_slice)
        DSC_slice.append(dscs)
        cm_total += cm_slice

        show_seg_results.display_figure(FLAGS.eval_split+'_pred{}'.format(i),
                                        [data[common.IMAGE][0,...,0], data[common.LABEL][0,...,0], pred[0]])
        foreground_pixel = sess.run(num_fg_pixel, _feed_dict)
        # Z-information Evaluation
        if common.OUTPUT_Z in output_dict:
          eval_z, z_pred = sess.run([z_mse, output_dict[common.OUTPUT_Z]], feed_dict=_feed_dict)
          z_label = data[common.Z_LABEL]
          z_pred_label.append((z_label, z_pred))
          total_eval_z += eval_z

        # Guidance Visualization
        if FLAGS.vis_guidance:
          guid_dict = sess.run(guidance_dict, feed_dict=_feed_dict)
          
        # Features Visualization
        if FLAGS.vis_features:
          # features, sram_layers = sess.run([feature_dict, sram_dict], feed_dict=_feed_dict)
          pass

        # activate = np.sum(features["resnet_feature"], axis=1)
        # activate = np.sum(activate, axis=1)
        # zero_activate = np.sum(activate==0)
        # total_resnet_activate[str(i)] = (zero_activate, activate)

        # fig, ax = plt.subplots()
        # ax.imshow(guid_dict['fused_guidance'][0,...,0])
        # fig.savefig(FLAGS.eval_logdir+'sample{}_fused_guidance.png'.format(i))
        # plt.close(fig)

        # if i in [0,70,80,90,100,110,120,130,140]:
        #   for kernel in range(32):
        #     fig, (ax1,ax2) = plt.subplots(1,2)
        #     ax1.imshow(sram_layers['RM_2/sram_embeded'][0,...,kernel])
        #     ax2.imshow(sram_layers['RM_2/sram_output_class7'][0,...,kernel])
        #     ax1.set_title('RM_2/input_{}'.format(kernel))
        #     ax2.set_title('RM_2/sram_output_class7_{}'.format(kernel))
        #     fig.savefig(FLAGS.eval_logdir+'sample{}_rm{}_kernel{}_clsass{}.png'.format(i,2,kernel, 7))
        #     plt.close(fig)

        #     fig, (ax1,ax2) = plt.subplots(1,2)
        #     ax1.imshow(sram_layers['RM_3/sram_embeded'][0,...,kernel])
        #     ax2.imshow(sram_layers['RM_3/sram_output_class7'][0,...,kernel])
        #     ax1.set_title('RM_3/input_{}'.format(kernel))
        #     ax2.set_title('RM_3/sram_output_class7_{}'.format(kernel))
        #     fig.savefig(FLAGS.eval_logdir+'sample{}_rm{}_kernel{}_clsass{}.png'.format(i,3,kernel, 7))
        #     plt.close(fig)

    # for step in range(668):
    #     data = sess.run(samples)
    #     _feed_dict = {placeholder_dict[k]: v for k, v in data.items() if k in placeholder_dict}
    #     preds, _ = sess.run([predictions, update_op], feed_dict=_feed_dict)
    #     if step % 100 == 0:
    #         print('step {:d}'.format(step))
    # print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))


    mean_iou = eval_utils.compute_mean_iou(cm_total)
    mean_dice_score, dice_score = eval_utils.compute_mean_dsc(cm_total)
    pixel_acc = eval_utils.compute_accuracy(cm_total)
    # TODO: save instead of showing
    # eval_utils.plot_confusion_matrix(cm_total, classes=np.arange(dataset.num_of_classes), normalize=True,
    #                                  title='Confusion matrix, without normalization')
    
    if common.Z_LABEL in samples:
      
      total_eval_z /= 668
      print("MSE of z prediction {}".format(total_eval_z))
    
    if FLAGS.display_box_plot:
        show_seg_results = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir,
                                                            is_showfig=False,
                                                            is_savefig=False,
                                                            subplot_split=(1,1),
                                                            type_list=['plot'])
        # box_plot_figure(DSC_slice)
        
    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.
    #     TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    # tf.contrib.tfprof.model_analyzer.print_model_analysis(
    #     tf.get_default_graph(),
    #     tfprof_options=tf.contrib.tfprof.model_analyzer.FLOAT_OPS_OPTIONS)
    # tf.contrib.training.evaluate_repeatedly(
    #     master=FLAGS.master,
    #     checkpoint_dir=FLAGS.checkpoint_dir,
    #     eval_ops=[miou, update_op],
    #     max_number_of_evaluations=num_eval_iters,
    #     hooks=hooks,
    #     eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
    
    
    
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)