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

import os
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import common
import model
from datasets import data_generator
from utils import eval_utils
import experiments
from test_flownet import FlowNetS, WarpingLayer
import math
from core import utils, features_extractor
spatial_transfom_exp = experiments.spatial_transfom_exp

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EVAL_CROP_SIZE = [256,256]
# EVAL_CROP_SIZE = [512,512]
ATROUS_RATES = None
# TODO: Multi-Scale Test
# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
EVAL_SCALES = [1.0]
HU_WINDOW = [-125, 275]
IMG_LIST = [50,60, 61, 62, 63, 64, 80, 81, 82, 83, 84,220,221,222,223,224,228,340,350,480,481,482,483,484,495]
IMG_LIST = []
# IMG_LIST = [60, 64, 70, 82, 222, 227, 350, 481]

THRESHOLD = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# THRESHOLD = [0.5, 0.9]

CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_042/model.ckpt-40000' # fusion, gradually, meandsc loss, no feature adding
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_043/model.ckpt-30000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_057/model.ckpt-30000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_058/model.ckpt-30000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_059/model.ckpt-30000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_002/model.ckpt-30000'

DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord/'
PRIOR_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/'

parser = argparse.ArgumentParser()

parser.add_argument('--master', type=str, default='',
                    help='')

# Settings for log directories.
parser.add_argument('--eval_logdir', type=str, default=CHECKPOINT+'eval/',
                    help='')

parser.add_argument('--prior_dir', type=str, default=PRIOR_PATH,
                    help='')

parser.add_argument('--checkpoint_dir', type=str, default=CHECKPOINT,
                    help='')

# Settings for evaluating the model.
parser.add_argument('--drop_prob', type=float, default=None,
                    help='')

parser.add_argument('--guid_weight', type=bool, default=False,
                    help='')

parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='')

parser.add_argument('--eval_interval_secs', type=int, default=5,
                    help='')

parser.add_argument('--output_stride', type=int, default=8,
                    help='')

parser.add_argument('--prior_num_slice', type=int, default=1,
                    help='')

parser.add_argument('--prior_num_subject', type=int, default=20,
                    help='')

parser.add_argument('--fusion_slice', type=int, default=3,
                    help='')

parser.add_argument('--guidance_type', type=str, default="training_data_fusion",
                    help='')
                    
# Change to True for adding flipped images during test.
parser.add_argument('--add_flipped_images', type=bool, default=False,
                    help='')

parser.add_argument('--z_label_method', type=str, default=None,
                    help='')

parser.add_argument('--affine_transform', type=bool, default=False,
                    help='')

parser.add_argument('--deformable_transform', type=bool, default=False,
                    help='')

parser.add_argument('--zero_guidance', type=bool, default=False,
                    help='')

parser.add_argument('--vis_guidance', type=bool, default=False,
                    help='')

parser.add_argument('--vis_features', type=bool, default=False,
                    help='')

parser.add_argument('--display_box_plot', type=bool, default=False,
                    help='')

parser.add_argument('--store_all_imgs', type=bool, default=False,
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

# Exp.
# TODO: dx, dy should get together, tensorflow argument
parser.add_argument('--stn_exp_angle', type=int, default=None,
                    help='')

parser.add_argument('--stn_exp_dx', type=int, default=None,
                    help='')

parser.add_argument('--stn_exp_dy', type=int, default=None,
                    help='')

parser.add_argument('--learning_cases', type=str, default="img-prior",
                    help='')

parser.add_argument('--threshold', type=float, default=0.5,
                    help='')

parser.add_argument('--model_variant', type=str, default="unet",
                    help='')


# TODO: boxplot
# TODO: MSE for z information
# TODO: Automatically output --> mean_iou, mean_dsc, pixel_acc as text file,
# TODO: image, label, prediciton for subplots and full image

# TODO: Warning for dataset_split, guidance and procedure of guidance, MSE for z information
# TODO: testing mode for online eval
# TODO: add run_xxx in  feature, guidance folder name

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
                affine_transform=FLAGS.affine_transform,
                deformable_transform=FLAGS.deformable_transform,
                batch_size=1,
                HU_window=HU_WINDOW,
                z_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                z_class=FLAGS.prior_num_slice,
                crop_size=EVAL_CROP_SIZE,
                min_resize_value=EVAL_CROP_SIZE[0],
                max_resize_value=EVAL_CROP_SIZE[0],
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
                prior_dir=FLAGS.prior_dir)             
  # TODO: make dirs?
  # TODO: Add model name in dir to distinguish
  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default() as graph:
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
    threshold_placeholder = tf.placeholder(tf.float32)
    
    placeholder_dict = {common.IMAGE: image_placeholder,
                        common.LABEL: label_placeholder,
                        common.NUM_SLICES: num_slices_placeholder,
                        "threshold": threshold_placeholder}

    if common.Z_LABEL in samples:
      samples[common.Z_LABEL] = tf.identity(samples[common.Z_LABEL], name=common.Z_LABEL)
      z_label_placeholder = tf.placeholder(tf.float32, shape=[None])
      placeholder_dict[common.Z_LABEL] = z_label_placeholder
    else:
      placeholder_dict[common.Z_LABEL] = None

    if common.PRIOR_IMGS in samples:
      samples[common.PRIOR_IMGS] = tf.identity(samples[common.PRIOR_IMGS], name=common.PRIOR_IMGS)
      prior_img_placeholder = tf.placeholder(tf.float32,
                                           shape=[None, EVAL_CROP_SIZE[0],
                                                  EVAL_CROP_SIZE[1], None])
      placeholder_dict[common.PRIOR_IMGS] = prior_img_placeholder
    else:
      placeholder_dict[common.PRIOR_IMGS] = None

    prior_seg_placeholder = tf.placeholder(tf.float32,shape=[None, EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1], 14])
    placeholder_dict[common.PRIOR_SEGS] = prior_seg_placeholder
    # if common.PRIOR_SEGS in samples:
    #   samples[common.PRIOR_SEGS] = tf.identity(samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
    #   prior_seg_placeholder = tf.placeholder(tf.float32,
    #                                        shape=[None, EVAL_CROP_SIZE[0],
    #                                               EVAL_CROP_SIZE[1], dataset.num_of_classes])
    #   placeholder_dict[common.PRIOR_SEGS] = prior_seg_placeholder
    # else:
    #   placeholder_dict[common.PRIOR_SEGS] = None

    if 'prior_slices' in samples:
      prior_slices_placeholder = tf.placeholder(tf.int64, shape=[None])
      placeholder_dict['prior_slices'] = prior_slices_placeholder
    else:
      placeholder_dict['prior_slices'] = None

    placeholder_dict['organ_label'] = tf.placeholder(tf.int32, shape=[None,dataset.num_of_classes])
    # guidance = tf.convert_to_tensor(np.load("/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/training_seg_merge_010.npy"))
    # guidance = tf.expand_dims(guidance, axis=0)
    
    if FLAGS.guidance_type == "gt":
      if FLAGS.stn_exp_angle is not None:
        angle = FLAGS.stn_exp_angle * math.pi / 180
      else:
        angle = None
      if FLAGS.stn_exp_dx is not None and FLAGS.stn_exp_dy is not None:
        translations = [FLAGS.stn_exp_dx,FLAGS.stn_exp_dy]
      else:
        translations  = None
      samples[common.PRIOR_SEGS] = spatial_transfom_exp(samples[common.LABEL], angle, 
                                                  translations, "NEAREST")


    
    translations = tf.random_uniform([], minval=-10,maxval=10,dtype=tf.float32)
    angle = tf.random_uniform([], minval=-10,maxval=10,dtype=tf.float32)
    angle = angle * math.pi / 180
    transform_images = spatial_transfom_exp(samples[common.IMAGE], angle, 
                                                [translations,0], "BILINEAR")
    transform_labels = spatial_transfom_exp(samples[common.LABEL], angle, 
                                                [translations,0], "NEAREST")
    #   transform_images = samples[common.IMAGE]
    #   transform_labels = samples[common.LABEL]
    net = FlowNetS()
    if FLAGS.learning_cases == "img-img":
        input_a, input_b, query = samples[common.IMAGE], transform_images, samples[common.IMAGE]
    elif FLAGS.learning_cases == "seg-seg":
        input_a, input_b, query = samples[common.LABEL], transform_labels, samples[common.LABEL]
    elif FLAGS.learning_cases == "img-seg":
        input_a, input_b, query = samples[common.IMAGE], transform_images, samples[common.LABEL]
    elif FLAGS.learning_cases == "img-prior":
        input_a, input_b, query = placeholder_dict[common.IMAGE], placeholder_dict[common.PRIOR_SEGS], placeholder_dict[common.PRIOR_SEGS]
            
    inputs = {"input_a": input_a, "input_b": input_b, "query": query}
    training_schedule = {
            # 'step_values': [400000, 600000, 800000, 1000000],
            'step_values': [400000, 600000, 800000, 1000000],
            'learning_rates': [0.0001, 0.00005, 0.000025, 0.0000125, 0.00000625],
            'momentum': 0.9,
            'momentum2': 0.999,
            'weight_decay': 0.0004,
            'max_iter': 120000,
        }
    with tf.variable_scope("flow_model"):
      if FLAGS.model_variant == "unet":
        concat_inputs = tf.concat([inputs['input_a'], inputs['input_b']], axis=3)
        flow = utils._simple_unet(concat_inputs, out=2, stage=3, channels=32, is_training=True)
      elif FLAGS.model_variant == "FlowNet-S":
        flow_dict = net.model(inputs, training_schedule, trainable=True)
        flow = flow_dict["flow"]
      elif FLAGS.model_variant == "resnet_decoder":
        features, _ = features_extractor.extract_features(images=samples[common.IMAGE],
                                                                  output_stride=FLAGS.output_stride,
                                                                  multi_grid=model_options.multi_grid,
                                                                  model_variant=model_options.model_variant,
                                                                  reuse=tf.AUTO_REUSE,
                                                                  is_training=False,
                                                                  fine_tune_batch_norm=model_options.fine_tune_batch_norm,
                                                                  preprocessed_images_dtype=model_options.preprocessed_images_dtype)

        concat_inputs = tf.concat([features, inputs['input_b']], axis=3)
        flow = utils._simple_decoder(concat_inputs, out=2, stage=3, channels=32, is_training=False)
        
    #   pred = stn.bilinear_sampler(query, flow[...,0], flow[...,1])
    if FLAGS.learning_cases.split("-")[1] in ("img", "prior"):
        warp_func = WarpingLayer('bilinear')
    elif FLAGS.learning_cases.split("-")[1] == "seg":
        warp_func = WarpingLayer('nearest')
        
    pred = warp_func(query, flow)
    output_dict = {common.OUTPUT_TYPE: pred,
                    "flow": flow}
    
    
    # Add name to graph node so we can add to summary.
    before_transform = query
    after_transform = pred
    pred_b = tf.reshape(before_transform, shape=[-1,])
    
    assert len(THRESHOLD) > 0
    cm_list = []
    for c in range(dataset.num_of_classes):
      pred_a = pred[...,c]
      
      one = tf.ones_like(pred_a)
      zero = tf.zeros_like(pred_a)
      pred_a = tf.where(pred_a>placeholder_dict["threshold"], x=one, y=zero)
      pred_a_flat = tf.reshape(pred_a, shape=[-1,])  
      
      labels = tf.squeeze(placeholder_dict[common.LABEL], axis=3)
      label_onehot = tf.one_hot(indices=labels,
                                depth=dataset.num_of_classes,
                                on_value=1,
                                off_value=0,
                                axis=3)
      labels_flat = tf.reshape(label_onehot[...,c], shape=[-1,])
      cm = tf.confusion_matrix(labels_flat, pred_a_flat, num_classes=2)
      cm_list.append(cm)
    cm_class = tf.stack(cm_list)  
      
    # assign_c = tf.placeholder(tf.int32, shape=[None])
    # assign_c = 6
    # iou = tf.compat.v1.metrics.mean_iou(label_onehot[...,assign_c], pred_a[...,assign_c], num_classes=2)
    
    
    
    # # Define Confusion Maxtrix
    # cm_a = tf.confusion_matrix(labels_flat, pred_a, num_classes=dataset.num_of_classes)
    # cm_b = tf.confusion_matrix(labels_flat, pred_b, num_classes=dataset.num_of_classes)
    
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
    
    
    cm_total_a = 0
    cm_total_b = 0
    xx = 10*[]
    dsc_in_diff_th = [[] for _ in range(dataset.num_of_classes)]
    _cm_g1_t, _cm_g2_t, _cm_g3_t, _cm_g4_t = 0, 0, 0, 0
    total_eval_z = 0
    foreground_pixel = 0
    DSC_slice = []
    total_resnet_activate = {}
    total_z_label = []
    total_z_pred = []

    # Build up Pyplot displaying tool
    # TODO: Image could only show once
    show_seg_results = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir,
                                                        is_showfig=False,
                                                        is_savefig=True,
                                                        subplot_split=(1,3),
                                                        type_list=3*['img'])
    
    show_transform_results = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir,
                                                              is_showfig=False,
                                                              is_savefig=True,
                                                              subplot_split=(1,3),
                                                              type_list=3*['img'])

    if FLAGS.store_all_imgs:
        display_imgs = np.arange(dataset.splits_to_sizes[FLAGS.eval_split])
    else:
        display_imgs = IMG_LIST
        
    # TODO:         
    for idx, th in enumerate(THRESHOLD):
      cm_total = 0
      for i in range(dataset.splits_to_sizes[FLAGS.eval_split]):
          data = sess.run(samples)
          _feed_dict = {placeholder_dict[k]: v for k, v in data.items() if k in placeholder_dict}
          _feed_dict[placeholder_dict["threshold"]] = th
          print('Sample {} Slice {}'.format(i, data[common.DEPTH][0]))
          
          # Segmentation Evaluation
          cm_slice = sess.run(cm_class, feed_dict=_feed_dict)
          cm_total += cm_slice
          
          
          if idx==0 and i in display_imgs:
            # Transform Comparision
            pred_a, pred_b = sess.run([after_transform, before_transform], feed_dict=_feed_dict)
            show_transform_results.set_title(["before", "after", "ground_truth"])
            show_transform_results.set_axis_off()
            for c in range(14):
              show_transform_results.display_figure(FLAGS.eval_split+'-transform_compare-%04d-%03d-%02f' %(i,c,th),
                                              [pred_b[0,...,c], 
                                               pred_a[0,...,c], 
                                               np.int32(data[common.LABEL][0,...,0]==c)],
                                              parameters=None)
            
            # Flow Visualization
      m_dsc = []
      for c in range(1, dataset.num_of_classes):      
        mean_dice_score, dice_score = eval_utils.compute_mean_dsc(cm_total[c])
        dsc_in_diff_th[c-1].append(dice_score[1])
        m_dsc.append(dice_score[1])
      dsc_in_diff_th[dataset.num_of_classes-1].append(sum(m_dsc)/len(m_dsc))
        # dsc_in_diff_th[th] = dice_score
    # for c in range(14):
    #   print(30*str(c))
    #   print(20*"=", "Before Transform Segmentation Evaluation", 20*"=")
    #   mean_iou = eval_utils.compute_mean_iou(cm_total[c])
    #   mean_dice_score, dice_score = eval_utils.compute_mean_dsc(cm_total[c])
    #   pixel_acc = eval_utils.compute_accuracy(cm_total[c])
      
      # _, _ = eval_utils.precision_and_recall(cm_total[c])
    

    # Eval of performance trend

    eval_utils.eval_flol_model(dsc_in_diff_th, THRESHOLD)

    # print(10*"=", "After Transform Segmentation Evaluation", 10*"=")
    # mean_iou = eval_utils.compute_mean_iou(cm_total_a)
    # mean_dice_score, dice_score = eval_utils.compute_mean_dsc(cm_total_a)
    # pixel_acc = eval_utils.compute_accuracy(cm_total_a)
    # _, _ = eval_utils.precision_and_recall(cm_total_a)
    
    return mean_dice_score
  
if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
    