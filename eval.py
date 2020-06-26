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
from utils import eval_utils, train_utils
import experiments
import cv2
import math
spatial_transfom_exp = experiments.spatial_transfom_exp

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

EVAL_CROP_SIZE = [256,256]
EVAL_CROP_SIZE = [512,512]
ATROUS_RATES = None
# TODO: Multi-Scale Test
# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
EVAL_SCALES = [1.0]
HU_WINDOW = [-125, 275]
IMG_LIST = [50,60, 61, 62, 63, 64, 80, 81, 82, 83, 84,220,221,222,223,224,228,340,350,480,481,482,483,484,495]
# TODO: train image list (optional)
# TODO: if dir not exist. Don't build new one
IMG_LIST = [50, 60, 64, 70, 82, 222,226, 227, 228, 350, 481]

FUSIONS = 5*["sum"]
FUSIONS = 5*["guid_uni"]
EVAL_SPLIT = ["val"]

SEG_LOSS = "softmax_dice_loss"
GUID_LOSS = "softmax_dice_loss"
GUID_LOSS = "sigmoid_cross_entropy"
STAGE_PRED_LOSS = "softmax_dice_loss"
STAGE_PRED_LOSS = "sigmoid_cross_entropy"
SEG_WEIGHT_FLAG = False

CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_000/model.ckpt-140000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-110000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_004/model.ckpt-140000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/guid_1.0_uni_binary_convfuse/model.ckpt-140000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-30000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/guid_bug_fix_aligne_false/model.ckpt-200000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-80000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/guid_bug_fix_align_corner_false_entropy/model.ckpt-200000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_002/model.ckpt-200000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/guid_2convs_in_sram/model.ckpt-200000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_010/model.ckpt-200000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_012/model.ckpt-165000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_014/model.ckpt-145000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_003/model.ckpt-165000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_017/model.ckpt-200000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_014/model.ckpt-195000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_018/model.ckpt-200000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/118_run_013/model.ckpt-160000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_000/model.ckpt-168000'

# CHECKPOINT = None

DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord/'
PRIOR_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/'


parser = argparse.ArgumentParser()

parser.add_argument('--predict_without_background', type=bool, default=False,
                    help='')

parser.add_argument('--fuse_flag', type=bool, default=True,
                    help='')

parser.add_argument('--guid_encoder', type=str, default="early",
                    help='')

parser.add_argument('--guid_method', type=str, default=None,
                    help='')

parser.add_argument('--out_node', type=int, default=32,
                    help='')

parser.add_argument('--guid_conv_type', type=str, default="conv",
                    help='')
                    
parser.add_argument('--guid_conv_nums', type=int, default=2,
                    help='')

parser.add_argument('--share', type=bool, default=True,
                    help='')

parser.add_argument('--guidance_acc', type=str, default=None,
                    help='')

parser.add_argument('--master', type=str, default='',
                    help='')

# Settings for log directories.
parser.add_argument('--eval_logdir', type=str, default=CHECKPOINT+'-eval/',
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

parser.add_argument('--eval_split', type=str, default='train-val',
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
  # from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
  # import os
  # checkpoint_path = FLAGS.checkpoint_dir

  # # List ALL tensors example output: v0/Adam (DT_FLOAT) [3,3,1,80]
  # print_tensors_in_checkpoint_file(file_name=checkpoint_path, tensor_name='', all_tensors=False, 
  #                                  all_tensor_names=True)
  tf.gfile.MakeDirs(FLAGS.eval_logdir)
  
  parameters_dict = vars(FLAGS)
  with open(os.path.join(FLAGS.eval_logdir, 'eval_logging.txt'), 'w') as f:
    f.write("Start Evaluation\n")
    f.write(60*"="+"\n")
    for key in parameters_dict:
      f.write( "{}: {}".format(str(key), str(parameters_dict[key])))
      f.write("\n")
    f.write("\n")
  
    
  tf.logging.set_verbosity(tf.logging.INFO)

  # TODO:
  model_options = common.ModelOptions(
              outputs_to_num_classes=14,
              crop_size=EVAL_CROP_SIZE,
              output_stride=FLAGS.output_stride)

  dataset = data_generator.Dataset(
                dataset_name=FLAGS.dataset,
                split_name=EVAL_SPLIT,
                dataset_dir=FLAGS.dataset_dir,
                # affine_transform=FLAGS.affine_transform,
                # deformable_transform=FLAGS.deformable_transform,
                batch_size=1,
                HU_window=HU_WINDOW,
                mt_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                mt_class=FLAGS.prior_num_slice,
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
                repeat_data=False,
                prior_num_slice=FLAGS.prior_num_slice,
                prior_num_subject=FLAGS.prior_num_subject,
                prior_dir=FLAGS.prior_dir,
                seq_length=1,
                seq_type="forward")            
  # TODO: make dirs?
  # TODO: Add model name in dir to distinguish
  
  tf.logging.info('Evaluating on %s set', EVAL_SPLIT)

  with tf.Graph().as_default() as graph:
    iterator = dataset.get_one_shot_iterator().make_one_shot_iterator()
    samples = iterator.get_next()
    
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
                                           shape=[None, EVAL_CROP_SIZE[0],
                                                  EVAL_CROP_SIZE[1], None])
      placeholder_dict[common.PRIOR_IMGS] = prior_img_placeholder
    else:
      placeholder_dict[common.PRIOR_IMGS] = None

    if FLAGS.guidance_type == "gt":
      prior_seg_placeholder = tf.placeholder(tf.int32,shape=[None, EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1], 1])
    elif FLAGS.guidance_type in ("training_data_fusion", "training_data_fusion_h"):
      prior_seg_placeholder = tf.placeholder(tf.float32,shape=[None, EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1], 14, 1])
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
    
    # if FLAGS.guidance_type == "gt":
    #   if FLAGS.stn_exp_angle is not None:
    #     angle = FLAGS.stn_exp_angle * math.pi / 180
    #   else:
    #     angle = None
    #   if FLAGS.stn_exp_dx is not None and FLAGS.stn_exp_dy is not None:
    #     translations = [FLAGS.stn_exp_dx,FLAGS.stn_exp_dy]
    #   else:
    #     translations  = None
    #   samples[common.PRIOR_SEGS] = spatial_transfom_exp(samples[common.LABEL], angle, 
    #                                               translations, "NEAREST")

    if FLAGS.guid_method is not None:
      FUSIONS[0] = FLAGS.guid_method

    output_dict, layers_dict = model.pgb_network(
                placeholder_dict[common.IMAGE],
                model_options=model_options,
                affine_transform=FLAGS.affine_transform,
                # deformable_transform=FLAGS.deformable_transform,
                # labels=placeholder_dict[common.LABEL],
                samples=placeholder_dict["organ_label"],
                # prior_imgs=placeholder_dict[common.PRIOR_IMGS],
                prior_segs=placeholder_dict[common.PRIOR_SEGS],
                num_class=dataset.num_of_classes,
                # num_slices=placeholder_dict[common.NUM_SLICES],
                # prior_slice=prior_slices,
                batch_size=FLAGS.eval_batch_size,
                z_label_method=FLAGS.z_label_method,
                # z_label=placeholder_dict[common.Z_LABEL],
                # z_class=FLAGS.prior_num_slice,
                guidance_type=FLAGS.guidance_type,
                fusion_slice=FLAGS.fusion_slice,
                prior_dir=FLAGS.prior_dir,
                drop_prob=FLAGS.drop_prob,
                guid_weight=FLAGS.guid_weight,
                stn_in_each_class=True,
                is_training=False,
                weight_decay=0.0,
                # fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                guidance_acc=FLAGS.guidance_acc,
                share=FLAGS.share,
                fusions=FUSIONS,
                out_node=FLAGS.out_node,
                guid_encoder=FLAGS.guid_encoder,
                guidance_loss=GUID_LOSS,
                stage_pred_loss=STAGE_PRED_LOSS,
                guid_conv_nums=FLAGS.guid_conv_nums,
                guid_conv_type=FLAGS.guid_conv_type,
                fuse_flag=FLAGS.fuse_flag,
                predict_without_background=FLAGS.predict_without_background,
                reuse=tf.AUTO_REUSE,
                apply_sram2=True,
                )
    
    if FLAGS.vis_guidance:      
      if "softmax" in GUID_LOSS:
        guidance_dict = {dict_key: tf.nn.softmax(layers_dict[dict_key],3) for dict_key in layers_dict if 'guidance' in dict_key}
        pred_dict = {dict_key: tf.arg_max(guidance_dict[dict_key],3) for dict_key in guidance_dict}
        guidance_dict["guidance0"] = tf.nn.softmax(output_dict[common.GUIDANCE], 3)
        pred_dict["guidance0"] = tf.argmax(guidance_dict["guidance0"], 3)
      elif "sigmoid" in GUID_LOSS:
        guidance_dict = {dict_key: tf.nn.sigmoid(layers_dict[dict_key]) for dict_key in layers_dict if 'guidance' in dict_key}      
        pred_dict = guidance_dict
        guidance_dict["guidance0"] = tf.nn.sigmoid(output_dict[common.GUIDANCE])
        pred_dict["guidance0"] = guidance_dict["guidance0"]

    # Add name to graph node so we can add to summary.
    logits = output_dict[common.OUTPUT_TYPE]
    preds = tf.nn.softmax(logits)
    predictions = tf.identity(preds, name=common.OUTPUT_TYPE)
    predictions = tf.argmax(predictions, axis=3)
    predictions = tf.cast(predictions, tf.int32)
    pred_flat = tf.reshape(predictions, shape=[-1,])

    labels = tf.squeeze(placeholder_dict[common.LABEL], axis=3)
    label_onehot = tf.one_hot(indices=labels,
                              depth=dataset.num_of_classes,
                              on_value=1.0,
                              off_value=0.0,
                              axis=3)
    num_fg_pixel = tf.reduce_sum(label_onehot, axis=[1,2]) 
    labels_flat = tf.reshape(labels, shape=[-1,])

    kernel2 = tf.ones((6, 6, dataset.num_of_classes))
    # kernel2 = tf.cast(kernel2, tf.int32)
    kernel3 = tf.ones((7, 7, dataset.num_of_classes))
    # kernel3 = tf.cast(kernel3, tf.int32)
    kernel4 = tf.ones((8, 8, dataset.num_of_classes))
    # kernel4 = tf.cast(kernel4, tf.int32)
    
    label2 = tf.nn.dilation2d(label_onehot, filter=kernel2, strides=(1,1,1,1), 
                              rates=(1,1,1,1), padding="SAME")
    label2 = label2 - tf.ones_like(label2)                          
    label3 = tf.nn.dilation2d(label_onehot, filter=kernel3, strides=(1,1,1,1), 
                              rates=(1,1,1,1), padding="SAME")
    label3 = label3 - tf.ones_like(label3)                          
    label4 = tf.nn.dilation2d(label_onehot, filter=kernel4, strides=(1,1,1,1), 
                              rates=(1,1,1,1), padding="SAME")
    label4 = label4 - tf.ones_like(label4)                          
    image_128 = tf.image.resize_bilinear(samples[common.IMAGE],[128, 128], align_corners=True)
    # image_diff = samples[common.IMAGE] - image_128

    # if FLAGS.vis_guidance:
    #   def guid_mean_dsc(logits, label):
    #     h, w = label.get_shape().as_list()[1:3]
    #     logits = tf.compat.v2.image.resize(logits, [h, w])
    #     loss = train_utils.loss_utils(logits, label, "mean_dice_coefficient")
    #     return 1 - loss
    #   guid0 = layers_dict["guidance_in"]
    #   guid1 = layers_dict["guidance1"]
    #   guid2 = layers_dict["guidance2"]
    #   guid3 = layers_dict["guidance3"]
    #   guid4 = layers_dict["guidance4"]

    #   guid1_dsc = guid_mean_dsc(logits, labels)

    #   guid_list = tf.get_collection("guidance")
                       
                 
    if common.OUTPUT_Z in output_dict:
      z_mse = tf.losses.mean_squared_error(placeholder_dict[common.Z_LABEL], output_dict[common.OUTPUT_Z])
      
    # Define the evaluation metric.
    predictions_tag = 'miou'
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred_flat, labels_flat, num_classes=dataset.num_of_classes,
                                                            )
    tf.summary.scalar(predictions_tag, mIoU)

    # Define Confusion Maxtrix
    cm = tf.confusion_matrix(labels_flat, pred_flat, num_classes=dataset.num_of_classes)

    # ggsmida = tf.get_collection("f")
    # if FLAGS.vis_guidance:
    #   def cm_in_each_stage(pred, label):
    #     h, w = pred.get_shape().as_list()[1:3]
    #     label =tf.expand_dims(label, axis=3)
    #     label = tf.compat.v2.image.resize(label, [h, w], method='nearest')
        
    #     label_flat = tf.reshape(label, shape=[-1,])
    #     pred_flat = tf.reshape(pred, shape=[-1,])
    #     cm = tf.confusion_matrix(label_flat, pred_flat, num_classes=dataset.num_of_classes)
    #     return cm, label
    #   cm_g1, label_g1 = cm_in_each_stage(pred_stage1, labels)
    #   cm_g2, label_g2 = cm_in_each_stage(pred_stage2, labels)
    #   cm_g3, label_g3 = cm_in_each_stage(pred_stage3, labels)
    #   cm_g4, label_g4 = cm_in_each_stage(pred_stage4, labels)
      # guid_pred = [pred_stage1, pred_stage2, pred_stage3, pred_stage4]
      # guid_label = [label_g1, label_g2, label_g3, label_g4]
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
    _cm_g1_t, _cm_g2_t, _cm_g3_t, _cm_g4_t = 0, 0, 0, 0
    h_min_total = []
    w_min_total = []
    h_max_total = []
    w_max_total = []
    g1_dsc_t = 0
    total_eval_z = 0
    foreground_pixel = 0
    DSC_slice = []
    total_resnet_activate = {}
    total_z_label = []
    total_z_pred = []

    # TODO: Determine saving path while display? remove the dependency then show_guidance and show_seg_results could be the same one
    if FLAGS.vis_guidance:
        if not os.path.isdir(FLAGS.eval_logdir+"guidance"):
          os.mkdir(FLAGS.eval_logdir+"guidance")
          
        show_guidance = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir+"guidance/",
                                                            is_showfig=False,
                                                            is_savefig=True,
                                                            subplot_split=(1,3),
                                                            type_list=3*['img'])
    # Build up Pyplot displaying tool
    # TODO: Image could only show once
    show_seg_results = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir,
                                                        is_showfig=False,
                                                        is_savefig=True,
                                                        subplot_split=(1,3),
                                                        type_list=3*['img'])
    # Start Evaluate
    # TODO: The order of subject
    
    if FLAGS.vis_features:
        if not os.path.isdir(FLAGS.eval_logdir+"feature"):
            os.mkdir(FLAGS.eval_logdir+"feature")
            
        show_feature = eval_utils.Build_Pyplot_Subplots(saving_path=FLAGS.eval_logdir+"feature/",
                                                            is_showfig=False,
                                                            is_savefig=True,
                                                            subplot_split=(1,3),
                                                            type_list=3*['img'])
        
    sram_conv = tf.get_collection("/sram_embed")      
    if FLAGS.store_all_imgs:
        display_imgs = np.arange(dataset.splits_to_sizes[EVAL_SPLIT])
    else:
        display_imgs = IMG_LIST

    flops, params = eval_utils.compute_params_and_flops(graph)
    with open(os.path.join(FLAGS.eval_logdir, 'eval_logging.txt'), 'a') as f:
      f.write("\nFLOPs: {}".format(flops))
      f.write("\nGFLOPs: {}".format(flops/1e9))
      f.write("\nParameters: {} MB".format(params))
      f.write("\n")  
    
    num_class = dataset.num_of_classes
    if FLAGS.predict_without_background:
      num_class -= 1 
       
    for split_name in EVAL_SPLIT:
      num_sample = dataset.splits_to_sizes[split_name]
      
      for i in range(num_sample):
          data = sess.run(samples)
          _feed_dict = {placeholder_dict[k]: v for k, v in data.items() if k in placeholder_dict}
          print('Sample {} Slice {}'.format(i, data[common.DEPTH][0]))
          
          # Segmentation Evaluation
          cm_slice, pred, l2,l3,l4 = sess.run([cm, predictions, label2, label3, label4], feed_dict=_feed_dict)
          _, dscs = eval_utils.compute_mean_dsc(cm_slice)
          DSC_slice.append(dscs)
          cm_total += cm_slice

          
          if i in display_imgs:
              parameters = [{"cmap": "gray"}]
              parameters.extend(2*[{"vmin": 0, "vmax": dataset.num_of_classes}])
              show_seg_results.set_title(["image", "label","prediction"])
              show_seg_results.set_axis_off()
              show_seg_results.display_figure(split_name+'_pred_%04d' %i,
                                              [data[common.IMAGE][0,...,0], data[common.LABEL][0,...,0], pred[0]],
                                              parameters=parameters)
              h_min,w_min,h_max,w_max = eval_utils.get_label_range(data[common.LABEL][0], 512, 512)
              
              h_min_total.append(h_min)
              w_min_total.append(w_min)
              h_max_total.append(h_max)
              w_max_total.append(w_max)
              # print(h_min,w_min,h_max,w_max)
              # plt.imshow(data[common.LABEL][0,...,0])
              # plt.show()
              
              # img_128 = sess.run(image_128, feed_dict=_feed_dict)
              
              # img_128_cv = cv2.resize(data[common.IMAGE][0,...,0], (128, 128))
              # img_diff = img_128[0,...,0] - img_128_cv
              # parameters = 3*[{"cmap": "gray"}]
              # show_seg_results.set_title(["image", "down_up","diff"])
              # show_seg_results.set_axis_off()
              # show_seg_results.display_figure(split_name+'_aliasing_test_%04d' %i,
              #                                 [img_128_cv, img_128[0,...,0], img_diff],
              #                                 parameters=parameters)

              # show_seg_results.set_title(["label2", "label3","label4"])
              # show_seg_results.display_figure(split_name+'_dilated_label_%04d' %i,
              #                                 [l2[0,...,6], l3[0,...,6], l4[0,...,6]])
              # # plt.imshow(l2[0,...,6]+l3[0,...,6]+l4[0,...,6]+np.int32(data[common.LABEL][0,...,0]==6))
              # # plt.show()
              # plt.savefig(FLAGS.eval_logdir+"sample{}-label_compare.png".format(i))
              
          # Z-information Evaluation
          if common.OUTPUT_Z in output_dict:
            eval_z, z_pred = sess.run([z_mse, output_dict[common.OUTPUT_Z]], feed_dict=_feed_dict)
            z_label = data[common.Z_LABEL]
            total_z_label.append(z_label)
            total_z_pred.append(z_pred)
            total_eval_z += eval_z

          # Guidance Visualization
          if FLAGS.vis_guidance:
            if i in display_imgs:
              guid_avg = tf.get_collection("guid_avg")
              guid_avgs = sess.run(guid_avg, feed_dict=_feed_dict)     
              layers, pred_layers, gg = sess.run([guidance_dict, pred_dict, guidance_dict["guidance0"]], feed_dict=_feed_dict)
              
              # if i == 64:
              #   for ii in range(5):
              #     plt.imshow(guid_avgs[ii][0,...,0])
              #     plt.show()
                
              show_guidance.set_title(["guid_avg0", "guid_avg1", "guid_avg2"])
              show_guidance.display_figure(split_name+'-guid_avg012-%04d' %i,
                                            [guid_avgs[0][0,...,0],
                                            guid_avgs[1][0,...,0],
                                            guid_avgs[2][0,...,0]])

              show_guidance.set_title(["guid_avg3", "guid_avg4", "guid_avg5"])
              show_guidance.display_figure(split_name+'-guid_avg345-%04d' %i,
                                            [guid_avgs[3][0,...,0],
                                            guid_avgs[4][0,...,0],
                                            guid_avgs[4][0,...,0]])

              # show_guidance.set_title(["pred0", "pred1", "pred2"])
              # show_guidance.display_figure(split_name+'-pred012-%04d' %i,
              #                               [pred_layers["guidance0"][0],
              #                               pred_layers["guidance5"][0],
              #                               pred_layers["guidance4"][0]])

              # show_guidance.set_title(["pred3", "pred4", "pred5"])
              # show_guidance.display_figure(split_name+'-pred345-%04d' %i,
              #                               [pred_layers["guidance3"][0],
              #                               pred_layers["guidance2"][0],
              #                               pred_layers["guidance1"][0]])
                
              for c in range(num_class):
                show_guidance.set_title(["guidance0", "guidance1", "guidance2"])
                show_guidance.display_figure(split_name+'-guid012-%04d-%03d' % (i,c),
                                              [layers["guidance0"][0,...,c],
                                              layers["guidance5"][0,...,c],
                                              layers["guidance4"][0,...,c]])
                
                show_guidance.set_title(["guidance3", "guidance4", "guidance5"])
                show_guidance.display_figure(split_name+'-guid345-%04d-%03d' % (i,c),
                                              [layers["guidance3"][0,...,c],
                                              layers["guidance2"][0,...,c],
                                              layers["guidance1"][0,...,c]])
                
                
                  
                                        
          # Features Visualization
          if FLAGS.vis_features:
            if i in display_imgs:
              sram1, sram2, embed, feature, refining, guid_f = sess.run([tf.get_collection("sram1"), 
                                                                tf.get_collection("sram2"), 
                                                                tf.get_collection("embed"), 
                                                                tf.get_collection("feature"), 
                                                                tf.get_collection("refining"),
                                                                #  tf.get_collection("feature"),
                                                                tf.get_collection("guid_f")], feed_dict=_feed_dict)

              for cc in range(0, 32, 4):
                for w in range(5):
                  filename = "{}-feature1-sample{}-stage{}-feature{}".format(split_name, i, w+1, cc)
                  show_feature.set_title(["embed", "sram1", "feature"])
                  # show_feature.set_axis_off()
                  show_feature.display_figure(filename, [embed[w][0,...,cc], sram1[w][0,...,cc], feature[w][0,...,cc]])
                  
                  filename = "{}-feature2-sample{}-stage{}-feature{}".format(EVAL_SPLIT, i, w+1, cc)
                  show_feature.set_title(["sram1+feature", "sram2", "refine"])
                  # show_feature.set_axis_off()
                  show_feature.display_figure(filename, [sram1[w][0,...,cc]+feature[w][0,...,cc], sram2[w][0,...,cc], refining[w][0,...,cc]])
              
                filename = "{}-guidingf-sample{}-feature{}".format(EVAL_SPLIT, i, cc)
                show_feature.set_title(["guiding_feature{}".format(cc), 
                                        "guiding_feature{}".format(cc+1), 
                                        "guiding_feature{}".format(cc+2)])
                # show_feature.set_axis_off()
                show_feature.display_figure(filename, [guid_f[0][0,...,cc],
                                                      guid_f[0][0,...,cc+1],
                                                      guid_f[0][0,...,cc+2]])
              
              # features, sram_layers = sess.run([feature_dict, sram_dict], feed_dict=_feed_dict)

    def get_list_stats(value):
      val_arr = np.stack(value)
      return np.mean(val_arr), np.std(val_arr), np.min(val_arr), np.max(val_arr)
    h_min_total = [v for v in h_min_total if v!=0]
    h_max_total = [v for v in h_max_total if v!=0]
    w_min_total = [v for v in w_min_total if v!=0]
    w_max_total = [v for v in w_max_total if v!=0]
    
    print("Height Minimum mean: {:5.3f} std: {:5.3f} min: {:5.3f} max: {:5.3f}".format(*get_list_stats(h_min_total)))
    print("Height Maximum mean: {:5.3f} std: {:5.3f} min: {:5.3f} max: {:5.3f}".format(*get_list_stats(h_max_total)))
    print("Width Minimum mean: {:5.3f} std: {:5.3f} min: {:5.3f} max: {:5.3f}".format(*get_list_stats(w_min_total)))
    print("Width Maximum mean: {:5.3f} std: {:5.3f} min: {:5.3f} max: {:5.3f}".format(*get_list_stats(w_max_total)))
    print(10*"=", "Segmentation Evaluation", 10*"=")
    mean_iou = eval_utils.compute_mean_iou(cm_total)
    mean_dice_score, dice_score = eval_utils.compute_mean_dsc(cm_total)
    pixel_acc = eval_utils.compute_accuracy(cm_total)
    p_mean, p_std, r_mean, r_std = eval_utils.precision_and_recall(cm_total)
    # print(foreground_pixel, foreground_pixel/(256*256*dataset.splits_to_sizes[EVAL_SPLIT]))

    with open(os.path.join(FLAGS.eval_logdir, 'eval_logging.txt'), 'a') as f:
      f.write("\nPixel ACC: {:.4f}".format(pixel_acc))
      f.write("\nMean IoU: {:.4f}".format(mean_iou))
      
      for i, dsc in enumerate(dice_score):
          f.write("\n    class {}: {:.4f}".format(i, dsc))
      f.write("\nMean DSC: {:.4f}".format(mean_dice_score))
      f.write("\nPrecision mean: {:.4f} std: {:.4f}".format(p_mean, p_std))
      f.write("\nRecall mean: {:.4f} std: {:.4f}".format(r_mean, r_std))
      f.write("\n")  
      f.write(60*"="+"\n")
      f.write("End Evaluation\n")
    
    # if FLAGS.vis_guidance:
    #   print(10*"=", "Guidance Evaluation", 10*"=")
    #   print(g1_dsc_t, dataset.splits_to_sizes[EVAL_SPLIT])
    #   g1_dsc_t /= dataset.splits_to_sizes[EVAL_SPLIT]
    #   # g1_mdsc, _ = eval_utils.compute_mean_dsc(_cm_g1_t)
    #   # g2_mdsc, _ = eval_utils.compute_mean_dsc(_cm_g2_t)
    #   # g3_mdsc, _ = eval_utils.compute_mean_dsc(_cm_g3_t)
    #   # g4_mdsc, _ = eval_utils.compute_mean_dsc(_cm_g4_t)
      
    #   print("guidance 1 mean dsc %f" %g1_dsc_t)

    # TODO: save instead of showing
    eval_utils.plot_confusion_matrix(cm_total, classes=np.arange(dataset.num_of_classes), normalize=True,
                                     title='Confusion matrix, without normalization', save_path=FLAGS.eval_logdir)
    
    if common.Z_LABEL in samples:
      total_eval_z /= dataset.splits_to_sizes[EVAL_SPLIT]
      print("MSE of z prediction {}".format(total_eval_z))
      plt.plot(total_z_label, total_z_pred, ".")
      plt.hold(True)
      plt.plot(total_z_label, total_z_label, "r")
      plt.title("Z labeld and prediction")
      plt.xlabel("z_label")
      plt.ylabel("z_prediction")
      plt.savefig(FLAGS.eval_logdir+"z_information.png")
    
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

    return mean_dice_score
  
if __name__ == '__main__':
    # guidance = np.load("/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/training_seg_merge_001.npy")
    # for i in range(14):
    #   plt.imshow(guidance[...,i])
    #   plt.show()
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
    