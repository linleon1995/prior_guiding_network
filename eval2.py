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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
# CHECKPOINT = None

DATASET_DIR = '/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord_seq/'
PRIOR_PATH = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/'


parser = argparse.ArgumentParser()

parser.add_argument('--predict_without_background', type=bool, default=True,
                    help='')

parser.add_argument('--fuse_flag', type=bool, default=False,
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

parser.add_argument('--vis_guidance', type=bool, default=True,
                    help='')

parser.add_argument('--vis_features', type=bool, default=True,
                    help='')

parser.add_argument('--display_box_plot', type=bool, default=False,
                    help='')

parser.add_argument('--store_all_imgs', type=bool, default=False,
                    help='')

# Dataset settings.
parser.add_argument('--dataset', type=str, default='2013_MICCAI_Abdominal',
                    help='')

parser.add_argument('--eval_split', type=str, default='train',
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
                split_name=FLAGS.eval_split,
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
                seq_length=3,
                seq_type="forward")            
  # TODO: make dirs?
  # TODO: Add model name in dir to distinguish
  
  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

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
         3,
         EVAL_CROP_SIZE[0],
         EVAL_CROP_SIZE[1],
         1])

    # Set up tf session and initialize variables.
    sess = tf.Session()
    init = tf.global_variables_initializer()

    sess.run(init)
    sess.run(tf.local_variables_initializer())

    # image_placeholder = tf.placeholder(tf.float32, shape=[1,EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1],1])
    # label_placeholder = tf.placeholder(tf.int32, shape=[None,EVAL_CROP_SIZE[0],EVAL_CROP_SIZE[1],1])
    # num_slices_placeholder = tf.placeholder(tf.int64, shape=[None])
    
    # placeholder_dict = {common.IMAGE: image_placeholder,
    #                     common.LABEL: label_placeholder,
    #                     common.NUM_SLICES: num_slices_placeholder}

    for i in range(dataset.splits_to_sizes[FLAGS.eval_split]):
        data = sess.run(samples)
        print(i)
        # if i == 0:
        plt.imshow(data[common.IMAGE][0,0,...,0])
        plt.show()
        plt.imshow(data[common.IMAGE][0,1,...,0])
        plt.show()
        plt.imshow(data[common.IMAGE][0,2,...,0])
        plt.show()
        

  
if __name__ == '__main__':
    # guidance = np.load("/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/training_seg_merge_001.npy")
    # for i in range(14):
    #   plt.imshow(guidance[...,i])
    #   plt.show()
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)
    