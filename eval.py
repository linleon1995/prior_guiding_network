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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

EVAL_CROP_SIZE = [256,256]
# EVAL_CROP_SIZE = [512,512]
ATROUS_RATES = None
# TODO: Multi-Scale Test
# Change to [0.5, 0.75, 1.0, 1.25, 1.5, 1.75] for multi-scale test.
EVAL_SCALES = [1.0]
HU_WINDOW = [-125, 275]
IMG_LIST = [50,60, 61, 62, 63, 64, 80, 81, 82, 83, 84,220,221,222,223,224,228,340,350,480,481,482,483,484,495]


# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_012/model.ckpt-40000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_007/model.ckpt-40000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_005/model.ckpt-36235'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_008/model.ckpt-40000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_009/model.ckpt-29382'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_000/model.ckpt-40000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_004/model.ckpt-80000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_009/model.ckpt-80000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_010/model.ckpt-80000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_011/model.ckpt-50000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_003/model.ckpt-40000'

# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_016/model.ckpt-80000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_018/model.ckpt-80000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_021/model.ckpt-80000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_030/model.ckpt-65000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_065/model.ckpt-80000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_059/model.ckpt-15000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_057/model.ckpt-15000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_060/model.ckpt-10000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_001/model.ckpt-10000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_000/model.ckpt-25000'

# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_006/model.ckpt-20000'
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_023/model.ckpt-40000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_062/model.ckpt-40000' # w/o transform
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_029/model.ckpt-40000' # guid weight
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_078/model.ckpt-25000' # single aff
# CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_080/model.ckpt-20000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_081/model.ckpt-35000' # multiple aff
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_089/model.ckpt-45000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_093/model.ckpt-45000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_025/model.ckpt-35000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_100/model.ckpt-50000'
CHECKPOINT = '/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/thesis_trained/run_111/model.ckpt-20000'

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
                repeat_data=False,
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

    if common.PRIOR_SEGS in samples:
      samples[common.PRIOR_SEGS] = tf.identity(samples[common.PRIOR_SEGS], name=common.PRIOR_SEGS)
      prior_seg_placeholder = tf.placeholder(tf.float32,
                                           shape=[None, EVAL_CROP_SIZE[0],
                                                  EVAL_CROP_SIZE[1], dataset.num_of_classes])
      placeholder_dict[common.PRIOR_SEGS] = prior_seg_placeholder
    else:
      placeholder_dict[common.PRIOR_SEGS] = None

    if 'prior_slices' in samples:
      prior_slices_placeholder = tf.placeholder(tf.int64, shape=[None])
      placeholder_dict['prior_slices'] = prior_slices_placeholder
    else:
      placeholder_dict['prior_slices'] = None

    # guidance = tf.convert_to_tensor(np.load("/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/training_seg_merge_010.npy"))
    # guidance = tf.expand_dims(guidance, axis=0)
    
    output_dict, layers_dict = model.pgb_network(
                placeholder_dict[common.IMAGE],
                model_options=model_options,
                affine_transform=FLAGS.affine_transform,
                # deformable_transform=FLAGS.deformable_transform,
                labels=placeholder_dict[common.LABEL],
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
                # weight_decay=FLAGS.weight_decay,
                # fine_tune_batch_norm=FLAGS.fine_tune_batch_norm,
                )
                
    guidance_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'guid' in dict_key}
    feature_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'feature' in dict_key}
    sram_dict = {dict_key: layers_dict[dict_key] for dict_key in layers_dict if 'sram' in dict_key}

    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    print(sum(var_sizes) / (1024 ** 2), 'MB')
    
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

    guid0 = layers_dict["guidance_in"]
    guid1 = tf.nn.softmax(layers_dict["guidance1"])
    guid2 = tf.nn.softmax(layers_dict["guidance2"])
    guid3 = tf.nn.softmax(layers_dict["guidance3"])
    guid4 = tf.nn.softmax(layers_dict["guidance4"])

    # guidance = output_dict[common.GUIDANCE]
    if FLAGS.affine_transform or FLAGS.deformable_transform:
      pp = output_dict[common.GUIDANCE]
      # pp = tf.image.resize_bilinear(output_dict[common.PRIOR_SEGS], 
      #                               [EVAL_CROP_SIZE[0]//FLAGS.output_stride,EVAL_CROP_SIZE[1]//FLAGS.output_stride])
    else:
      pp = label_onehot                   
                 
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
        display_imgs = np.arange(dataset.splits_to_sizes[FLAGS.eval_split])
    else:
        display_imgs = IMG_LIST
    for i in range(dataset.splits_to_sizes[FLAGS.eval_split]):
        data = sess.run(samples)
        _feed_dict = {placeholder_dict[k]: v for k, v in data.items() if k in placeholder_dict}
        print('Sample {} Slice {}'.format(i, data[common.DEPTH][0]))
        
        # if i in range(50, 80, 10):
        #   tt, xx = sess.run([theta, x_s], feed_dict=_feed_dict)
        #   print(tt)
        #   fig, ax = plt.subplots(2,2)
        #   ax[0,0].imshow(xx[0][0,...,0])
        #   ax[0,1].imshow(xx[1][0,...,0])
        #   ax[1,0].imshow(xx[2][0,...,0])
        #   ax[1,1].imshow(xx[3][0,...,0])
        #   plt.show()
        #   fig, ax = plt.subplots(2,2)
        #   ax[0,0].imshow(xx[4][0,...,0])
        #   ax[0,1].imshow(xx[5][0,...,0])
        #   ax[1,0].imshow(xx[6][0,...,0])
        #   ax[1,1].imshow(xx[7][0,...,0])
        #   plt.show()
        # Segmentation Evaluation
        cm_slice, pred = sess.run([cm, predictions], feed_dict=_feed_dict)
        _, dscs = eval_utils.compute_mean_dsc(cm_slice)
        DSC_slice.append(dscs)
        cm_total += cm_slice

        
        if i in display_imgs:
            parameters = [{"cmap": "gray"}]
            parameters.extend(2*[{"vmin": 0, "vmax": dataset.num_of_classes}])
            show_seg_results.set_title(["image", "label","prediction"])
            show_seg_results.set_axis_off()
            show_seg_results.display_figure(FLAGS.eval_split+'_pred_%04d' %i,
                                            [data[common.IMAGE][0,...,0], data[common.LABEL][0,...,0], pred[0]],
                                            parameters=parameters)
      
        foreground_pixel += sess.run(num_fg_pixel, _feed_dict)
        
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
            # TODO: cc, pp
            # TODO: The situation of prior_seg not exist
            class_list = np.arange(14)
            # if i in [220,228,340,350,495]:
            #   weight = tf.get_collection("weight")
            #   w = sess.run(weight, feed_dict=_feed_dict)
            #   print(w)
            for c in class_list:
              g0, g1, g2, g3, g4, prior_seg = sess.run([guid0, guid1, guid2, guid3, guid4, pp], feed_dict=_feed_dict)
              show_guidance.set_title(["guidance1", "guidance2", "guidance3"])
              show_guidance.display_figure(FLAGS.eval_split+'_all_guid-%04d-%03d' % (i,c),
                                            [g1[0,...,c],
                                            g2[0,...,c],
                                            g3[0,...,c]])
                                            
              show_guidance.set_title(["prediction of class {}".format(c), "input prior", "guidance_in (32,32)"])
              show_guidance.display_figure(FLAGS.eval_split+'_prior_and_guid_%04d-%03d' % (i,c),
                                          [np.int32(data[common.LABEL][0,...,0]==c),
                                          prior_seg[0,...,c],
                                          g0[0,...,c]])
          
        # Features Visualization
        if FLAGS.vis_features:
          if i in display_imgs:
            sram_conv = tf.get_collection("/sram_embed")
            s = 46
            conv2 = sram_conv[s]["conv2"]
            guidance = sram_conv[s]["guidance_tile"]
            output = sram_conv[s]["output"]
            c2, guid, out = sess.run([conv2, guidance, output], feed_dict=_feed_dict)
            for cc in range(0, 32, 4):
              filename = [FLAGS.eval_split, "sram_feature", "sample%04d" %i, "sram%03d" %s, "feature%04d" %cc]
              filename = "-".join(filename)
              show_feature.display_figure(filename, [c2[0,...,cc],guid[0,...,cc],out[0,...,cc]])
          
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

    print(30*"=")
    mean_iou = eval_utils.compute_mean_iou(cm_total)
    mean_dice_score, dice_score = eval_utils.compute_mean_dsc(cm_total)
    pixel_acc = eval_utils.compute_accuracy(cm_total)
    _, _ = eval_utils.precision_and_recall(cm_total)
    print(foreground_pixel, foreground_pixel/(256*256*dataset.splits_to_sizes[FLAGS.eval_split]))

    # TODO: save instead of showing
    eval_utils.plot_confusion_matrix(cm_total, classes=np.arange(dataset.num_of_classes), normalize=True,
                                     title='Confusion matrix, without normalization', save_path=FLAGS.eval_logdir)
    
    if common.Z_LABEL in samples:
      total_eval_z /= dataset.splits_to_sizes[FLAGS.eval_split]
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
    var_sizes = [np.product(list(map(int, v.shape))) * v.dtype.size
                for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]
    print(sum(var_sizes) / (1024 ** 2), 'MB')
    # eval_utils.compute_params_and_flops(graph)    
    
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
    