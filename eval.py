import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import SimpleITK as sitk
import nibabel as nib

import common
import model
from evals import chaos_eval
from datasets import data_generator, file_utils
from utils import train_utils
from core import preprocess_utils
from evals import evaluator
from evals import eval_utils
import cv2
import math
import nibabel as nib
# spatial_transfom_exp = experiments.spatial_transfom_exp
str2bool = train_utils.str2bool
load_model = eval_utils.load_model
IMG_LIST = [136, 137, 138, 143, 144, 145, 161, 162, 163, 248, 249, 250, 253, 254, 255, 256, 257, 258, 447, 448, 449, 571, 572, 573]
IMG_LIST = [136]

parser = argparse.ArgumentParser()
# Model Parameters, keep it same as the checkpoint used
parser.add_argument('--fusions', nargs='+', required=True,
                    help='')

parser.add_argument('--dataset_name', required=True,
                    help='')

parser.add_argument('--eval_split', nargs='+', required=True,
                    help='')

parser.add_argument('--seq_length', type=int, default=1,
                    help='')

parser.add_argument('--guid_fuse', type=str, default="sum_wo_back",
                    help='')

parser.add_argument('--cell_type', type=str, default="ConvGRU",
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

parser.add_argument('--master', type=str, default='',
                    help='')

# Settings for log directories.
# parser.add_argument('--eval_logdir', type=str, default=CHECKPOINT+'-eval/',
#                     help='')

parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='')

# Settings for evaluating the model.
parser.add_argument('--drop_prob', type=float, default=None,
                    help='')

parser.add_argument('--guid_weight', type=str2bool, nargs='?', const=True, default=False,
                    help='')

parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='')

parser.add_argument('--eval_interval_secs', type=int, default=5,
                    help='')

parser.add_argument('--output_stride', type=int, default=8,
                    help='')

parser.add_argument('--prior_num_slice', type=int, default=1,
                    help='')

parser.add_argument('--prior_num_subject', type=int, default=None,
                    help='')

parser.add_argument('--fusion_slice', type=int, default=3,
                    help='')

parser.add_argument('--guidance_type', type=str, default=None,
                    help='')

# Change to True for adding flipped images during test.
parser.add_argument('--add_flipped_images', type=str2bool, nargs='?', const=True, default=False,
                    help='')

parser.add_argument('--z_model', type=str, default=None,
                    help='')

parser.add_argument('--z_label_method', type=str, default=None,
                    help='')

parser.add_argument('--z_class', type=int, default=None,
                    help='')

parser.add_argument('--stage_pred_loss', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--guidance_loss', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--seg_loss_name', type=str, default="softmax_dice_loss",
                    help='')

parser.add_argument('--guid_loss_name', type=str, default="sigmoid_cross_entropy",
                    help='')

parser.add_argument('--stage_pred_loss_name', type=str, default="sigmoid_cross_entropy",
                    help='')

parser.add_argument('--zero_guidance', type=str2bool, nargs='?', const=True, default=False,
                    help='')

# Evaluation Parameters
parser.add_argument('--eval_metrics', nargs='+', default=None,
                    help='')

parser.add_argument('--display_box_plot', type=str2bool, nargs='?', const=True, default=False,
                    help='')

parser.add_argument('--store_all_imgs', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--show_pred_only', type=str2bool, nargs='?', const=True, default=True,
                    help='')

parser.add_argument('--raw_data_dir', type=str,
                    help='')

parser.add_argument('--use_3d_metrics', type=str2bool, nargs='?', const=True, default=True,
                    help='')

                    
def get_pred_vis(eval_logdir, num_class, image, prediction, label=None):
    subplot_split=(1,3)
    show_seg_results = eval_utils.Build_Pyplot_Subplots(saving_path=eval_logdir,
                                                        is_showfig=False,
                                                        is_savefig=True,
                                                        subplot_split=subplot_split,
                                                        type_list=3*['img'])
    show_seg_results.set_title(["image", "label", "prediction"])
    show_seg_results.set_axis_off()
    return show_seg_results
      
      
def aggregate_evaluation(total_results, metrics, path):
    total_aggregate = {}
    for m in metrics:
        aggregate = []
        for vol_results in total_results:
            aggregate.append(vol_results[m])
    print(total_aggregate)
    #     total_aggregate[m] = sum(aggregate) / len(aggregate)
    
    # with open(os.path.join(path, 'eval_logging.txt'), 'a') as f:
    #     for m in total_aggregate:
    #       if isinstance(total_aggregate[m], (int, float)):
    #         f.write("\n{}: {:.4f}".format(m, total_aggregate[m]))
    #       else:
    #         for c, class_r in enumerate(total_aggregate[m]):
    #           f.write("\n{} in class {}: {:.4f}".format(m, c+1, class_r))
            
    return total_aggregate
        

def get_placeholders(samples, data_information):
    placeholder_dict = {}
    placeholder_dict[common.HEIGHT] = tf.placeholder(tf.int32,shape=[1])
    placeholder_dict[common.WIDTH] = tf.placeholder(tf.int32,shape=[1])
    if FLAGS.seq_length > 1:
      # Shape specific for GRU cell
      placeholder_dict[common.IMAGE] = tf.placeholder(tf.float32,
                                         shape=[1,FLAGS.seq_length,data_information.height,data_information.width,1])
    else:
      placeholder_dict[common.IMAGE] = tf.placeholder(tf.float32, shape=[1,None,None,1])
    placeholder_dict[common.NUM_SLICES] = tf.placeholder(tf.int64, shape=[None])
    if "train" in FLAGS.eval_split or "val" in FLAGS.eval_split:
      if FLAGS.seq_length > 1:
        placeholder_dict[common.LABEL] = tf.placeholder(tf.int32, shape=[1,FLAGS.seq_length,data_information.height,data_information.width,1])
      else:
        placeholder_dict[common.LABEL] = tf.placeholder(tf.int32, shape=[None,None, None,1])

    if common.Z_LABEL in samples:
      samples[common.Z_LABEL] = tf.identity(samples[common.Z_LABEL], name=common.Z_LABEL)
      z_label_placeholder = tf.placeholder(tf.float32, shape=[None])
      placeholder_dict[common.Z_LABEL] = z_label_placeholder
    else:
      placeholder_dict[common.Z_LABEL] = None

    if FLAGS.guidance_type == "gt":
      prior_seg_placeholder = tf.placeholder(tf.int32,shape=[None,None, None, 1])
    elif FLAGS.guidance_type in ("training_data_fusion", "training_data_fusion_h"):
      # TODO: CHAOS MR case --> general way
      if FLAGS.dataset_name == "2019_ISBI_CHAOS_MR_T1" and FLAGS.dataset_name == "2019_ISBI_CHAOS_MR_T2":
        prior_seg_placeholder = tf.placeholder(
          tf.float32,shape=[None, data_information.height, data_information.width, 10, 1])
      else:
        prior_seg_placeholder = tf.placeholder(
          tf.float32,shape=[None, data_information.height, data_information.width, data_information.num_classes, 1])
      # prior_seg_placeholder = tf.placeholder(tf.float32,shape=[None,None, None, 1])
    else:
      prior_seg_placeholder = None
    placeholder_dict[common.PRIOR_SEGS] = prior_seg_placeholder

    if 'prior_slices' in samples:
      prior_slices_placeholder = tf.placeholder(tf.int64, shape=[None])
      placeholder_dict['prior_slices'] = prior_slices_placeholder
    else:
      placeholder_dict['prior_slices'] = None
    return placeholder_dict


def main(unused_argv):
  eval_logdir = FLAGS.checkpoint_dir+'-eval/'
  data_information = data_generator._DATASETS_INFORMATION[FLAGS.dataset_name]

  tf.gfile.MakeDirs(eval_logdir)

  parameters_dict = vars(FLAGS)
  with open(os.path.join(eval_logdir, 'eval_logging.txt'), 'w') as f:
    f.write("Start Evaluation\n")
    f.write(60*"="+"\n")
    f.write("\n")


  tf.logging.set_verbosity(tf.logging.INFO)

  dataset = data_generator.Dataset(
                dataset_name=FLAGS.dataset_name,
                split_name=FLAGS.eval_split,
                batch_size=1,
                mt_label_method=FLAGS.z_label_method,
                guidance_type=FLAGS.guidance_type,
                mt_class=FLAGS.z_class,
                mt_label_type="z_label",
                crop_size=[data_information.height, data_information.width],
                min_resize_value=data_information.height,
                max_resize_value=data_information.height,
                num_readers=2,
                is_training=False,
                shuffle_data=False,
                repeat_data=False,
                prior_num_slice=FLAGS.prior_num_slice,
                prior_num_subject=FLAGS.prior_num_subject,
                seq_length=FLAGS.seq_length,
                seq_type="forward")

  tf.logging.info('Evaluating on %s set', FLAGS.eval_split)

  with tf.Graph().as_default() as graph:
    iterator = dataset.get_dataset().make_one_shot_iterator()
    samples = iterator.get_next()

    # Add name to input and label nodes so we can add to summary.
    samples[common.IMAGE] = tf.identity(samples[common.IMAGE], name=common.IMAGE)
    if "train" in FLAGS.eval_split or "val" in FLAGS.eval_split:
      samples[common.LABEL] = tf.identity(samples[common.LABEL], name=common.LABEL)

    model_options = common.ModelOptions(
      outputs_to_num_classes=dataset.num_of_classes,
      crop_size=[data_information.height, data_information.width],
      output_stride=FLAGS.output_stride)

    placeholder_dict = get_placeholders(samples, data_information)

    output_dict, layers_dict = model.pgb_network(
                placeholder_dict[common.IMAGE],
                placeholder_dict[common.HEIGHT],
                placeholder_dict[common.WIDTH],
                model_options=model_options,
                prior_segs=placeholder_dict[common.PRIOR_SEGS],
                num_class=dataset.num_of_classes,
                batch_size=FLAGS.eval_batch_size,
                guidance_type=FLAGS.guidance_type,
                fusion_slice=FLAGS.fusion_slice,
                drop_prob=FLAGS.drop_prob,
                guid_weight=FLAGS.guid_weight,
                stn_in_each_class=True,
                is_training=False,
                weight_decay=0.0,
                share=FLAGS.share,
                fusions=FLAGS.fusions,
                out_node=FLAGS.out_node,
                guid_encoder=FLAGS.guid_encoder,
                z_label_method=FLAGS.z_label_method,
                z_model=FLAGS.z_model,
                z_class=FLAGS.z_class,
                guidance_loss=FLAGS.guidance_loss,
                stage_pred_loss=FLAGS.stage_pred_loss,
                guid_loss_name=FLAGS.guid_loss_name,
                stage_pred_loss_name=FLAGS.stage_pred_loss_name,
                guid_conv_nums=FLAGS.guid_conv_nums,
                guid_conv_type=FLAGS.guid_conv_type,
                reuse=tf.AUTO_REUSE,
                apply_sram2=FLAGS.apply_sram2,
                guid_fuse=FLAGS.guid_fuse,
                seq_length=FLAGS.seq_length,
                cell_type=FLAGS.cell_type
                )

    logits = output_dict[common.OUTPUT_TYPE]
    # TODO: not general
    if FLAGS.eval_split[0] == "test":
      logits = tf.image.resize_bilinear(logits, tf.concat([placeholder_dict[common.HEIGHT], placeholder_dict[common.WIDTH]],axis=0), align_corners=False)
    prediction = eval_utils.inference_segmentation(logits, dim=3)

    if "train" in FLAGS.eval_split or "val" in FLAGS.eval_split:
      pred_flat = tf.reshape(prediction, shape=[-1,])
      if FLAGS.seq_length > 1:
        labels = placeholder_dict[common.LABEL][:,1]
      else:
        labels = placeholder_dict[common.LABEL]
      labels = tf.squeeze(labels, axis=3)
      label_onehot = tf.one_hot(indices=labels,
                                depth=dataset.num_of_classes,
                                on_value=1.0,
                                off_value=0.0,
                                axis=3)
      labels_flat = tf.reshape(labels, shape=[-1,])
      # Define Confusion Maxtrix
      cm = tf.confusion_matrix(labels_flat, pred_flat, num_classes=dataset.num_of_classes)

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


    cm_total, cm_volume  = 0, 0
    j = 0
    total_results = []
    num_class = dataset.num_of_classes
    
    raw_data_list = file_utils.get_file_list(FLAGS.raw_data_dir, fileExt=["nii.gz"], sort_files=True)
    raw_data_list = raw_data_list[16:]
    if "CHAOS" in FLAGS.dataset_name:
      evaluate = evaluator.build_dicom_evaluator(FLAGS.eval_metrics)
    elif "MICCAI" in FLAGS.dataset_name:
      evaluate = evaluator.build_evaluator(FLAGS.eval_metrics)
    else:
      raise ValueError("Unknown dataset name") 
    
    for split_name in FLAGS.eval_split:
      num_sample = dataset.splits_to_sizes[split_name]
      if FLAGS.store_all_imgs:
          display_imgs = np.arange(num_sample)
      else:
          display_imgs = IMG_LIST

      for i in range(num_sample):
          data = sess.run(samples)
          _feed_dict = {placeholder_dict[k]: v for k, v in data.items() if k in placeholder_dict}
          if FLAGS.seq_length > 1:
            depth = data[common.DEPTH][0,FLAGS.seq_length//2]
            data[common.IMAGE] = data[common.IMAGE][:,FLAGS.seq_length//2]
            if split_name in ("train", "val"):
              data[common.LABEL] = data[common.LABEL][:,FLAGS.seq_length//2]
          else:
            depth = data[common.DEPTH][0]
          num_slice = data[common.NUM_SLICES][0]
          pred = sess.run(prediction, feed_dict=_feed_dict)
          print(FLAGS.dataset_name, 'Sample {} Slice {}/{}'.format(i, depth, num_slice))

          # Segmentation Evaluation
          if split_name in ("train", "val"):
            cm_slice = sess.run(cm, feed_dict=_feed_dict)
            cm_total += cm_slice
            
            if FLAGS.use_3d_metrics:
              cm_volume += cm_slice
              if depth == 0:
                ref, seg = [], []
              elif depth == num_slice-1:
                ref = np.concatenate(ref, axis=2)
                seg = np.concatenate(seg, axis=2)
                # 3D visualization
                ref, seg = np.int32(ref==1), np.int32(seg==1)
                data_dict = {"Reference": ref, "Segmentation": seg}
                # evaluate.visualize_in_3d(data_dict, raw_data_path=raw_data_list[j])
                results = evaluate(ref, seg, total_cm=cm_volume, raw_data_path=raw_data_list[j], num_class=num_class)
                total_results.append(results)
                print(results)
                
                j += 1
                cm_volume = 0
              else:
                ref.append(data[common.LABEL][0])
                seg.append(pred[0][...,np.newaxis])
            else:
              ref = data[common.LABEL][0,...,0]
              seg = pred[0]
              results = evaluate(ref, seg, total_cm=cm_slice)
              print(results)
          
          # 2D visualization
          if FLAGS.store_all_imgs:
              display_imgs = np.arange(num_sample)
          else:
              display_imgs = IMG_LIST
          if i in display_imgs:
            image, label, pred = data[common.IMAGE][0,...,0], data[common.LABEL][0,...,0], pred[0]
            show_seg_results  = get_pred_vis(eval_logdir, num_class, image, pred, label)
            
            parameters = [{"cmap": "gray"}]
            parameters.extend(2*[{"vmin": 0, "vmax": num_class}])
            show_seg_results.display_figure(
              split_name+'_pred_%04d' %i, [image, label, pred], parameters=parameters)
            
    # Save confusion matrix in figure
    eval_utils.plot_confusion_matrix(
      cm_total, num_class=np.arange(num_class), filename="CM", normalize=True, save_path=eval_logdir)

    # Aggregate and log the evaluation
    agg = aggregate_evaluation(total_results, evaluate.metrics, path=eval_logdir)
    # for a in agg:
    #     print("{}: {}".format(a, agg[a]))
    

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    main(unparsed)