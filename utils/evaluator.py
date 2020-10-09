
import os
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import SimpleITK as sitk
import nibabel as nib

import common
import model
from chaos_eval import CHAOSmetrics
from datasets import data_generator, file_utils
from utils import eval_utils, train_utils
from core import preprocess_utils
from metrics import _ALL_METRICS
# import experiments
import cv2
import math
import nibabel as nib
# spatial_transfom_exp = experiments.spatial_transfom_exp
SSD = CHAOSmetrics.SSD
DICE = CHAOSmetrics.DICE
RAVD = CHAOSmetrics.RAVD

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# TODO: if dir not exist. Don't build new one
IMG_LIST = [50, 60, 64, 70, 82, 222,226, 227, 228, 350, 481]
IMG_LIST = [136, 137, 138, 143, 144, 145, 161, 162, 163, 248, 249, 250, 253, 254, 255, 256, 257, 258, 447, 448, 449, 571, 572, 573]
# IMG_LIST = []

# TODO: do it correctly
CT_TRAIN_SET = [1,2,5,6,8,10,14,16,18,19,21,22,23,24,25,26,27,28,29,30]
CT_TEST_SET = [11,12,13,15,17,20,3,31,32,33,34,35,36,37,38,39,4,40,7,9,]
MR_TRAIN_SET = [1,2,3,5,8,10,13,15,19,20,21,22,31,32,33,34,36,37,38,39]
MR_TEST_SET = [11,12,14,16,17,18,23,24,25,26,27,28,29,30,35,4,40,6,7,9]



class build_evaluator(object):
    """
    
    """
    default_metrics = ["Precision",
                       "Recall",
                       "DSC"]
    
    def __init__(self, metrics=None):
        if metrics is not None:
            for m in metrics:
                self.metrics.append(m)
        else:
            for m in self.default_metrics:
                self.metrics.append(m)
                
    def __call__(self, ref, test):
        results = []
        self.check_shape(ref, test)
        for m in self.metrics:
            results.append(_ALL_METRICS[m](ref, test))
        return results
    
    def check_shape(self, ref, test):
        assert ref.shape == test.shape
        # These metrics require 3D data
        for _3d_m in ["DSC", "RAVD", "MSSD", "ASSD"]:
            if _3d_m in self.metrics:
                if ref.ndim != 3 or test.ndim != 3:
                    raise ValueError("Incorrect shape for %s, this metrics require 3d data" %_3d_m)
        
class nitfi_evaluator():
class dicom_evaluator():        
        
class build_chaos_evaluator(build_evaluator):
    def __init__(self, metrics, dicom_dir):
        super().__init__(metrics)
        self.dicom_dir = dicom_dir
    

for split_name in FLAGS.eval_split:
    num_sample = dataset.splits_to_sizes[sub_dataset][split_name]
    for i in range(num_sample):
        if "MICCAI" in FLAGS.dataset_name:
            evaluator = build_btcv_evaluator(FLAGS.metrics)
        elif "CHAOS" in FLAGS.dataset_name:
            evaluator = build_chaos_evaluator(FLAGS.metrics)
        else:
            raise ValueError("Unknown Dataset Name")
        
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
        print(sub_dataset, 'Sample {} Slice {}/{}'.format(i, depth, num_slice))

        # Segmentation Evaluation
        pred = sess.run(prediction, feed_dict=_feed_dict)       
        
        results = evaluator(data[common.LABEL], pred)      
        aggregate_evaluation(results)
        
        
def eval_chaos():
  # get prdiction
  # get confusion matrix
  # get assigned evaluator
  
  # guidance visualization
  # feature visualization
  # saving file


def eval_btcv():
  pass

def main2(unused_argv):
  if FLAGS.dataset_name == "":
    eval_btcv()
  elif FLAGS.dataset_name == "CHAOS":
    eval_chaos()
  else:
    raise ValueError("Unknown Dataset Name")
