#!/bin/bash
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
#
# This script is used to run local test on PASCAL VOC 2012. Users could also
# modify from this script for their use case.
#
# Usage:
#   # From the tensorflow/models/research/deeplab directory.
#   sh ./local_test.sh
#
#

# DATASET_NAME = ['2013_MICCAI_Abdominal']
# DATASET_NAME = ['2019_ISBI_CHAOS_MR_T1', '2019_ISBI_CHAOS_MR_T2']
# DATASET_NAME = ['2019_ISBI_CHAOS_CT']
gpu_ids=0


# T2 image decay=1e-3 out_node=64 conv_num=2 n=16 dice_loss
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T2 \
    --batch_size=16 \
    --train_split train \
    --guid_fuse mean_wo_back \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=50000 \
    --save_checkpoint_steps=500 \
    --guid_encoder image_only \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid guid_uni guid_uni guid_uni guid_uni \
    --weight_decay=0.001 \
    --out_node=64 \
    --guidance_loss False \
    --stage_pred_loss False \
    --guid_conv_nums=2 \


