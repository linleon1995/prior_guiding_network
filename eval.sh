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

# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train \
#     --weight_decay=0.001 \
#     --training_number_of_steps=50000 \
#     --prior_num_subject=16 \
#     --cell_type=BiConvGRU \
#     --min_resize_value=256 \
#     --max_resize_value=256

# python train.py \
#     --dataset_name 2013_MICCAI_Abdominal \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train \
#     --weight_decay=0.001 \
#     --training_number_of_steps=180000 \
#     --prior_num_subject=24 \
#     --cell_type=BiConvGRU \

# TODO: checkpoint_dir should select the latest one automatically
# TODO: test empty img list
# python eval.py \
#     --dataset_name 2013_MICCAI_Abdominal \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_034/model.ckpt-128000 \
#     --seq_length=3 \
#     --eval_split val \
#     --prior_num_subject=24 \
#     --store_all_imgs=False

python eval.py \
    --dataset_name 2013_MICCAI_Abdominal \
    --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_082/model.ckpt-best-5960 \
    --seq_length=3 \
    --prior_num_subject=24 \
    --store_all_imgs False \
    --show_pred_only False \
    --eval_split val \
    --_3d_metrics False \

# python eval.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_073/model.ckpt-50000 \
#     --seq_length=3 \
#     --eval_split val \
#     --prior_num_subject=16 \
#     --store_all_imgs=True \
#     --show_pred_only=True

# python train.py --weight_decay=0.1
# python train.py --weight_decay=0.001

# python train.py --batch_size=8 --crop-size=513 --min_scale_factor=0.75 --max_scale_factor=1.25 scale_factor_step_size=0.125