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


# # prior bi train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=2000 \
#     --training_number_of_steps=50000 \
#     --save_checkpoint_steps=2000 \
#     --prior_num_subject=16 \
#     --cell_type=BiConvGRU \
#     --min_resize_value=256 \
#     --max_resize_value=256

# # prior forward train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=2000 \
#     --training_number_of_steps=50000 \
#     --save_checkpoint_steps=2000 \
#     --prior_num_subject=16 \
#     --cell_type=ConvGRU \
#     --min_resize_value=256 \
#     --max_resize_value=256

# # image bi train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --batch_size=4 \
#     --seq_length=3 \
#     --guid_encoder=image_only \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=2000 \
#     --training_number_of_steps=50000 \
#     --save_checkpoint_steps=2000 \
#     --cell_type=BiConvGRU \
#     --min_resize_value=256 \
#     --max_resize_value=256

# image forward train
python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --batch_size=4 \
    --seq_length=3 \
    --guid_encoder=image_only \
    --train_split train \
    --weight_decay=0.001 \
    --validation_steps=2000 \
    --training_number_of_steps=50000 \
    --save_checkpoint_steps=2000 \
    --cell_type=ConvGRU \
    --min_resize_value=256 \
    --max_resize_value=256

# # prior bi train val
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train val \
#     --weight_decay=0.001 \
#     --validation_steps=2000 \
#     --training_number_of_steps=50000 \
#     --save_checkpoint_steps=2000 \
#     --prior_num_subject=20 \
#     --cell_type=BiConvGRU \
#     --min_resize_value=256 \
#     --max_resize_value=256

# # prior forward train val
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train val \
#     --weight_decay=0.001 \
#     --validation_steps=2000 \
#     --training_number_of_steps=50000 \
#     --save_checkpoint_steps=2000 \
#     --prior_num_subject=20 \
#     --cell_type=ConvGRU \
#     --min_resize_value=256 \
#     --max_resize_value=256

# image bi train val
python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --batch_size=4 \
    --seq_length=3 \
    --guid_encoder=image_only \
    --train_split train val \
    --weight_decay=0.001 \
    --validation_steps=2000 \
    --training_number_of_steps=50000 \
    --save_checkpoint_steps=2000 \
    --cell_type=BiConvGRU \
    --min_resize_value=256 \
    --max_resize_value=256

# image forward train val
python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --batch_size=4 \
    --seq_length=3 \
    --guid_encoder=image_only \
    --train_split train val \
    --weight_decay=0.001 \
    --validation_steps=2000 \
    --training_number_of_steps=50000 \
    --save_checkpoint_steps=2000 \
    --cell_type=ConvGRU \
    --min_resize_value=256 \
    --max_resize_value=256

