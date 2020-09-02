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

# prior single_image train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse mean_wo_back \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=80000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# prior single_image context_att train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse mean_wo_back \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=80000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions context_att context_att context_att guid_uni guid_uni \

# prior single_image train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse mean_wo_back \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=60000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# prior single_image train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T2 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse mean_wo_back \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=60000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# prior single_image train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse conv \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=80000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# prior single_image context_att train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse conv \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=80000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions context_att context_att context_att guid_uni guid_uni \

# prior single_image train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T1 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse conv \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=60000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# prior single_image train
CUDA_VISIBLE_DEVICES=$gpu_ids python train.py \
    --dataset_name 2019_ISBI_CHAOS_MR_T2 \
    --tf_initial_checkpoint /home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_012/model.ckpt-best \
    --batch_size=16 \
    --train_split train \
    --guid_fuse conv \
    --weight_decay=0.001 \
    --validation_steps=500 \
    --training_number_of_steps=60000 \
    --save_checkpoint_steps=500 \
    --prior_num_subject=16 \
    --min_resize_value=256 \
    --max_resize_value=256 \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

#######################################################################################################################
# # prior bi train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --prior_num_subject=16 \
#     --cell_type=BiConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# # image bi train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --guid_encoder=image_only \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --cell_type=BiConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \


# # prior context single train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=12 \
#     --seq_length=1 \
#     --guid_encoder=early \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --prior_num_subject=16 \
#     --cell_type=BiConvGRU \
#     --fusions context_att context_att context_att guid_uni guid_uni \


# # prior context sequence train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --guid_encoder=early \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --prior_num_subject=16 \
#     --cell_type=BiConvGRU \
#     --fusions context_att context_att context_att guid_uni guid_uni \


# # prior bi train val
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train val \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --prior_num_subject=20 \
#     --cell_type=BiConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# # image bi train val
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --guid_encoder=image_only \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --cell_type=BiConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# # image forward train val
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --guid_encoder=image_only \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --cell_type=ConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# # prior forward train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --prior_num_subject=16 \
#     --cell_type=ConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# # image forward train
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --guid_encoder=image_only \
#     --train_split train \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --cell_type=ConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \

# # prior forward train val
# python train.py \
#     --dataset_name 2019_ISBI_CHAOS_CT \
#     --batch_size=4 \
#     --seq_length=3 \
#     --train_split train val \
#     --weight_decay=0.001 \
#     --validation_steps=1000 \
#     --training_number_of_steps=30000 \
#     --save_checkpoint_steps=1000 \
#     --prior_num_subject=20 \
#     --cell_type=ConvGRU \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \