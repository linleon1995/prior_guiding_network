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


gpu_ids=1

# ================================================ MICCAI ==================================
CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
    --dataset_name 2013_MICCAI_Abdominal \
    --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
    --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_057/model.ckpt-best \
    --seq_length=1 \
    --eval_split val \
    --guid_fuse mean_wo_back \
    --prior_num_subject 24 \
    --guid_encoder early \
    --show_pred_only True \
    --_3d_metrics True \


# ================================================ CHAOS ==================================
# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_107/model.ckpt-best \
#     --guid_fuse conv \
#     --prior_num_subject=16 \
#     --eval_split test \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions context_att context_att context_att guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_000/model.ckpt-best \
#     --guid_fuse conv \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_040/model.ckpt-best \
#     --guid_fuse conv \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \


# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_046/model.ckpt-best \
#     --guid_fuse conv \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_053/model.ckpt-best \
#     --guid_fuse mean_wo_back \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions context_att context_att context_att guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_054/model.ckpt-best \
#     --guid_fuse mean_wo_back \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_051/model.ckpt-best \
#     --guid_fuse mean_wo_back \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_055/model.ckpt-best \
#     --guid_fuse mean_wo_back \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_058/model.ckpt-best \
#     --guid_fuse conv \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions context_att context_att context_att guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_059/model.ckpt-best \
#     --guid_fuse conv \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_060/model.ckpt-best \
#     --guid_fuse conv \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_062/model.ckpt-best \
#     --guid_fuse conv \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \