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
# DATASET_NAME = ['2015_MICCAI_Abdominal']
# DATASET_NAME = ['2019_ISBI_CHAOS_MR_T1', '2019_ISBI_CHAOS_MR_T2']
# DATASET_NAME = ['2019_ISBI_CHAOS_CT']

gpu_ids=1


CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
    --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
    --dataset_name 2019_ISBI_CHAOS_MR_T2 \
    --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_119/model.ckpt-best \
    --eval_split test \
    --guid_encoder image_only \
    --store_all_imgs True \
    --show_pred_only True \
    --_3d_metrics False \
    --guid_fuse mean_wo_back \
    --out_node 32 \
#####################################################################################################
# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_118/model.ckpt-best \
#     --eval_split test \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 32 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_117/model.ckpt-best \
#     --eval_split test \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 32 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_114/model.ckpt-best \
#     --eval_split test \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 32 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_019/model.ckpt-best \
#     --eval_split test \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 32 \
#####################################################################################################
# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_112/model.ckpt-best \
#     --eval_split test \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 64 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_115/model.ckpt-best \
#     --eval_split test \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 64 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_110/model.ckpt-best \
#     --eval_split test \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 64 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_111/model.ckpt-best \
#     --eval_split test \
#     --prior_num_subject=16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#     --out_node 64 \

# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2015_MICCAI_Abdominal \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_057/model.ckpt-best\
#     --eval_split val \
#     --prior_num_subject=24 \
#     --store_all_imgs=True \
#     --show_pred_only False \
#     --_3d_metrics False \
#     --guid_fuse mean_wo_back \
#####################################################################################################



# # prior bi train val
# CUDA_VISIBLE_DEVICES=$gpu_ids python eval.py \
#     --fusions  guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_099/model.ckpt-best \
#     --seq_length=1 \
#     --guid_encoder image_only \
#     --guid_fuse conv \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \
#     --out_node=64 \
#     --guid_conv_nums=4 \

# # prior bi train val
# python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_072/model.ckpt-50000 \
#     --seq_length=3 \
#     --cell_type BiConvGRU \
#     --guid_encoder early \
#     --prior_num_subject 16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# # image bi train
# python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_081/model.ckpt-50000 \
#     --seq_length=3 \
#     --cell_type BiConvGRU \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# # image forward train
# python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_086/model.ckpt-50000 \
#     --seq_length=3 \
#     --cell_type ConvGRU \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \


# # prior bi train
# python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_035/model.ckpt-42000 \
#     --seq_length=3 \
#     --cell_type BiConvGRU \
#     --guid_encoder early \
#     --prior_num_subject 16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \


# # image bi train val
# python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_087/model.ckpt-50000 \
#     --seq_length=3 \
#     --cell_type BiConvGRU \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# # image forward train val
# python eval.py \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_038/model.ckpt-50000 \
#     --seq_length=3 \
#     --cell_type ConvGRU \
#     --guid_encoder image_only \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \

# # prior forward train val
# # prior forward train

# # prior att bi train
# python eval.py \
#     --fusions context_att context_att context_att guid_uni guid_uni \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_078/model.ckpt-best \
#     --seq_length=3 \
#     --cell_type BiConvGRU \
#     --guid_encoder early \
#     --prior_num_subject 16 \
#     --store_all_imgs True \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics False \



# python eval.py \
#     --dataset_name 2015_MICCAI_Abdominal \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_083/model.ckpt-best \
#     --seq_length=3 \
#     --cell_type ConvGRU \
#     --guid_encoder early \
#     --prior_num_subject 24 \
#     --store_all_imgs False \
#     --show_pred_only True \
#     --eval_split test \
#     --_3d_metrics True \
#     --fusions guid_uni guid_uni guid_uni guid_uni guid_uni


# python eval.py \
#     --dataset_name 2019_ISBI_CHAOS_MR_T1 2019_ISBI_CHAOS_MR_T2 \
#     --checkpoint_dir=/home/user/DISK/data/Jing/model/Thesis/thesis_trained/run_073/model.ckpt-50000 \
#     --seq_length=3 \
#     --eval_split val \
#     --prior_num_subject=16 \
#     --store_all_imgs=True \
#     --show_pred_only=True
