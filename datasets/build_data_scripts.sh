# build training and validation data

# build testing dataset
# python build_miccai2013.py \
#     --data-dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/" \
#     --output-dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord" \
#     --dataset-split="train"

# python build_miccai2013.py \
#     --data-dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/" \
#     --output-dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord" \
#     --dataset-split="val"

# python build_miccai2013.py \
#     --data-dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/training/" \
#     --output-dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/tfrecord" \
#     --dataset-split="test"



# build 2013 MICCAI BTCV prior
# python build_miccai_prior.py \
#     --num_subject 1 24 30 \
#     --prior_slice 1 \
#     --num_class=14 \
#     --data_dir="/home/acm528_02/Jing_Siang/data/Synpase_raw/label/" \
#     --output_dir="/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/2013_MICCAI_BTCV/"


# build 2019 ISBI CHAOS prior
# python build_chaos_prior.py \
#     --num_subject 1 16 20 \
#     --prior_slice 1 \
#     --num_class=2 \
#     --modality="CT" \
#     --data_dir="/home/acm528_02/Jing_Siang/data/2019_ISBI_CHAOS/Train_Sets/CT/" \
#     --output_dir="/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/2019_ISBI_CHAOS/"
    
#  python build_chaos_prior.py \
#     --num_subject 1 16 20 \
#     --prior_slice 1 \
#     --num_class=5 \
#     --modality="MR_T1" \
#     --data_dir="/home/acm528_02/Jing_Siang/data/2019_ISBI_CHAOS/Train_Sets/MR/" \
#     --output_dir="/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/2019_ISBI_CHAOS/"   
    
# python build_chaos_prior.py \
#   --num_subject 1 16 20 \
#   --prior_slice 1 \
#   --num_class=5 \
#   --modality="MR_T2" \
#   --data_dir="/home/acm528_02/Jing_Siang/data/2019_ISBI_CHAOS/Train_Sets/MR/" \
#   --output_dir="/home/acm528_02/Jing_Siang/project/Tensorflow/tf_thesis/priors/2019_ISBI_CHAOS/"        