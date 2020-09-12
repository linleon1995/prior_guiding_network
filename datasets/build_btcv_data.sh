#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="/home/user/DISK/data/Jing/data"
DATASET="2015_MICCAI_BTCV"

# Root path for BTCV dataset.
BTCV_ROOT="${WORK_DIR}/${DATASET}"

export PYTHONPATH="${BTCV_ROOT}:${PYTHONPATH}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${BTCV_ROOT}/tfrecord/"
mkdir -p "${OUTPUT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/build_btcv_data.py"
echo "Converting 2015 MICCAI BTCV dataset..."
python "${BUILD_SCRIPT}" \
  --data_dir="${BTCV_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --split_indices 0 24 \
  --extract_fg_exist_slice False \
  --seq_length 1 \


BUILD_SCRIPT="${CURRENT_DIR}/build_btcv_data.py"
echo "Converting 2015 MICCAI BTCV dataset..."
python "${BUILD_SCRIPT}" \
  --data_dir="${BTCV_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --split_indices 0 24 \
  --extract_fg_exist_slice False \
  --seq_length 3 \
  
  
# Build prior of the dataset
PRIOR_DIR="${BTCV_ROOT}/priors/"
mkdir -p "${PRIOR_DIR}"

echo "Converting 2015 MICCAI BTCV dataset to prior..."
python build_btcv_prior.py \
  --data_dir="${BTCV_ROOT}" \
  --output_dir="${PRIOR_DIR}" \
  --num_subject 24 30 \
  --prior_slice 1 \
