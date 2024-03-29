#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
# The directory that raw data saved
WORK_DIR="/home/user/DISK/data/Jing/data"
DATASET="2019_ISBI_CHAOS"

# Root path for CHAOS dataset.
CHAOS_ROOT="${WORK_DIR}/${DATASET}"

export PYTHONPATH="${CHAOS_ROOT}:${PYTHONPATH}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${CHAOS_ROOT}/tfrecord/"
mkdir -p "${OUTPUT_DIR}"

BUILD_SCRIPT="${CURRENT_DIR}/build_chaos_data.py"

echo "Converting 2019 ISBI CHAOS dataset..."
python "${BUILD_SCRIPT}" \
  --data_dir="${CHAOS_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --seq_length=1 \
  --split_indices 0 16

echo "Converting 2019 ISBI CHAOS dataset in length=3 sequence..."
python "${BUILD_SCRIPT}" \
  --data_dir="${CHAOS_ROOT}" \
  --output_dir="${OUTPUT_DIR}" \
  --seq_length=3 \
  --split_indices 0 16

