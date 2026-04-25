#!/usr/bin/env bash
set -euo pipefail

ARCH="CLIP-LoRA:ViT-L/14"
CONFIG_FILE="/root/autodl-tmp/codes/val_configs/all_datasets.yaml"
CROP_SIZE=224
LORA_RANK=16
LORA_ALPHA=32
BATCH_SIZE=64
JPEG_QUALITY=100
GPU_ID=0
SAVE_BAD_CASE=false
SKIP_PATH_CHECK=false
USE_JPEG=false
USE_RESIZE=false
USE_BLUR=false
USE_IS_RESIZE=true
JPEG=96
RESIZE=0.05
BLUR=2.0
MAX_SAMPLE=-1
CKPT=""
RESULT_NAME=""
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

while getopts ":k:r:g:t:m:" opt; do
  case $opt in
    k) CKPT="$OPTARG" ;;
    r) RESULT_NAME="$OPTARG" ;;
    g) GPU_ID="$OPTARG" ;;
    t) RUN_TIMESTAMP="$OPTARG" ;;
    m) MAX_SAMPLE="$OPTARG" ;;
    \?)
      echo "Usage: $0 -k CKPT_PATH -r RESULT_NAME [-g GPU_ID] [-t RUN_TIMESTAMP] [-m MAX_SAMPLE]"
      exit 1
      ;;
  esac
done

if [[ -z "$CKPT" || -z "$RESULT_NAME" ]]; then
  echo "Both -k CKPT_PATH and -r RESULT_NAME are required."
  exit 1
fi

OPT_FLAGS=""
$SAVE_BAD_CASE && OPT_FLAGS+=" --save_bad_case"
$SKIP_PATH_CHECK && OPT_FLAGS+=" --skip_path_check"
$USE_IS_RESIZE && OPT_FLAGS+=" --is_resize"
$USE_JPEG && OPT_FLAGS+=" --jpeg ${JPEG}"
$USE_RESIZE && OPT_FLAGS+=" --resize ${RESIZE}"
$USE_BLUR && OPT_FLAGS+=" --blur ${BLUR}"

RESULT_FOLDER="/root/autodl-tmp/codes/SPACE/result/SPACE/${RESULT_NAME}_${RUN_TIMESTAMP}"
mkdir -p "$RESULT_FOLDER"

echo "=== Configuration ===" >"$RESULT_FOLDER/config_summary.txt"
echo "Architecture: $ARCH" >>"$RESULT_FOLDER/config_summary.txt"
echo "Checkpoint: $CKPT" >>"$RESULT_FOLDER/config_summary.txt"
echo "Config file: $CONFIG_FILE" >>"$RESULT_FOLDER/config_summary.txt"
echo "LoRA rank: $LORA_RANK" >>"$RESULT_FOLDER/config_summary.txt"
echo "LoRA alpha: $LORA_ALPHA" >>"$RESULT_FOLDER/config_summary.txt"
echo "JPEG quality: $JPEG_QUALITY" >>"$RESULT_FOLDER/config_summary.txt"
echo "Run date: $(date)" >>"$RESULT_FOLDER/config_summary.txt"

echo "Starting evaluation with $ARCH model"
echo "Results will be saved to: $RESULT_FOLDER"

python validate.py \
  --arch="$ARCH" \
  --config="$CONFIG_FILE" \
  --ckpt="$CKPT" \
  --result_folder="$RESULT_FOLDER" \
  --batch_size="$BATCH_SIZE" \
  --lora_rank="$LORA_RANK" \
  --lora_alpha="$LORA_ALPHA" \
  --jpeg_quality="$JPEG_QUALITY" \
  --gpu_id="$GPU_ID" \
  --max_sample="$MAX_SAMPLE" \
  --crop_size="$CROP_SIZE" \
  $OPT_FLAGS

echo "Evaluation finished: $RESULT_FOLDER"
