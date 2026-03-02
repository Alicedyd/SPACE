#!/bin/bash
#
# DeepFake Detection Model Evaluation Script
# This script runs validation tests on various datasets for AI-generated image detection
#

USE_JPEG=false
USE_RESIZE=false
USE_BLUR=false
JPEG=96
RESIZE=0.05
BLUR=2.0

# ===== MODEL CONFIGURATION =====
ARCH="CLIP-LoRA:ViT-L/14"
CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_CLIP_224_LR1e-4_single_run/EXP_CLIP_jpeg90_nomixup_CLIP_224_JPEGaug_lora8_lr1e-4_BS_16_ACC_1_colorspace_RGB/model_epoch_best.pth"

RESULT_FOLDER="./result/clip_wildrf"
CONFIG_FILE="../val_configs/WildRF.yaml"
CROP_SIZE=224
# ===== TRAINING PARAMETERS =====
LORA_RANK=8
LORA_ALPHA=1
BATCH_SIZE=64

# ===== TEST CONDITIONS =====
JPEG_QUALITY=100 # Set quality for JPEG compression test (100 = no compression)
GPU_ID=0         # GPU ID to use for evaluation

# ===== DATA PARAMETERS =====
DATA_MODE="" # Optional data mode parameter

# ===== OPTIONS =====
SAVE_BAD_CASE=false   # Whether to save misclassified examples
SKIP_PATH_CHECK=false # Whether to skip checking if paths exist

USE_IS_RESIZE=true

# Build optional flags
OPT_FLAGS=""

if $SAVE_BAD_CASE; then
  OPT_FLAGS+=" --save_bad_case"
  echo "Will save misclassified examples"
fi

if $SKIP_PATH_CHECK; then
  OPT_FLAGS+=" --skip_path_check"
  echo "Will skip path verification"
fi

if $USE_IS_RESIZE; then
  OPT_FLAGS+=" --is_resize"
  echo "Will use is_resize option"
fi

$USE_JPEG && OPT_FLAGS+=" --jpeg ${JPEG}"
$USE_RESIZE && OPT_FLAGS+=" --resize ${RESIZE}"
$USE_BLUR && OPT_FLAGS+=" --blur ${BLUR}"

$USE_JPEG && RESULT_FOLDER+="_JPEG_${JPEG}"
$USE_RESIZE && RESULT_FOLDER+="_RESIZE_${RESIZE}"
$USE_BLUR && RESULT_FOLDER+="_BLUR_${BLUR}"

MAX_SAMPLE=-1

USE_PATCH=false

# Create results directory
mkdir -p "$RESULT_FOLDER"

# Log configuration to result folder
echo "=== Configuration ===" >"$RESULT_FOLDER/config_summary.txt"
echo "Architecture: $ARCH" >>"$RESULT_FOLDER/config_summary.txt"
echo "Checkpoint: $CKPT" >>"$RESULT_FOLDER/config_summary.txt"
echo "Config file: $CONFIG_FILE" >>"$RESULT_FOLDER/config_summary.txt"
echo "LoRA rank: $LORA_RANK" >>"$RESULT_FOLDER/config_summary.txt"
echo "LoRA alpha: $LORA_ALPHA" >>"$RESULT_FOLDER/config_summary.txt"
echo "JPEG quality: $JPEG_QUALITY" >>"$RESULT_FOLDER/config_summary.txt"
echo "Run date: $(date)" >>"$RESULT_FOLDER/config_summary.txt"

# Print startup message
echo "Starting evaluation with $ARCH model"
echo "Results will be saved to: $RESULT_FOLDER"

# Run the validation script
echo "Running validation..."
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
  --max_sample=$MAX_SAMPLE \
  --crop_size=$CROP_SIZE \
  $OPT_FLAGS

# Check if the run was successful
if [ $? -eq 0 ]; then
  echo "Evaluation completed successfully"
  echo "Results are available in: $RESULT_FOLDER"
else
  echo "Evaluation failed with error code $?"
fi
