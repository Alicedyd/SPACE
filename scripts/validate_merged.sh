#!/usr/bin/env bash
set -euo pipefail

source /root/miniconda3/etc/profile.d/conda.sh
conda activate DDA

CONFIG_FILE="/root/autodl-tmp/codes/val_configs/all_datasets.yaml"
GPU_ID=0
BATCH_SIZE=64
MAX_SAMPLE=-1
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RESULT_NAME="SPACE_MERGED_MIX_SD/rem_FLUX/d3_CLIP/rem"

# SD_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_REM_codex/SD_REM_model_iters_110000_20260419_145110.pth"
# FLUX_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_REM_codex/FLUX_REM_model_iters_110000_20260420_101642.pth"
# CLIP_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_REM_codex/CLIP_REM_model_iters_85000_20260421_001237.pth"

# SD_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_dinov2/SD-DDA.pth"
# FLUX_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_dinov2/FLUX-DDA.pth"
# CLIP_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_dinov2/CLIP.pth"

SD_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_REM_codex/SD_REM_model_iters_110000_20260419_145110.pth"
FLUX_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_dinov3/FLUX.pth"
CLIP_CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE/best_REM_codex/CLIP_REM_model_iters_85000_20260421_001237.pth"

SD_THRESHOLD=0.5
FLUX_THRESHOLD=0.98
CLIP_THRESHOLD=0.5

SD_ARCH="DINOv3-LoRA:dinov3_vith16plus"
FLUX_ARCH="DINOv3-LoRA:dinov3_vith16plus"
CLIP_ARCH="CLIP-LoRA:ViT-L/14"

# SD_ARCH="DINOv2-LoRA:dinov2_vitl14"
# FLUX_ARCH="DINOv2-LoRA:dinov2_vitl14"
# CLIP_ARCH="CLIP-LoRA:ViT-L/14"

FLUX_CROP_SIZE=224
SD_CROP_SIZE=224
CLIP_CROP_SIZE=224

SD_LORA_RANK=16
FLUX_LORA_RANK=16
CLIP_LORA_RANK=16

SD_LORA_ALPHA=32
FLUX_LORA_ALPHA=32
CLIP_LORA_ALPHA=32

# SD_LORA_RANK=8
# FLUX_LORA_RANK=8
# CLIP_LORA_RANK=8
#
# SD_LORA_ALPHA=1
# FLUX_LORA_ALPHA=1
# CLIP_LORA_ALPHA=1

SD_USE_IS_RESIZE=false
FLUX_USE_IS_RESIZE=false
CLIP_USE_IS_RESIZE=true

SKIP_PATH_CHECK=false
USE_JPEG=false
USE_RESIZE=false
USE_BLUR=false
JPEG=96
RESIZE=0.05
BLUR=2.0

while getopts ":s:f:c:r:g:t:m:" opt; do
  case $opt in
  s) SD_CKPT="$OPTARG" ;;
  f) FLUX_CKPT="$OPTARG" ;;
  c) CLIP_CKPT="$OPTARG" ;;
  r) RESULT_NAME="$OPTARG" ;;
  g) GPU_ID="$OPTARG" ;;
  t) RUN_TIMESTAMP="$OPTARG" ;;
  m) MAX_SAMPLE="$OPTARG" ;;
  \?)
    echo "Usage: $0 -s SD_CKPT -f FLUX_CKPT -c CLIP_CKPT [-r RESULT_NAME] [-g GPU_ID] [-t RUN_TIMESTAMP] [-m MAX_SAMPLE]"
    exit 1
    ;;
  esac
done

RESULT_FOLDER="/root/autodl-tmp/codes/SPACE/result/SPACE_merged/${RESULT_NAME}_${RUN_TIMESTAMP}"
mkdir -p "$RESULT_FOLDER"

OPT_FLAGS=""
$SD_USE_IS_RESIZE && OPT_FLAGS+=" --sd_is_resize"
$FLUX_USE_IS_RESIZE && OPT_FLAGS+=" --flux_is_resize"
$CLIP_USE_IS_RESIZE && OPT_FLAGS+=" --clip_is_resize"
$SKIP_PATH_CHECK && OPT_FLAGS+=" --skip_path_check"
$USE_JPEG && OPT_FLAGS+=" --jpeg ${JPEG}"
$USE_RESIZE && OPT_FLAGS+=" --resize ${RESIZE}"
$USE_BLUR && OPT_FLAGS+=" --blur ${BLUR}"

python validate_merged.py \
  --config "$CONFIG_FILE" \
  --result_folder "$RESULT_FOLDER" \
  --batch_size "$BATCH_SIZE" \
  --gpu_id "$GPU_ID" \
  --max_sample "$MAX_SAMPLE" \
  --sd_ckpt "$SD_CKPT" \
  --flux_ckpt "$FLUX_CKPT" \
  --clip_ckpt "$CLIP_CKPT" \
  --sd_threshold "$SD_THRESHOLD" \
  --flux_threshold "$FLUX_THRESHOLD" \
  --clip_threshold "$CLIP_THRESHOLD" \
  --sd_arch "$SD_ARCH" \
  --flux_arch "$FLUX_ARCH" \
  --clip_arch "$CLIP_ARCH" \
  --sd_crop_size "$SD_CROP_SIZE" \
  --flux_crop_size "$FLUX_CROP_SIZE" \
  --clip_crop_size "$CLIP_CROP_SIZE" \
  --sd_lora_rank "$SD_LORA_RANK" \
  --flux_lora_rank "$FLUX_LORA_RANK" \
  --clip_lora_rank "$CLIP_LORA_RANK" \
  --sd_lora_alpha "$SD_LORA_ALPHA" \
  --flux_lora_alpha "$FLUX_LORA_ALPHA" \
  --clip_lora_alpha "$CLIP_LORA_ALPHA" \
  $OPT_FLAGS

echo "Merged evaluation finished: $RESULT_FOLDER"
