#!/usr/bin/env bash
set -euo pipefail

REAL_LIST="/root/autodl-tmp/datasets/COCO-rec/sd2.0/real"
REAL_LIST_ADD=""
VAE_PATH="/root/autodl-tmp/datasets/COCO-rec/sd2.0/sd2.0/"
VAE_PATH_ADD=""
FAKE_LIST="${VAE_PATH}"
DATA_MODE="mscoco"
ARCH="DINOv3-LoRA:dinov3_vith16plus"
LORA_RANK=16
LORA_ALPHA=32
OPTIM="adam"
NITER=2
CROP_SIZE=224
BATCH_SIZE=16
LEARNING_RATE=1e-4
DOWN_RESIZE_FACTOR=0.2
UPPER_RESIZE_FACTOR=3.5
P_JPEG_FAKE=1.0
P_PNG_REAL=0.0
JPEG_QUALITY=100
P_PIXELMIX=0.2
R_PIXELMIX=0.8
METH_PIXELMIX="uniform"
P_FREQMIX=0.0
R_FREQMIX=0.8
METH_FREQMIX="uniform"
QUALITY_JSON="/root/autodl-tmp/codes/DDA/util_files/MSCOCO_train2017.json"
EXP_ADD="DDA_REM"
CHECKPOINTS_DIR="/root/autodl-tmp/codes/ckpt/checkpoints_SPACE"
USE_CONTRASTIVE=true
USE_FOCAL_LOSS=false
USE_RANDOMSCALE=true
GPU_ID=0
MIX_COLOR_SPACE="RGB"
ACCUM_STEPS=16
EXP_SUFFIX=""
RUN_TIMESTAMP="$(date '+%Y%m%d_%H%M%S')"

while getopts ":g:c:a:n:t:" opt; do
  case $opt in
    g) GPU_ID="$OPTARG" ;;
    c) MIX_COLOR_SPACE="$OPTARG" ;;
    a) ACCUM_STEPS="$OPTARG" ;;
    n) EXP_SUFFIX="$OPTARG" ;;
    t) RUN_TIMESTAMP="$OPTARG" ;;
    \?)
      echo "Usage: $0 [-g GPU_ID] [-c MIX_COLOR_SPACE] [-a ACCUM_STEPS] [-n EXP_SUFFIX] [-t RUN_TIMESTAMP]"
      exit 1
      ;;
  esac
done

OPT_FLAGS=""
$USE_CONTRASTIVE && OPT_FLAGS+=" --contrastive"
$USE_FOCAL_LOSS && OPT_FLAGS+=" --use_focal_loss"
$USE_RANDOMSCALE && OPT_FLAGS+=" --use_randomscale"

EXP_NAME="EXP_${EXP_ADD}_DINO_${CROP_SIZE}_JPEGaug_lora${LORA_RANK}_lr${LEARNING_RATE}_BS_${BATCH_SIZE}_ACC_${ACCUM_STEPS}_colorspace_${MIX_COLOR_SPACE}"
[[ -n "${EXP_SUFFIX}" ]] && EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
EXP_NAME="${EXP_NAME}_${RUN_TIMESTAMP}"

mkdir -p "$CHECKPOINTS_DIR"

echo ">>> Starting training: ${EXP_NAME}"
echo "GPU: ${GPU_ID} | Color: ${MIX_COLOR_SPACE} | Accum: ${ACCUM_STEPS} | Timestamp: ${RUN_TIMESTAMP}"

python train.py \
  --gpu_ids "${GPU_ID}" \
  --name "${EXP_NAME}" \
  --cropSize "${CROP_SIZE}" \
  --real_list_path "${REAL_LIST}" \
  --real_list_path_add "${REAL_LIST_ADD}" \
  --fake_list_path "${FAKE_LIST}" \
  --data_mode "${DATA_MODE}" \
  --arch "${ARCH}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LEARNING_RATE}" \
  --accumulation_steps "${ACCUM_STEPS}" \
  --optim "${OPTIM}" \
  --niter "${NITER}" \
  --lora_rank "${LORA_RANK}" \
  --lora_alpha "${LORA_ALPHA}" \
  --vae_models "${VAE_PATH}" \
  --vae_models_add "${VAE_PATH_ADD}" \
  --p_jpeg_fake "${P_JPEG_FAKE}" \
  --p_png_real "${P_PNG_REAL}" \
  --jpeg_quality "${JPEG_QUALITY}" \
  --checkpoints_dir "${CHECKPOINTS_DIR}" \
  --down_resize_factors "${DOWN_RESIZE_FACTOR}" \
  --upper_resize_factors "${UPPER_RESIZE_FACTOR}" \
  --quality_json "${QUALITY_JSON}" \
  --mix_color_space "${MIX_COLOR_SPACE}" \
  --p_pixelmix "${P_PIXELMIX}" \
  --r_pixelmix "${R_PIXELMIX}" \
  --meth_pixelmix "${METH_PIXELMIX}" \
  --p_freqmix "${P_FREQMIX}" \
  --r_freqmix "${R_FREQMIX}" \
  --meth_freqmix "${METH_FREQMIX}" \
  $OPT_FLAGS

echo ">>> Training finished. Checkpoints saved in: ${CHECKPOINTS_DIR}/${EXP_NAME}"
