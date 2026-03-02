#!/usr/bin/env bash
# 单次训练脚本（由消融脚本改写）
# 用法示例：
#   bash train_single_dino336_lora_jpeg.sh \
#     -g 0 \
#     -c "RGB" \
#     -a 8 \
#     -n "exp1"
#
# 仅一次训练，不做循环与并行。

set -euo pipefail

# ========= 基础配置（与原脚本保持一致） =========
REAL_LIST="/root/autodl-tmp/datasets/COCO-rec/flux-dev/real"
REAL_LIST_ADD=""                 # 可选的额外 real 列表（留空即可）
FAKE_LIST="/path/to/fake/images" # ← 请改成你的假图像列表/目录
DATA_MODE="mscoco"
ARCH="DINOv2-LoRA:dinov2_vitl14"
LORA_RANK=8
LORA_ALPHA=1
OPTIM="adam"
NITER=2
BATCH_SIZE=16
CROP_SIZE=336
LEARNING_RATE=1e-4

DOWN_RESIZE_FACTOR=0.2
UPPER_RESIZE_FACTOR=3.5
P_JPEG_FAKE=0.5
P_PNG_REAL=0.0
JPEG_QUALITY=100
P_PIXELMIX=0.2
R_PIXELMIX=0.8
METH_PIXELMIX="uniform"
P_FREQMIX=0.2
R_FREQMIX=0.8
METH_FREQMIX="uniform"

QUALITY_JSON="./util_files/MSCOCO_train2017.json"

# 用来添加不固定的实验名称
EXP_ADD="flux_logitadjustment"

VAE_PATH="/root/autodl-tmp/datasets/COCO-rec/flux-dev/flux-dev"
VAE_PATH_ADD="" # 可选的额外 VAE（留空即可）
CHECKPOINTS_DIR="/root/autodl-tmp/codes/ckpt/checkpoints_DINO_${CROP_SIZE}_LR${LEARNING_RATE}_single_run"

USE_CONTRASTIVE=true
USE_FOCAL_LOSS=false
USE_LOGIT_ADJUSTMENT=true

OPT_FLAGS=""

if $USE_CONTRASTIVE; then
  OPT_FLAGS+=" --contrastive"
  echo "Will use contrastive learning"
fi

if $USE_FOCAL_LOSS; then
  OPT_FLAGS+=" --use_focal_loss"
  echo "Will replace BCEWithLogitsLoss with FocalLoss"
fi

if $USE_LOGIT_ADJUSTMENT; then
  OPT_FLAGS+="  --use_logit_adjustment"
  echo "Will use logit adjustment"
fi

# ========= 可通过命令行覆盖的参数 =========
GPU_ID=0              # -g
MIX_COLOR_SPACE="RGB" # -c
ACCUM_STEPS=4         # -a
EXP_SUFFIX=""         # -n 仅用于区分实验名，可选

# 解析命令行参数
while getopts ":g:c:a:n:" opt; do
  case $opt in
  g) GPU_ID="$OPTARG" ;;
  c) MIX_COLOR_SPACE="$OPTARG" ;;
  a) ACCUM_STEPS="$OPTARG" ;;
  n) EXP_SUFFIX="$OPTARG" ;;
  \?)
    echo "用法: $0 [-g GPU_ID] [-c MIX_COLOR_SPACE] [-a ACCUM_STEPS] [-n EXP_SUFFIX]"
    exit 1
    ;;
  esac
done

# ========= 组装实验名（去掉原脚本中未定义的 VAE_MODEL）=========
EXP_NAME="EXP_${EXP_ADD}_DINO_${CROP_SIZE}_JPEGaug_lora${LORA_RANK}_lr${LEARNING_RATE}_BS_${BATCH_SIZE}_ACC_${ACCUM_STEPS}_colorspace_${MIX_COLOR_SPACE}"
if [[ -n "${EXP_SUFFIX}" ]]; then
  EXP_NAME="${EXP_NAME}_${EXP_SUFFIX}"
fi

echo ">>> 开始训练：${EXP_NAME}"
echo "GPU: ${GPU_ID} | 颜色空间: ${MIX_COLOR_SPACE} | 累积步数: ${ACCUM_STEPS}"

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

echo ">>> 训练完成。权重保存在：${CHECKPOINTS_DIR}"
