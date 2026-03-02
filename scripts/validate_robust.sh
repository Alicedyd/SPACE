#!/bin/bash
set -euo pipefail

# ========= 基础配置（按需修改） =========
ARCH="DINOv2-LoRA:dinov2_vitl14" # 模型结构
# FLUX 224 nomixup
# CKPT="/root/autodl-tmp/codes/ckpt/checkpoints_DINO_224_LR1e-4_single_run/COCO_flux_dev_randomjpeg90_nomixup_DINO_224_JPEGaug_lora8_lr1e-4_BS_16_ACC_1_colorspace_RGB/model_iters_100000.pth"
# SD 336 mixup (DDA)
CKPT="/root/autodl-tmp/codes/ckpt/model_iters_130000.pth"
CONFIG_FILE="/root/autodl-tmp/codes/val_configs/GenImage.yaml" # 验证配置
BATCH_SIZE=64
LORA_RANK=8
LORA_ALPHA=1
GPU_ID=0
CROP_SIZE=336
MAX_SAMPLE=-1                         # -1 表示全量
SAVE_BAD_CASE=false                   # 是否保存误判样例
SKIP_PATH_CHECK=false                 # 是否跳过路径检查
BASE_RESULT="./result/dda_robustness" # 结果总目录
SCRIPT="validate.py"

# ========= 扰动设置 =========
# JPEG_LIST=(100 90 80 70 60)
# RESIZE_LIST=(0.5 0.75 1 1.25 1.5 1.75 2)
# BLUR_LIST=(0 0.5 1 1.5 2)
RESIZE_LIST=(1.5 1.75)

# ========= 功能函数 =========
run_eval() {
  local mode="$1"  # jpeg | resize | blur
  local value="$2" # 具体数值
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"

  # 构造结果目录（按模式与数值区分）
  local result_dir="${BASE_RESULT}/${mode^^}_${value}"
  mkdir -p "${result_dir}"

  # 记录本次配置
  {
    echo "=== Configuration ==="
    echo "Run time: ${stamp}"
    echo "Mode: ${mode}"
    echo "Value: ${value}"
    echo "Architecture: ${ARCH}"
    echo "Checkpoint: ${CKPT}"
    echo "Config file: ${CONFIG_FILE}"
    echo "LoRA rank: ${LORA_RANK}"
    echo "LoRA alpha: ${LORA_ALPHA}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "GPU ID: ${GPU_ID}"
    echo "Crop size: ${CROP_SIZE}"
    echo "Max sample: ${MAX_SAMPLE}"
    echo "Save bad case: ${SAVE_BAD_CASE}"
    echo "Skip path check: ${SKIP_PATH_CHECK}"
  } >"${result_dir}/config_summary.txt"

  # 通用可选flag
  OPT_FLAGS=()
  $SAVE_BAD_CASE && OPT_FLAGS+=("--save_bad_case")
  $SKIP_PATH_CHECK && OPT_FLAGS+=("--skip_path_check")

  # 根据模式只启用一个扰动参数
  case "${mode}" in
  jpeg)
    echo "[RUN] JPEG quality = ${value} -> ${result_dir}"
    python "${SCRIPT}" \
      --arch="${ARCH}" \
      --config="${CONFIG_FILE}" \
      --ckpt="${CKPT}" \
      --result_folder="${result_dir}" \
      --batch_size="${BATCH_SIZE}" \
      --lora_rank="${LORA_RANK}" \
      --lora_alpha="${LORA_ALPHA}" \
      --jpeg="${value}" \
      --gpu_id="${GPU_ID}" \
      --max_sample="${MAX_SAMPLE}" \
      --crop_size="${CROP_SIZE}" \
      "${OPT_FLAGS[@]}"
    ;;

  resize)
    echo "[RUN] RESIZE scale = ${value} -> ${result_dir}"
    python "${SCRIPT}" \
      --arch="${ARCH}" \
      --config="${CONFIG_FILE}" \
      --ckpt="${CKPT}" \
      --result_folder="${result_dir}" \
      --batch_size="${BATCH_SIZE}" \
      --lora_rank="${LORA_RANK}" \
      --lora_alpha="${LORA_ALPHA}" \
      --resize="${value}" \
      --gpu_id="${GPU_ID}" \
      --max_sample="${MAX_SAMPLE}" \
      --crop_size="${CROP_SIZE}" \
      "${OPT_FLAGS[@]}"
    ;;

  blur)
    echo "[RUN] BLUR sigma = ${value} -> ${result_dir}"
    python "${SCRIPT}" \
      --arch="${ARCH}" \
      --config="${CONFIG_FILE}" \
      --ckpt="${CKPT}" \
      --result_folder="${result_dir}" \
      --batch_size="${BATCH_SIZE}" \
      --lora_rank="${LORA_RANK}" \
      --lora_alpha="${LORA_ALPHA}" \
      --blur="${value}" \
      --gpu_id="${GPU_ID}" \
      --max_sample="${MAX_SAMPLE}" \
      --crop_size="${CROP_SIZE}" \
      "${OPT_FLAGS[@]}"
    ;;

  *)
    echo "Unknown mode: ${mode}" >&2
    exit 1
    ;;
  esac

  if [ $? -eq 0 ]; then
    echo "[OK] ${mode^^}=${value} completed. Results: ${result_dir}"
  else
    echo "[ERR] ${mode^^}=${value} failed." >&2
  fi
}

# ========= 主流程：按顺序执行三类鲁棒性测试 =========
mkdir -p "${BASE_RESULT}"

echo "===== (1/3) JPEG Robustness ====="
for q in "${JPEG_LIST[@]}"; do
  run_eval "jpeg" "${q}"
done

echo "===== (2/3) RESIZE Robustness ====="
for s in "${RESIZE_LIST[@]}"; do
  run_eval "resize" "${s}"
done

echo "===== (3/3) BLUR Robustness ====="
for b in "${BLUR_LIST[@]}"; do
  run_eval "blur" "${b}"
done

echo "All robustness tests finished. Root results: ${BASE_RESULT}"
