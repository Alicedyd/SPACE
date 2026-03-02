# #!/bin/bash

# # ===== 固定参数 =====
# USE_JPEG=false
# USE_RESIZE=false
# USE_BLUR=false
# JPEG=96
# RESIZE=2.0
# BLUR=1.0

# ARCH="DINOv2-LoRA:dinov2_vitl14"
# CONFIG_FILE="../val_configs/all_datasets.yaml"
# LORA_RANK=8
# LORA_ALPHA=1
# BATCH_SIZE=64
# JPEG_QUALITY=100
# GPU_ID=2
# MAX_SAMPLE=-1
# USE_PATCH=false
# SCRIPT="validate_new.py"
# if $USE_PATCH; then
#     SCRIPT="validate_patch.py"
# fi

# OPT_FLAGS=""
# $USE_JPEG && OPT_FLAGS+=" --jpeg ${JPEG}"
# $USE_RESIZE && OPT_FLAGS+=" --resize ${RESIZE}"
# $USE_BLUR && OPT_FLAGS+=" --blur ${BLUR}"

# # ===== 依次测试所有 .pth =====
# MODEL_DIR="/data0/dda/ckpt/checkpoints_DINO_224_LR1e-4_single_run/COCO_flux_dev_norandomjpeg_nomixup_DINO_224_JPEGaug_lora8_lr1e-4_BS_16_ACC_1_colorspace_RGB"
# for CKPT in "$MODEL_DIR"/*.pth; do
#     MODEL_NAME=$(basename "$CKPT" .pth)
#     RESULT_FOLDER="./result/dda_flux_dev_norandomjpeg_nomixup/${MODEL_NAME}"
#     RESULT_FILE="$RESULT_FOLDER/results.csv"

#     # 如果结果文件已存在，则跳过
#     if [ -f "$RESULT_FILE" ]; then
# 	echo "Skipping $MODEL_NAME because $RESULT_FILE already exists."
#         continue
#     fi

#     mkdir -p "$RESULT_FOLDER"

#     echo "=== Configuration ===" > "$RESULT_FOLDER/config_summary.txt"
#     echo "Architecture: $ARCH" >> "$RESULT_FOLDER/config_summary.txt"
#     echo "Checkpoint: $CKPT" >> "$RESULT_FOLDER/config_summary.txt"
#     echo "Config file: $CONFIG_FILE" >> "$RESULT_FOLDER/config_summary.txt"
#     echo "LoRA rank: $LORA_RANK" >> "$RESULT_FOLDER/config_summary.txt"
#     echo "LoRA alpha: $LORA_ALPHA" >> "$RESULT_FOLDER/config_summary.txt"
#     echo "JPEG quality: $JPEG_QUALITY" >> "$RESULT_FOLDER/config_summary.txt"
#     echo "Run date: $(date)" >> "$RESULT_FOLDER/config_summary.txt"

#     echo "Starting evaluation for $MODEL_NAME"
#     python $SCRIPT \
#         --arch="$ARCH" \
#         --config="$CONFIG_FILE" \
#         --ckpt="$CKPT" \
#         --result_folder="$RESULT_FOLDER" \
#         --batch_size="$BATCH_SIZE" \
#         --lora_rank="$LORA_RANK" \
#         --lora_alpha="$LORA_ALPHA" \
#         --jpeg_quality="$JPEG_QUALITY" \
#         --gpu_id="$GPU_ID" \
#         --max_sample=$MAX_SAMPLE \
#         --crop_size=336 \
#         $OPT_FLAGS

#     if [ $? -eq 0 ]; then
#         echo "$MODEL_NAME completed successfully"
#     else
#         echo "$MODEL_NAME failed"
#     fi
# done

#!/bin/bash
set -u

#################### 固定参数（保持你原来写法） ####################
USE_JPEG=false
USE_RESIZE=false
USE_BLUR=false
JPEG=96
RESIZE=2.0
BLUR=1.0

CROP_SIZE=336

ARCH="DINOv2-LoRA:dinov2_vitl14"
CONFIG_FILE="/root/autodl-tmp/codes/val_configs/all_datasets.yaml"
LORA_RANK=8
LORA_ALPHA=1
BATCH_SIZE=64
JPEG_QUALITY=100
MAX_SAMPLE=-1
USE_PATCH=false
SCRIPT="validate.py"
$USE_PATCH && SCRIPT="validate_patch.py"

MODEL_DIR="/data0/dda/ckpt/checkpoints_DINO_336_LR1e-4_single_run/EXP_FLUX_JPEG_90_nomixup_DINO_336_JPEGaug_lora8_lr1e-4_BS_16_ACC_1_colorspace_RGB"
RESULT_ROOT="/root/autodl-tmp/codes/DDAresult/flux_336_nomixup"

OPT_FLAGS=""
$USE_JPEG && OPT_FLAGS+=" --jpeg ${JPEG}"
$USE_RESIZE && OPT_FLAGS+=" --resize ${RESIZE}"
$USE_BLUR && OPT_FLAGS+=" --blur ${BLUR}"

#################### 新增：并行控制参数（可用环境变量覆盖） ####################
NUM_GPUS="${NUM_GPUS:-2}"   # 并行GPU数量
START_GPU="${START_GPU:-6}" # 起始GPU编号

#################### 生成GPU列表与状态 ####################
declare -a DEVICES=()
for ((g = START_GPU; g < START_GPU + NUM_GPUS; g++)); do DEVICES+=("$g"); done
declare -A PIDS=() # 每个GPU当前任务的PID

#################### 准备锁目录（避免并发重复跑同一ckpt） ####################
LOCK_DIR="${RESULT_ROOT}/.locks"
mkdir -p "$LOCK_DIR"

#################### 工具函数 ####################
run_eval() {
  local gpu_phys="$1"
  local ckpt="$2"

  local model_name
  model_name="$(basename "$ckpt" .pth)"
  local result_folder="${RESULT_ROOT}/${model_name}"
  local result_file="${result_folder}/results.csv"
  local my_lock="${LOCK_DIR}/${model_name}.lock"

  # 原子文件锁：谁能创建这目录，谁就拥有该ckpt执行权
  if ! mkdir "$my_lock" 2>/dev/null; then
    echo "[GPU ${gpu_phys}] Lock exists, skip duplicate: ${model_name}"
    return 0
  fi

  # 再次保险：结果已存在直接放弃（释放锁）
  if [ -f "$result_file" ]; then
    echo "[GPU ${gpu_phys}] Skipping ${model_name} (already exists)."
    rmdir "$my_lock" 2>/dev/null || true
    return 0
  fi

  mkdir -p "$result_folder"

  echo "[GPU ${gpu_phys}] Start: ${model_name}"
  CUDA_VISIBLE_DEVICES="${gpu_phys}" \
    python "$SCRIPT" \
    --arch="$ARCH" \
    --config="$CONFIG_FILE" \
    --ckpt="$ckpt" \
    --result_folder="$result_folder" \
    --batch_size="$BATCH_SIZE" \
    --lora_rank="$LORA_RANK" \
    --lora_alpha="$LORA_ALPHA" \
    --jpeg_quality="$JPEG_QUALITY" \
    --gpu_id=0 \
    --max_sample="$MAX_SAMPLE" \
    --crop_size=$CROP_SIZE \
    $OPT_FLAGS

  local status=$?
  if [ $status -eq 0 ]; then
    echo "[GPU ${gpu_phys}] Done : ${model_name}"
  else
    echo "[GPU ${gpu_phys}] Fail : ${model_name} (code $status)"
  fi
  # 任务结束释放锁
  rmdir "$my_lock" 2>/dev/null || true
  return $status
}

wait_any() {
  if wait -n 2>/dev/null; then return 0; fi
  for gg in "${DEVICES[@]}"; do
    local p=${PIDS[$gg]:-}
    if [ -n "$p" ] && kill -0 "$p" 2>/dev/null; then
      wait "$p" || true
      return 0
    fi
  done
}

cleanup_finished() {
  for gg in "${DEVICES[@]}"; do
    local p=${PIDS[$gg]:-}
    if [ -n "$p" ] && ! kill -0 "$p" 2>/dev/null; then unset PIDS[$gg]; fi
  done
}

#################### 主流程（队列式分配，保证每GPU不同ckpt） ####################
shopt -s nullglob
CKPTS=("$MODEL_DIR"/*.pth)
TOTAL=${#CKPTS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "No .pth files found in $MODEL_DIR"
  exit 0
fi
echo "Found $TOTAL checkpoints."
echo "GPUs: ${DEVICES[*]}"

# 去重（同basename只跑一次，避免 result/<basename> 冲突）
declare -A USED_BN=()
FILTERED=()
for CK in "${CKPTS[@]}"; do
  bn="$(basename "$CK" .pth)"
  if [[ -n "${USED_BN[$bn]+x}" ]]; then
    echo ">> [global-skip-name] duplicate basename '${bn}', skip path: $CK"
    continue
  fi
  USED_BN[$bn]=1
  FILTERED+=("$CK")
done
CKPTS=("${FILTERED[@]}")
TOTAL=${#CKPTS[@]}
echo "After basename dedup: $TOTAL checkpoints."

# 全局索引：保证每次取“下一个”ckpt，不会重复分配
NEXT=0

# 先尽量填满所有GPU
for gpu in "${DEVICES[@]}"; do
  if [ $NEXT -ge $TOTAL ]; then break; fi
  ck="${CKPTS[$NEXT]}"
  # 若结果已存在，直接跳过并推进索引
  model_name="$(basename "$ck" .pth)"
  if [ -f "${RESULT_ROOT}/${model_name}/results.csv" ]; then
    echo ">> [global-skip] ${model_name} already has results."
    ((NEXT++))
    # 重试当前gpu填充
    continue
  fi
  run_eval "$gpu" "$ck" &
  PIDS[$gpu]=$!
  ((NEXT++))
done

# 持续分配剩余ckpt：只要有GPU空闲就取“下一个”
while [ $NEXT -lt $TOTAL ]; do
  cleanup_finished
  idle_found=0
  for gpu in "${DEVICES[@]}"; do
    pid=${PIDS[$gpu]:-}
    if [ -z "${pid}" ] || ! kill -0 "$pid" 2>/dev/null; then
      # 这张GPU空闲，拿队列中的下一个ckpt
      while [ $NEXT -lt $TOTAL ]; do
        ck="${CKPTS[$NEXT]}"
        model_name="$(basename "$ck" .pth)"
        # 结果已存在则推进索引继续找下一个
        if [ -f "${RESULT_ROOT}/${model_name}/results.csv" ]; then
          echo ">> [global-skip] ${model_name} already has results."
          ((NEXT++))
          continue
        fi
        run_eval "$gpu" "$ck" &
        PIDS[$gpu]=$!
        ((NEXT++))
        idle_found=1
        break
      done
    fi
    [ $NEXT -ge $TOTAL ] && break
  done

  # 如果这一轮没有空闲GPU被填充，则等待任一任务结束
  if [ $idle_found -eq 0 ]; then
    wait_any
  fi
done

# 等全部任务结束
for gpu in "${DEVICES[@]}"; do
  pid=${PIDS[$gpu]:-}
  [ -n "$pid" ] && wait "$pid" || true
done

echo "All evaluations finished."
