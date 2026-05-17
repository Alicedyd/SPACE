#!/usr/bin/env bash

# Source this file before running SPACE scripts:
#   source setup_space_env.sh gsdata
#   source setup_space_env.sh autodl
#
# You may also override any path before or after sourcing, for example:
#   SPACE_DATA_ROOT=/my/datasets source setup_space_env.sh gsdata

SPACE_PROFILE="${1:-auto}"

case "$SPACE_PROFILE" in
  gsdata)
    DEFAULT_CODES_ROOT="/gsdata/home/crx/jw/codes"
    ;;
  autodl)
    DEFAULT_CODES_ROOT="/root/autodl-tmp/codes"
    ;;
  auto)
    if [[ -d "/gsdata/home/crx/jw/codes" ]]; then
      DEFAULT_CODES_ROOT="/gsdata/home/crx/jw/codes"
    elif [[ -d "/root/autodl-tmp/codes" ]]; then
      DEFAULT_CODES_ROOT="/root/autodl-tmp/codes"
    else
      DEFAULT_CODES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    fi
    ;;
  *)
    DEFAULT_CODES_ROOT="$SPACE_PROFILE"
    ;;
esac

export SPACE_CODES_ROOT="${SPACE_CODES_ROOT:-$DEFAULT_CODES_ROOT}"
export SPACE_PROJECT_ROOT="${SPACE_PROJECT_ROOT:-$SPACE_CODES_ROOT/SPACE}"
export SPACE_DATA_ROOT="${SPACE_DATA_ROOT:-$(dirname "$SPACE_CODES_ROOT")/datasets}"
export SPACE_MODEL_ROOT="${SPACE_MODEL_ROOT:-$SPACE_CODES_ROOT/model_pth}"
export SPACE_CKPT_ROOT="${SPACE_CKPT_ROOT:-$SPACE_CODES_ROOT/ckpt}"
export SPACE_VAL_CONFIG_ROOT="${SPACE_VAL_CONFIG_ROOT:-$SPACE_CODES_ROOT/val_configs}"
export PERCEPTION_MODELS_PATH="${PERCEPTION_MODELS_PATH:-$SPACE_MODEL_ROOT/perception_models}"

echo "SPACE_CODES_ROOT=$SPACE_CODES_ROOT"
echo "SPACE_PROJECT_ROOT=$SPACE_PROJECT_ROOT"
echo "SPACE_DATA_ROOT=$SPACE_DATA_ROOT"
echo "SPACE_MODEL_ROOT=$SPACE_MODEL_ROOT"
echo "SPACE_CKPT_ROOT=$SPACE_CKPT_ROOT"
echo "SPACE_VAL_CONFIG_ROOT=$SPACE_VAL_CONFIG_ROOT"
echo "PERCEPTION_MODELS_PATH=$PERCEPTION_MODELS_PATH"

