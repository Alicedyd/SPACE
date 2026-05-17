#!/usr/bin/env bash

if [[ -z "${SPACE_CODES_ROOT:-}" ]]; then
  if [[ -d "/root/autodl-tmp/codes" ]]; then
    export SPACE_CODES_ROOT="/root/autodl-tmp/codes"
  elif [[ -d "/gsdata/home/crx/jw/codes" ]]; then
    export SPACE_CODES_ROOT="/gsdata/home/crx/jw/codes"
  else
    export SPACE_CODES_ROOT="/root/autodl-tmp/codes"
  fi
fi

if [[ -z "${SPACE_DATA_ROOT:-}" ]]; then
  export SPACE_DATA_ROOT="$(dirname "$SPACE_CODES_ROOT")/datasets"
fi

export SPACE_MODEL_ROOT="${SPACE_MODEL_ROOT:-$SPACE_CODES_ROOT/model_pth}"
export SPACE_CKPT_ROOT="${SPACE_CKPT_ROOT:-$SPACE_CODES_ROOT/ckpt}"
export SPACE_VAL_CONFIG_ROOT="${SPACE_VAL_CONFIG_ROOT:-$SPACE_CODES_ROOT/val_configs}"
export SPACE_PROJECT_ROOT="${SPACE_PROJECT_ROOT:-$SPACE_CODES_ROOT/SPACE}"
export DDA_PROJECT_ROOT="${DDA_PROJECT_ROOT:-$SPACE_CODES_ROOT/DDA}"
