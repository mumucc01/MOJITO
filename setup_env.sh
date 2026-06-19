#!/usr/bin/env bash
# Source this file before training or evaluation:
#   source setup_env.sh

export MOJITO_ROOT="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/MOJITO"
export MOJITO_PKG_ROOT="${MOJITO_ROOT}/MOJITO"
export DIFFUSION_PLANNER_ROOT="${MOJITO_ROOT}/Diffusion-Planner"
export NUPLAN_DEVKIT_ROOT="${MOJITO_ROOT}/nuplan-devkit"
export NAVSIM_DEVKIT_ROOT="${MOJITO_PKG_ROOT}/navsim"

export DINOV3_ROOT="${MOJITO_PKG_ROOT}/navsim/agents/diffusiondrive/dinov3"

# navsim lives under MOJITO/MOJITO; nuplan under nuplan-devkit; dinov3 hubconf needs repo root on path
export HYDRA_FULL_ERROR=1

# Avoid duplicating PYTHONPATH when setup_env is sourced multiple times
if [[ -z "${MOJITO_ENV_LOADED:-}" ]]; then
  export PYTHONPATH="${MOJITO_PKG_ROOT}:${NUPLAN_DEVKIT_ROOT}:${DIFFUSION_PLANNER_ROOT}:${DINOV3_ROOT}:${MOJITO_ROOT}"
  export MOJITO_ENV_LOADED=1
fi

# --- local defaults (override by exporting before source if needed) ---
export OPENSCENE_DATA_ROOT="${OPENSCENE_DATA_ROOT:-/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive_raw/DiffusionDrive/dataset}"
export NAVSIM_EXP_ROOT="${NAVSIM_EXP_ROOT:-/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/MOJITO/MOJITO/exp}"
export MOJITO_CACHE_PATH="${MOJITO_CACHE_PATH:-/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/AD-Dataset/V8}"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NUPLAN_MAPS_ROOT="${OPENSCENE_DATA_ROOT}/maps"

echo "MOJITO_ROOT=${MOJITO_ROOT}"
echo "MOJITO_PKG_ROOT=${MOJITO_PKG_ROOT}"
echo "PYTHONPATH=${PYTHONPATH}"
echo "OPENSCENE_DATA_ROOT=${OPENSCENE_DATA_ROOT}"
echo "NAVSIM_EXP_ROOT=${NAVSIM_EXP_ROOT}"
echo "MOJITO_CACHE_PATH=${MOJITO_CACHE_PATH}"
