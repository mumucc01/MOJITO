#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../setup_env.sh"

TRAIN_TEST_SPLIT=navtest
CKPT="${MOJITO_ROOT}/weights/checkpoints/mojito_navsim.ckpt"
export CUDA_VISIBLE_DEVICES='3'
python "${NAVSIM_DEVKIT_ROOT}/planning/script/run_pdm_score.py" \
    train_test_split="${TRAIN_TEST_SPLIT}" \
    agent=diffusiondrive_agent \
    worker=single_machine_thread_pool \
    agent.checkpoint_path="${CKPT}" \
    experiment_name=mojito_eval
