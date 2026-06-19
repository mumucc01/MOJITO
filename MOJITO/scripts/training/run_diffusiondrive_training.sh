#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../setup_env.sh"

export MASTER_PORT=2345
export CUDA_VISIBLE_DEVICES='3'
python "${NAVSIM_DEVKIT_ROOT}/planning/script/run_training.py" \
    agent=diffusiondrive_agent \
    experiment_name=training_mojito_agent \
    train_test_split=navtrain \
    split=trainval \
    trainer.params.max_epochs=250 \
    cache_path="${MOJITO_CACHE_PATH}" \
    use_cache_without_dataset=True \
    force_cache_computation=False
