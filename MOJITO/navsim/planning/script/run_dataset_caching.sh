#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../../setup_env.sh"

python "${NAVSIM_DEVKIT_ROOT}/planning/script/run_dataset_caching.py" \
    agent=diffusiondrive_agent \
    experiment_name=training_mojito_agent \
    train_test_split=navtrain \
    worker=single_machine_thread_pool
