#!/usr/bin/env bash
# Install MOJITO Python dependencies (run from MOJITO repo root).
#
# Usage:
#   conda activate mojito   # or your env
#   bash scripts/install_env.sh
#
# Reference env: /lpai/volumes/base-3da-ali-sh-mix/chengzhijing/conda_env/diffusion_drive

set -euo pipefail

MOJITO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOJITO_PKG="${MOJITO_ROOT}/MOJITO"
POINTNET2="${MOJITO_PKG}/navsim/agents/diffusiondrive/Uni3D/Pointnet2_PyTorch"

echo "MOJITO_ROOT=${MOJITO_ROOT}"
echo "Python: $(which python) ($(python --version))"

# 1) NAVSIM + nuPlan base deps
pip install -r "${MOJITO_PKG}/requirements.txt"

# 2) MOJITO-specific extras
pip install -r "${MOJITO_ROOT}/requirements-mojito.txt"

# 3) PointNet++ (required for MOJITO LiDAR path)
if [[ -d "${POINTNET2}" ]]; then
  echo "Installing pointnet2_ops from ${POINTNET2}"
  pip install -e "${POINTNET2}"
else
  echo "WARN: Pointnet2_PyTorch not found at ${POINTNET2}"
fi

echo ""
echo "Done. Verify with:"
echo "  cd ${MOJITO_ROOT} && source setup_env.sh"
echo "  python -c \"from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent; print('OK')\""
