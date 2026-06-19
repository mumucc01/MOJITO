"""Project path helpers for MOJITO open-source release."""
from pathlib import Path

MOJITO_REPO_ROOT = Path(__file__).resolve().parents[2]
MOJITO_PKG_ROOT = MOJITO_REPO_ROOT / "MOJITO"
DIFFUSION_PLANNER_ROOT = MOJITO_REPO_ROOT / "Diffusion-Planner"
NUPLAN_DEVKIT_ROOT = MOJITO_REPO_ROOT / "nuplan-devkit"
NAVSIM_ROOT = MOJITO_PKG_ROOT / "navsim"
AGENT_ROOT = NAVSIM_ROOT / "agents" / "diffusiondrive"
DINOV3_ROOT = AGENT_ROOT / "dinov3"
WEIGHTS_ROOT = MOJITO_REPO_ROOT / "weights"
PRETRAINED_WEIGHTS = WEIGHTS_ROOT / "pretrained"
CHECKPOINTS_DIR = WEIGHTS_ROOT / "checkpoints"


def weight_path(*parts: str) -> str:
    return str(WEIGHTS_ROOT.joinpath(*parts))


def pretrained_path(*parts: str) -> str:
    return str(PRETRAINED_WEIGHTS.joinpath(*parts))


def checkpoint_path(*parts: str) -> str:
    return str(CHECKPOINTS_DIR.joinpath(*parts))
