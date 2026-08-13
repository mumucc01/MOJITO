# MOJITO (**ECCV 2026**)

[![arXiv](https://img.shields.io/badge/arXiv-2607.23511-b31b1b.svg)](https://arxiv.org/abs/2607.23511)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/mumucc1/MOJITO)

MOJITO is an end-to-end autonomous driving framework built upon [DiffusionDrive](https://github.com/hustvl/DiffusionDrive). It introduces hierarchical three-modal fusion across **vision**, **LiDAR**, and **trajectory/action representations**, enabling unified perception and planning for end-to-end autonomous driving.

## Project Structure

```text
MOJITO/
├── MOJITO/               # Main codebase: NAVSIM agent, training, and evaluation
├── Diffusion-Planner/    # Diffusion-based trajectory planning module
├── nuplan-devkit/        # nuPlan development toolkit
├── weights/              # Pre-trained weights and MOJITO checkpoints
├── scripts/              # Environment installation scripts
├── setup_env.sh          # Environment configuration script
└── README.md
```

---

# Local Setup Guide

## Step 1: Environment Setup

Please refer to [docs/environment.md](docs/environment.md) for the complete dependency list, installation instructions, and solutions to common issues.

### Installation from Scratch

```bash
conda create -n mojito python=3.9 -y
conda activate mojito

cd /path/to/MOJITO
bash scripts/install_env.sh
```

---

## Step 2: Configure Environment Variables

Before running training or evaluation, load the environment configuration:

```bash
cd /path/to/MOJITO
source setup_env.sh
```

`setup_env.sh` configures the dataset paths, experiment directories, and training cache locations.

To use custom paths, export the corresponding environment variables before sourcing the script:

```bash
export OPENSCENE_DATA_ROOT=/your/dataset
export NAVSIM_EXP_ROOT=/your/exp
export MOJITO_CACHE_PATH=/your/training_cache

source setup_env.sh
```

---

## Step 3: Dataset Preparation

The **training split requires additional preprocessing**.

The evaluation dataset preparation follows the same procedure as [DiffusionDrive](https://github.com/hustvl/DiffusionDrive), including:

- NAVSIM / OpenScene datasets
- nuPlan maps
- NAVSIM logs
- Sensor blobs, including camera images and LiDAR point clouds

A typical dataset directory is organized as follows:

```text
/path/to/MOJITO/MOJITO/dataset/
├── maps/
│   ├── sg-one-north
│   ├── us-ma-boston
│   └── ...
├── navsim_logs/
│   ├── trainval/
│   ├── test/
│   └── exp/
├── sensor_blobs/
│   ├── trainval/                  # Camera images and LiDAR point clouds
│   └── test/
├── navhard_two_stage/             # NAVSIM-v2 navhard split
├── private_test_hard_two_stage/
├── warmup_two_stage/
└── dataset/
```

---

## Step 4: Model Weights

The official MOJITO checkpoints are available on:

🤗 **[Hugging Face: mumucc1/MOJITO](https://huggingface.co/mumucc1/MOJITO)**

### Download from Hugging Face

Install the Hugging Face Hub command-line tool if needed:

```bash
pip install -U huggingface_hub
```

Download the released checkpoints:

```bash
cd /path/to/MOJITO

mkdir -p weights/checkpoints

hf download mumucc1/MOJITO \
    --local-dir weights/checkpoints
```

You can also download the checkpoint manually from the [Hugging Face repository](https://huggingface.co/mumucc1/MOJITO/tree/main).

After downloading, ensure that the default evaluation checkpoint is located at:

```text
weights/checkpoints/mojito_navsim.ckpt
```

The pre-trained backbone weights should be placed under:

```text
weights/pretrained/
```

The expected directory structure is:

```text
weights/
├── pretrained/
│   └── ...
└── checkpoints/
    └── mojito_navsim.ckpt
```

---

## Step 5: Training

To train MOJITO:

```bash
cd /path/to/MOJITO
source setup_env.sh

bash MOJITO/scripts/training/run_diffusiondrive_training.sh
```

The training pipeline automatically uses the preprocessing cache specified by:

```text
MOJITO_CACHE_PATH
```

If the preprocessing cache does not exist, build it first:

```bash
cd /path/to/MOJITO
source setup_env.sh

python MOJITO/navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent \
    experiment_name=training_mojito_agent \
    train_test_split=navtrain
```

---

## Step 6: Evaluation (NAVSIM navtest PDMS)

### Metric Cache Preparation

If the metric cache does not exist, build it following the same procedure as DiffusionDrive:

```bash
cd /path/to/MOJITO
source setup_env.sh

python MOJITO/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path="${NAVSIM_EXP_ROOT}/metric_cache"
```

### Run Evaluation

```bash
cd /path/to/MOJITO
source setup_env.sh

bash MOJITO/scripts/evaluation/run_diffusiondrive.sh
```

The default evaluation checkpoint is:

```text
weights/checkpoints/mojito_navsim.ckpt
```

Make sure the checkpoint has been downloaded from [Hugging Face](https://huggingface.co/mumucc1/MOJITO) before running the evaluation.

---

# Model Release

- [x] MOJITO source code
- [x] Environment setup and training scripts
- [x] Official MOJITO evaluation checkpoints on [Hugging Face](https://huggingface.co/mumucc1/MOJITO)

---

# Acknowledgements

MOJITO is built upon and benefits from the following excellent open-source projects:

- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) (**CVPR 2025 Highlight**)
- [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) (**ICLR 2025**)
- [NAVSIM](https://github.com/autonomousvision/navsim)
- [nuPlan Devkit](https://github.com/motional/nuplan-devkit)

We sincerely thank the authors for their valuable contributions to the autonomous driving community.

---

# Citation

If you find MOJITO useful for your research, please consider citing:

```bibtex
@misc{cheng2026mojitomodaljointlearning,
    title        = {MOJITO: Modal Joint Learning for Unified End-to-End Autonomous Driving},
    author       = {Zhijing Cheng and Xuancheng Zhang and Donglin Di and Lei Fan and Baorui Ma and Hao Li and Xun Yang},
    year         = {2026},
    eprint       = {2607.23511},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV},
    url          = {https://arxiv.org/abs/2607.23511}
}
```
