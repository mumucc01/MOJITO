# MOJITO (**ECCV 2026**)

[![arXiv](https://img.shields.io/badge/arXiv-2607.23511-b31b1b.svg)](https://arxiv.org/abs/2607.23511)

MOJITO is an end-to-end autonomous driving framework built upon [DiffusionDrive](https://github.com/hustvl/DiffusionDrive). It introduces hierarchical three-modal fusion across **vision**, **LiDAR**, and **trajectory/action representations**, enabling unified perception and planning for end-to-end autonomous driving.

## Project Structure

```
MOJITO/                   
├── MOJITO/               # Main codebase: NAVSIM agent, training, and evaluation
├── Diffusion-Planner/    # Diffusion-based trajectory planning module
├── nuplan-devkit/        # nuPlan development toolkit
├── weights/              # Pre-trained weights and model checkpoints
├── setup_env.sh          # Environment configuration script
└── README.md
```

---

# Local Setup Guide

## Step 1: Environment Setup

Please refer to **[docs/environment.md](docs/environment.md)** for the complete dependency list, installation instructions, and common issues.

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

If you need to customize these paths, override them before sourcing:

```bash
export OPENSCENE_DATA_ROOT=/your/dataset
export NAVSIM_EXP_ROOT=/your/exp
export MOJITO_CACHE_PATH=/your/training_cache

source setup_env.sh
```

---

# Step 3: Dataset Preparation

The **training split requires additional preprocessing**.

The evaluation dataset preparation follows the same procedure as **[DiffusionDrive](https://github.com/hustvl/DiffusionDrive)**, including:

- NAVSIM / OpenScene datasets
- nuPlan maps
- navsim logs
- sensor blobs (camera images and LiDAR point clouds)

A typical dataset directory is organized as follows:

```
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

# Step 4: Model Weights

The pre-trained backbone weights are located at:

```
weights/pretrained/
```

The official MOJITO training checkpoints used for evaluation will be released soon.

---

# Step 5: Training

To train MOJITO:

```bash
cd /path/to/MOJITO
source setup_env.sh

bash MOJITO/scripts/training/run_diffusiondrive_training.sh
```

The training pipeline automatically uses the preprocessing cache specified by:

```
MOJITO_CACHE_PATH
```

If the cache does not exist, build it first:

```bash
source setup_env.sh

python MOJITO/navsim/planning/script/run_dataset_caching.py \
    agent=diffusiondrive_agent \
    experiment_name=training_mojito_agent \
    train_test_split=navtrain
```

---

# Step 6: Evaluation (NAVSIM navtest PDMS)

## Metric Cache Preparation

If required, build the metric cache following the same procedure as DiffusionDrive:

```bash
source setup_env.sh

python MOJITO/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path="${NAVSIM_EXP_ROOT}/metric_cache"
```

## Run Evaluation

```bash
cd /path/to/MOJITO
source setup_env.sh

bash MOJITO/scripts/evaluation/run_diffusiondrive.sh
```

The default evaluation checkpoint is:

```
weights/checkpoints/mojito_navsim.ckpt
```

---

# Open-Source Roadmap

- We will release all official MOJITO model checkpoints under `weights/` soon.

---

# Acknowledgements

MOJITO is built upon and benefits from the following excellent open-source projects:

- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) (**CVPR 2025 Highlight**)
- [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) (**ICLR 2025**)
- [NAVSIM](https://github.com/autonomousvision/navsim)
- [nuplan-devkit](https://github.com/motional/nuplan-devkit)

We sincerely thank the authors for their valuable contributions to the autonomous driving community.

---

# Citation

If you find MOJITO useful for your research, please consider citing:

```bibtex
@misc{cheng2026mojitomodaljointlearning,
      title={MOJITO: Modal Joint Learning for Unified End-to-End Autonomous Driving}, 
      author={Zhijing Cheng and Xuancheng Zhang and Donglin Di and Lei Fan and Baorui Ma and Hao Li and Xun Yang},
      year={2026},
      eprint={2607.23511},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.23511}, 
}
```
