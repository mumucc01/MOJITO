# MOJITO

**MOJITO** 的官方 PyTorch 实现。 **（ECCV 2026）**

MOJITO 基于 [DiffusionDrive](https://github.com/hustvl/DiffusionDrive)，引入分层三模态融合（图像 / LiDAR / 轨迹）实现端到端自动驾驶。

## 项目结构

```
MOJITO/                   # 仓库根目录
├── MOJITO/               # 主代码：NAVSIM agent、训练与评测
├── Diffusion-Planner/   
├── nuplan-devkit/        
├── weights/              # 预训练权重与 checkpoint 目录
├── setup_env.sh          # 环境变量配置
└── README.md
```


## 本地运行指南

### 第一步：环境配置

详见 **[docs/environment.md](docs/environment.md)**（含完整依赖列表与常见问题）。



从零安装：

```bash
conda create -n mojito python=3.9 -y
conda activate mojito
cd /path/to/MOJITO
bash scripts/install_env.sh
```


### 第二步：加载环境变量

```bash
cd /path/to/MOJITO
source setup_env.sh
```

`setup_env.sh` 设置本机的数据集、实验输出目录与训练缓存路径。如需覆盖，在 `source` 之前设置：

```bash
export OPENSCENE_DATA_ROOT=/your/dataset
export NAVSIM_EXP_ROOT=/your/exp
export MOJITO_CACHE_PATH=/your/training_cache
source setup_env.sh
```

### 第三步：数据集
Train数据集需重新处理。
Eval数据集准备方式与 **[DiffusionDrive](https://github.com/hustvl/DiffusionDrive)** 相同（NAVSIM / OpenScene 下载、maps、navsim_logs、sensor_blobs）。本机示例路径：

```
/path/to/MOJITO/MOJITO/dataset/
├── maps/                          # nuPlan 地图 (sg-one-north, us-ma-boston, ...)
├── navsim_logs/
│   ├── trainval/                  
│   ├── test/                     
│   └── exp/
├── sensor_blobs/
│   ├── trainval/                  # 图像与 LiDAR 数据 
│   └── test/
├── navhard_two_stage/             # navhard 划分 
├── private_test_hard_two_stage/
├── warmup_two_stage/
└── dataset/                      
```

### 第四步：权重

预训练 backbone 位于 `weights/pretrained/`。评测用的 MOJITO 训练 checkpoint：


### 第五步：训练

```bash
cd /path/to/MOJITO
source setup_env.sh
bash MOJITO/scripts/training/run_diffusiondrive_training.sh
```

默认使用 `MOJITO_CACHE_PATH` 下的预处理缓存

若尚无缓存，可先构建：

```bash
source setup_env.sh
python MOJITO/navsim/planning/script/run_dataset_caching.py \ 
    agent=diffusiondrive_agent \
    experiment_name=training_mojito_agent \
    train_test_split=navtrain
```

### 第六步：评测（navtest PDMS）

若需要，先构建 metric cache(同DiffusionDrive)：

```bash
source setup_env.sh
python MOJITO/navsim/planning/script/run_metric_caching.py \
    train_test_split=navtest \
    cache.cache_path="${NAVSIM_EXP_ROOT}/metric_cache"
```

运行评测：

```bash
cd /path/to/MOJITO
source setup_env.sh
bash MOJITO/scripts/evaluation/run_diffusiondrive.sh
```

默认加载 `weights/checkpoints/mojito_navsim.ckpt`。


## 开源说明

- `weights/` 我们将在近期开源所有模型权重。

## 致谢

- [DiffusionDrive](https://github.com/hustvl/DiffusionDrive) (CVPR 2025 Highlight)
- [Diffusion-Planner](https://github.com/ZhengYinan-AIR/Diffusion-Planner) (ICLR 2025)
- [NAVSIM](https://github.com/autonomousvision/navsim)
- [nuplan-devkit](https://github.com/motional/nuplan-devkit)

## 引用

```bibtex
@inproceedings{mojito2026,
  title={MOJITO},
  author={},
  booktitle={ECCV},
  year={2026}
}

@inproceedings{diffusiondrive,
  title={DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving},
  author={Bencheng Liao and others},
  booktitle={CVPR},
  year={2025}
}
```
