# MOJITO 环境配置

本文档说明如何复现 MOJITO 训练/评估环境。你当前可用的环境：

```
/path/to/conda_env/mojito
```

## 快速使用（已有环境）

```bash
conda activate /path/to/conda_env/mojito
cd /path/to/MOJITO
source setup_env.sh
```

## 从零安装

### 1. 创建 Conda 环境

```bash
conda create -n mojito python=3.9 pip=23.3 -y
conda activate mojito
```

或使用项目内 `MOJITO/environment.yml`（基于 NAVSIM）：

```bash
cd MOJITO/MOJITO
conda env create -f environment.yml -n mojito
conda activate mojito
```

### 2. 安装 PyTorch（CUDA 12.x 示例）

你的工作环境为 **Python 3.9 + PyTorch 2.5.1 + CUDA 12.4**。新环境建议对齐：

```bash
pip install torch==2.5.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu124
```

若需与 `MOJITO/requirements.txt` 完全一致（官方 NAVSIM 配置）：

```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

### 3. 安装 MOJITO 核心依赖

```bash
cd /path/to/MOJITO/MOJITO
pip install -r requirements.txt
pip install -r ../requirements-mojito.txt
```

或一键脚本：

```bash
cd /path/to/MOJITO
bash scripts/install_env.sh
```

### 4. 编译 PointNet++（MOJITO LiDAR 特征必需）

```bash
cd MOJITO/navsim/agents/diffusiondrive/Uni3D/Pointnet2_PyTorch
pip install -e .
```

### 5. 可选：Flash Attention

Flash Attention 可加速 attention，**非必须**（失败时会自动回退到 PyTorch SDPA）。

```bash
# FA2（推荐，你的环境里已安装 2.6.3）
pip install flash-attn==2.6.3 --no-build-isolation

# FA3 可能在部分机器上因 libstdc++ 版本报错，可跳过
```

无显示器服务器请使用 headless OpenCV：

```bash
pip install opencv-python-headless
```

## 依赖说明

| 类别 | 包 | 用途 |
|------|-----|------|
| 深度学习 | `torch`, `torchvision`, `pytorch-lightning` | 训练框架 |
| 扩散模型 | `diffusers`, `einops` | DiffusionDrive / MOJITO decoder |
| 视觉 backbone | `timm` | ResNet / EVA 等 |
| 配置 | `hydra-core` | 训练/评估脚本 |
| 点云 | `pointnet2-ops`, `spconv-cu120` | Uni3D / PTv3 |
| 3D / 地图 | `open3d`, `geopandas`, `shapely` | nuPlan 地图与点云 |
| 仿真 | `ray`, `casadi`, `control` | NAVSIM / PDM 评估 |
| 图像 | `opencv-python-headless`, `Pillow` | 数据读取（无 GUI 用 headless） |
| 其他 | `mmcv`, `mmengine`, `safetensors` | 部分模块与权重加载 |


## 子项目说明

MOJITO 仓库内已 **vendored** 以下代码，**无需单独 pip 安装**：

- `nuplan-devkit/` — 通过 `PYTHONPATH` 引用
- `Diffusion-Planner/` — 通过 `PYTHONPATH` 引用
- `MOJITO/navsim/` — 主代码

安装后执行：

```bash
source setup_env.sh   # 自动设置 PYTHONPATH 与数据路径
```

## 验证安装

```bash
conda activate mojito   # 或你的 diffusion_drive 环境
cd /path/to/MOJITO
source setup_env.sh

python -c "
import torch, cv2, diffusers, timm, hydra
import navsim, nuplan, diffusion_planner
from navsim.agents.diffusiondrive.transfuser_agent import TransfuserAgent
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
print('navsim', navsim.__file__)
print('OK')
"
```

## 常见问题

### `libGLX.so.0` / OpenCV 报错

```bash
pip uninstall opencv-python -y
pip install opencv-python-headless
```

### `No module named 'dinov3'`

确保已 `source setup_env.sh`，`PYTHONPATH` 包含 `MOJITO/MOJITO/navsim/agents/diffusiondrive/dinov3`。

### `flash_attn` / `GLIBCXX_3.4.32` 报错

可忽略（代码已自动回退 SDPA），或只装 `flash-attn==2.6.3` 不装 `flash-attn-3`。

### torchvision 与 torch 版本警告

你当前环境为 torch 2.5.1 + torchvision 0.15.2（为 torch 2.0 构建），会有警告但一般可训练。若要消除警告：

```bash
pip install torchvision --upgrade --index-url https://download.pytorch.org/whl/cu124
```

### 导入旧 `Diffusion_Drive` 路径

不要混用旧工程 `PYTHONPATH`。始终先 `source setup_env.sh`，并建议新开终端。

## 参考

- [DiffusionDrive 安装](MOJITO/docs/install.md)
- [NAVSIM Getting Started](https://github.com/autonomousvision/navsim)
