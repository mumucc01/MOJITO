from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.transfuser_backbone import TransfuserBackbone
from navsim.agents.diffusiondrive.transfuser_features import BoundingBox2DIndex
from navsim.common.enums import StateSE2Index
from diffusers.schedulers import DDIMScheduler
from navsim.agents.diffusiondrive.modules.conditional_unet1d import ConditionalUnet1D,SinusoidalPosEmb
import torch.nn.functional as F
from navsim.agents.diffusiondrive.modules.blocks import linear_relu_ln,bias_init_with_prob, gen_sineembed_for_position, GridSampleCrossBEVAttention
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from torch.nn import TransformerDecoder,TransformerDecoderLayer
from typing import Any, List, Dict, Optional, Union
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt



def visualize_noise_grid_on_lidar(
    lidar_feature: torch.Tensor,
    noise: torch.Tensor,
    x_min: float,
    y_min: float,
    ppm: float,
    save_path: str,
):
    """
    lidar_feature: [1,H,W] or [H,W], 0/1 feasible-area mask
    noise: [B, P, T, 2]
    x_min, y_min, ppm: must match Lidar_CZJ.get_feasible_voxel_feature config
    save_path: output PNG path
    """
    if lidar_feature.dim() == 3:
        lidar_feature = lidar_feature.squeeze(0)
    assert lidar_feature.dim() == 2, f"lidar_feature should be [H,W], got {lidar_feature.shape}"

    device = lidar_feature.device
    H, W = lidar_feature.shape

   

    pts_meter = noise
    x = pts_meter[:, 0]
    y = pts_meter[:, 1]

    ix = ((x - x_min) * ppm).long()
    iy = ((y - y_min) * ppm).long()

    in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix = ix[in_bounds]
    iy = iy[in_bounds]

    bg = lidar_feature.detach().cpu().numpy().astype(np.float32)   # [H,W]
    bg_img = (bg * 255).astype(np.uint8)                           # [H,W]

    rgb = np.stack([bg_img, bg_img, bg_img], axis=-1)              # [H,W,3]

    ix_np = ix.cpu().numpy()
    iy_np = iy.cpu().numpy()
    rgb[iy_np, ix_np, 0] = 255  # R
    rgb[iy_np, ix_np, 1] = 0    # G
    rgb[iy_np, ix_np, 2] = 0    # B

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin='lower')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved noise visualization to: {save_path}")




def save_as_8bit(array, output_path):
    """Save array as 8-bit PNG."""
    if array.max() > array.min():
        normalized = (array - array.min()) / (array.max() - array.min())
        normalized = (normalized * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(array, dtype=np.uint8)
    
    img = Image.fromarray(normalized)
    img.save(output_path)
    print(f"8-bit PNG saved: {output_path}")
