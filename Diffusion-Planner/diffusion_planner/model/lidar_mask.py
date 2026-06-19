
from DiffusionDrive.navsim.agents.transfuser.transfuser_config_raster import TransfuserConfig_Raster
import torch.nn as nn
import torch
class Lidar_Mask(nn.Module):
    def __init__(self):
        super().__init__()
        self._config = TransfuserConfig_Raster()

    def get_feasible_voxel_feature(self, lidar_points: torch.Tensor) -> torch.Tensor:
        """
        输入: lidar_points [3, N] (x, y, z)
        输出: feature [ H, W] (float32, 0/1)
            1 = 可行区域
            0 = 不可行区域 (有 z >= 0.2 的点)
        规则:
            - 某格子内:
                - 没有任何点           → 可行
                - 有点 & 所有点 z<0.2 → 可行
                - 有点 & 存在 z>=0.2  → 不可行
        """
        assert lidar_points.dim() == 2 and lidar_points.shape[0] == 3, \
            f"lidar_points shape must be [3, N], got {lidar_points.shape}"

        device = lidar_points.device

        # -----------------------
        # -----------------------
        ppm = float(self._config.pixels_per_meter)  # e.g. 4.0
        x_min = float(self._config.lidar_min_x)     # e.g. -32
        x_max = float(self._config.lidar_max_x)
        y_min = float(self._config.lidar_min_y)     # e.g. -32
        y_max = float(self._config.lidar_max_y)

        z_threshold = 0.2

        W = int((x_max - x_min) * ppm)
        H = int((y_max - y_min) * ppm)

        # -----------------------
        # -----------------------
        xs = lidar_points[0]
        ys = lidar_points[1]
        zs = lidar_points[2]

        mask_in_region = (
            (xs >= x_min) & (xs < x_max) &
            (ys >= y_min) & (ys < y_max)
        )

        xs = xs[mask_in_region]
        ys = ys[mask_in_region]
        zs = zs[mask_in_region]

        if xs.numel() == 0:
            return torch.ones((1, H, W), dtype=torch.float32, device=device)

        # -----------------------
        # -----------------------
        ix = ((xs - x_min) * ppm).long()
        iy = ((ys - y_min) * ppm).long()

        ix = torch.clamp(ix, 0, W - 1)
        iy = torch.clamp(iy, 0, H - 1)

        flat_idx = iy * W + ix  # [N']

        # -----------------------
        # -----------------------
        mask_block_points = zs >= z_threshold

        blocked = torch.zeros(H * W, dtype=torch.bool, device=device)

        if mask_block_points.any():
            flat_idx_block = flat_idx[mask_block_points]
            blocked[flat_idx_block] = True

        # -----------------------
        # -----------------------
        feasible = ~blocked   # [H*W] bool

        feature = feasible.view(H, W).float()  # [H, W]

        return feature