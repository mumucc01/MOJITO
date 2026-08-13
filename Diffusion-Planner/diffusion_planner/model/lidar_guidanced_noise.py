import torch.nn as nn
import torch

class lidar_guidance_noise_sampling(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.y_min = float(config.lidar_min_y) 
        self.x_min = float(config.lidar_min_x) 
        self.ppm = float(config.pixels_per_meter)
        
    def v1(self, mask, B, P, T):
        """
        ppm: pixels per meter
        xy_meter = out * std_xy + mean_xy
        """
        x_min = self.x_min
        y_min = self.y_min
        ppm = self.ppm
        device = mask.device
        mask = mask.to(torch.int)
        H, W = mask.shape

        mean_xy = torch.tensor([10.0, 0.0], device=device)   # [2]
        std_xy  = torch.tensor([20.0, 20.0], device=device)  # [2]

        feas_y, feas_x = torch.nonzero(mask == 1, as_tuple=True)  # [N]
        if feas_y.numel() == 0:
            raise ValueError("mask 中没有任何可行格子 (mask==1)")

        # ######## 
        feas_x_meter = x_min + (feas_x.float() + 0.5) / ppm   # [N]
        feas_y_meter = y_min + (feas_y.float() + 0.5) / ppm   # [N]
        feas_centers = torch.stack([feas_x_meter, feas_y_meter], dim=-1)  # [N,2]

        feas_origin_dist2 = (feas_centers ** 2).sum(dim=-1)   # [N]

        out = torch.randn(B, P, T, 2, device=device) 
        mean_xy_b = mean_xy.view(1, 1, 1, 2)
        std_xy_b  = std_xy.view(1, 1, 1, 2)

        xy_meter = out * std_xy_b + mean_xy_b
    
        x = xy_meter[..., 0]   # [B,P,T]
        y = xy_meter[..., 1]   # [B,P,T]

        ix = ((x - x_min) * ppm).long()
        iy = ((y - y_min) * ppm).long()

        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        valid = torch.zeros(B, P, T, dtype=torch.bool, device=device)

        if in_bounds.any():
            iy_ib = iy[in_bounds]
            ix_ib = ix[in_bounds]
            mask_vals = (mask[iy_ib, ix_ib] == 1)
            valid[in_bounds] = mask_vals

        invalid = ~valid  # [B,P,T]

        if invalid.any():
            xy_invalid = xy_meter[invalid]

            # dist2[i,j] = || xy_invalid[i] - feas_centers[j] ||^2
            diff = xy_invalid.unsqueeze(1) - feas_centers.unsqueeze(0)  # [M,1,2] - [1,N,2]
            dist2 = (diff ** 2).sum(dim=-1)                            # [M,N]

            min_dist2, _ = dist2.min(dim=1)  # [M]

            eps = 1e-8
            same_min = (dist2 - min_dist2.unsqueeze(1)).abs() < eps    # [M,N] bool

            big = 1e12
            score = torch.where(
                same_min,
                feas_origin_dist2.unsqueeze(0).expand_as(dist2),       # [M,N]
                torch.full_like(dist2, big)
            )
            best_idx = score.argmin(dim=1)   # [M]

            nearest_centers = feas_centers[best_idx]  # [M,2] meter

            nearest_noise = (nearest_centers - mean_xy) / std_xy       # [M,2]
        
            out[invalid] = nearest_noise
        return out


    def v2(self, mask, B, P, T):
        """
        ppm: pixels per meter
        xy_meter = out * std_xy + mean_xy
        """
        x_min = self.x_min
        y_min = self.y_min
        ppm = self.ppm
        device = mask.device
        mask = mask.to(torch.int)
        H, W = mask.shape

        mean_xy = torch.tensor([10.0, 0.0], device=device)   # [2]
        std_xy  = torch.tensor([20.0, 20.0], device=device)  # [2]

        feas_y, feas_x = torch.nonzero(mask == 1, as_tuple=True)  # [N]
        if feas_y.numel() == 0:
            raise ValueError("mask 中没有任何可行格子 (mask==1)")

        # ######## 
        feas_x_meter = x_min + (feas_x.float() + 0.5) / ppm   # [N]
        feas_y_meter = y_min + (feas_y.float() + 0.5) / ppm   # [N]
        feas_centers = torch.stack([feas_x_meter, feas_y_meter], dim=-1)  # [N,2]

        feas_origin_dist2 = (feas_centers ** 2).sum(dim=-1)   # [N]

        out = torch.randn(B, P, T, 2, device=device) 
        mean_xy_b = mean_xy.view(1, 1, 1, 2)
        std_xy_b  = std_xy.view(1, 1, 1, 2)

        xy_meter = out * std_xy_b + mean_xy_b
    
        x = xy_meter[..., 0]   # [B,P,T]
        y = xy_meter[..., 1]   # [B,P,T]

        ix = ((x - x_min) * ppm).long()
        iy = ((y - y_min) * ppm).long()

        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        valid = torch.zeros(B, P, T, dtype=torch.bool, device=device)

        if in_bounds.any():
            iy_ib = iy[in_bounds]
            ix_ib = ix[in_bounds]
            mask_vals = (mask[iy_ib, ix_ib] == 1)
            valid[in_bounds] = mask_vals

        invalid = ~valid  # [B,P,T]

        if invalid.any():
            xy_invalid = xy_meter[invalid]

            # dist2[i,j] = || xy_invalid[i] - feas_centers[j] ||^2
            diff = xy_invalid.unsqueeze(1) - feas_centers.unsqueeze(0)  # [M,1,2] - [1,N,2]
            dist2 = (diff ** 2).sum(dim=-1)                            # [M,N]

            min_dist2, _ = dist2.min(dim=1)  # [M]

            eps = 1e-8
            same_min = (dist2 - min_dist2.unsqueeze(1)).abs() < eps    # [M,N] bool

            big = 1e12
            score = torch.where(
                same_min,
                feas_origin_dist2.unsqueeze(0).expand_as(dist2),       # [M,N]
                torch.full_like(dist2, big)
            )
            best_idx = score.argmin(dim=1)   # [M]

            nearest_centers = feas_centers[best_idx]  # [M,2] meter

            nearest_noise = (nearest_centers - mean_xy) / std_xy       # [M,2]
        
            out[invalid] = nearest_noise
        return out
