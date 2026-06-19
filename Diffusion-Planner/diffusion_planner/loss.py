from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn
from torchvision import transforms
from diffusion_planner.utils.normalizer import StateNormalizer


from DiffusionDrive.navsim.agents.transfuser.transfuser_config_raster import TransfuserConfig_Raster

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def normalization(tensor , device='cuda'):
    B, V, C, H, W = tensor.shape
    
    x = tensor.float().to(device)
    
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,1,C,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,1,C,1,1)
    x = x / 255.0
    x = (x - mean) / std
    return x

import torch

def gaussian_noise_constrained(mask, noise_init, B, P, T, x_min, y_min, ppm):
    """
    mask: [B, H, W] 或 [H, W]，1 = 可行, 0 = 不可行
    noise_init: [B, P, T, 2]，原始高斯噪声（标准正态）
    x_min, y_min: BEV 网格左下角在 meter 坐标中的值
    ppm: pixels per meter

    返回:
        out: [B, P, T, 2] 噪声 (与原版保持一致：在“噪声空间”)
             xy_meter = out * std_xy + mean_xy
    """

    device = noise_init.device

    if mask.dim() == 2:
        H, W = mask.shape
        mask = mask.to(torch.int).unsqueeze(0).expand(B, H, W).contiguous()  # [B,H,W]
    elif mask.dim() == 3:
        mask = mask.to(torch.int)
        B_m, H, W = mask.shape
        assert B_m == B, f"mask batch size {B_m} 与 B={B} 不一致"
    else:
        raise ValueError(f"mask 维度应为 2 或 3, 当前为 {mask.shape}")

    mean_xy = torch.tensor([10.0, 0.0], device=device)   # [2]
    std_xy  = torch.tensor([20.0, 20.0], device=device)  # [2]

    out = noise_init.clone()  # [B,P,T,2]

    mean_xy_b = mean_xy.view(1, 1, 1, 2)
    std_xy_b  = std_xy.view(1, 1, 1, 2)

    xy_meter = noise_init * std_xy_b + mean_xy_b

    x = xy_meter[..., 0]   # [B,P,T]
    y = xy_meter[..., 1]   # [B,P,T]

    ix = ((x - x_min) * ppm).long()  # [B,P,T]
    iy = ((y - y_min) * ppm).long()  # [B,P,T]

    in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)  # [B,P,T]
    valid = torch.zeros(B, P, T, dtype=torch.bool, device=device)

    if in_bounds.any():
        b_idx, p_idx, t_idx = torch.nonzero(in_bounds, as_tuple=True)  # [M]
        iy_ib = iy[b_idx, p_idx, t_idx]
        ix_ib = ix[b_idx, p_idx, t_idx]
        mask_vals = (mask[b_idx, iy_ib, ix_ib] == 1)
        valid[b_idx, p_idx, t_idx] = mask_vals

    invalid = ~valid  # [B,P,T]

    for b in range(B):
        invalid_b = invalid[b]  # [P,T]
        if not invalid_b.any():
            continue

        feas_y, feas_x = torch.nonzero(mask[b] == 1, as_tuple=True)  # [N]
        if feas_y.numel() == 0:
            raise ValueError(f"第 {b} 个 batch 的 mask 中没有任何可行格子 (mask==1)")

        feas_x_meter = x_min + (feas_x.float() + 0.5) / ppm   # [N]
        feas_y_meter = y_min + (feas_y.float() + 0.5) / ppm   # [N]
        feas_centers = torch.stack([feas_x_meter, feas_y_meter], dim=-1)  # [N,2]

        feas_origin_dist2 = (feas_centers ** 2).sum(dim=-1)   # [N]

        xy_invalid = xy_meter[b][invalid_b]   # [M_b, 2]

        diff = xy_invalid.unsqueeze(1) - feas_centers.unsqueeze(0)  # [M_b,1,2] - [1,N,2]
        dist2 = (diff ** 2).sum(dim=-1)                            # [M_b,N]

        min_dist2, _ = dist2.min(dim=1)  # [M_b]

        eps = 1e-8
        same_min = (dist2 - min_dist2.unsqueeze(1)).abs() < eps    # [M_b,N] bool

        big = 1e12
        score = torch.where(
            same_min,
            feas_origin_dist2.unsqueeze(0).expand_as(dist2),       # [M_b,N]
            torch.full_like(dist2, big)
        )

        best_idx = score.argmin(dim=1)         # [M_b]
        nearest_centers = feas_centers[best_idx]  # [M_b,2] meter

        nearest_noise = (nearest_centers - mean_xy) / std_xy       # [M_b,2]

        out[b][invalid_b] = nearest_noise

    return out


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
):   
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    B, Pn, T, _ = neighbors_future.shape  #T:80
    ego_current, neighbors_current = inputs["ego_current_state"][:, :2], inputs["neighbor_agents_past"][:, :Pn, -1, :2] #[2,4] [2,0,4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0. #[2048,10]
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1) #[2048,10,81]
    #gt_future [2048,11,80,4]. current_states[2048,11,4]  ego_future[2048,80,4]  neighbors_future[2048,10,80,4]
    gt_future = torch.cat([ego_future[:, None, :, :2], neighbors_future[..., :2]], dim=1) # [B, 1, 80, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, 1, 4]

    P = gt_future.shape[1] # 11
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
   
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0
    #t = torch.tensor([0,1]).to("cuda") 

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    xT = mean + std * z   #[2048,11,81,2]

    
    lidar_mask = inputs['lidar_mask']
    
    y_min = float(TransfuserConfig_Raster().lidar_min_y) 
    x_min = float(TransfuserConfig_Raster().lidar_min_x) 
    ppm = float(TransfuserConfig_Raster().pixels_per_meter)
    xT = gaussian_noise_constrained(lidar_mask, xT, B, P, T, x_min, y_min, ppm)
    
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    

    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT, #[2048,11,81,4] 
        "diffusion_time": t, #2048
    }

    sensor_image = inputs['senor_image'].contiguous()
    lidar_bev = inputs['lidar_bev'].contiguous() #[B,N,H,W,C]
 
    encoder_pred, decoder_output = model(merged_inputs, sensor_image, lidar_bev, None) # [B, P, 1 + T, 4] [2048,11,81,4]
    score = decoder_output["score"][:, :, 1:, :] # [B, P, T, 4]. [2048,11,80,4]
    
    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()



    
    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output