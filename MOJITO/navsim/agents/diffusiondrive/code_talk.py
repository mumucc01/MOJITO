


# tap
#                    [    C        |      L       |       T        ]

# ┌───┬───┬───┐
# │ 1 │ 0 │ 0 │  C attend C
# ├───┼───┼───┤
# │ 0 │ 1 │ 0 │  L attend L  
# ├───┼───┼───┤
# │ 0 │ 0 │ 1 │  T attend T
# └───┴───┴───┘

# ┌───┬───┬───┐
# │ 1 │ 1 │ 0 │  C attend C, L
# ├───┼───┼───┤
# │ 1 │ 1 │ 0 │  L attend C, L
# ├───┼───┼───┤
# │ 1 │ 1 │ 1 │  T attend C, L, T
# └───┴───┴───┘

# ┌───┬───┬───┐
# │ 1 │ 1 │ 1 │
# ├───┼───┼───┤
# │ 1 │ 1 │ 1 │
# ├───┼───┼───┤
# │ 1 │ 1 │ 1 │
# └───┴───┴───┘

import torch
import torch.nn as nn


def create_stage_mask(num_cam, num_lidar, num_traj, stage, device):
    """
    创建不同阶段的 attention mask
    
    Args:
        num_cam: camera token 数量
        num_lidar: lidar token 数量  
        num_traj: trajectory token 数量
        stage: 1, 2, or 3
        device: torch device
    
    Returns:
        mask: (total_len, total_len), True 表示可以 attend, False 表示被 mask
    """
    total = num_cam + num_lidar + num_traj
    
    c_end = num_cam
    l_end = num_cam + num_lidar
    
    if stage == 1:
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:c_end, :c_end] = True           # C attend C
        mask[c_end:l_end, c_end:l_end] = True  # L attend L
        mask[l_end:, l_end:] = True            # T attend T
        
    elif stage == 2:
        mask = torch.zeros(total, total, dtype=torch.bool, device=device)
        mask[:c_end, :l_end] = True            # C attend C, L
        mask[c_end:l_end, :l_end] = True       # L attend C, L
        mask[l_end:, :] = True                 # T attend all
        
    elif stage == 3:
        mask = torch.ones(total, total, dtype=torch.bool, device=device)
    
    return mask


class HierarchicalFusionBlock(nn.Module):
    """单个 block，根据 stage 使用不同的 mask"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )
    
    def forward(self, x, mask):
        """
        Args:
            x: (B, N, D) 所有 token concat 在一起
            mask: (N, N) attention mask, True = attend, False = mask
        """
        attn_mask = ~mask
        
        # Self-attention with mask
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        return x


class HierarchicalFusionTransformer(nn.Module):
    """
    12层 Transformer，分三个阶段使用不同的 attention pattern
    """
    
    def __init__(self, dim=384, num_heads=6, num_layers=12, mlp_ratio=4.0):
        super().__init__()
        
        self.num_layers = num_layers
        self.blocks = nn.ModuleList([
            HierarchicalFusionBlock(dim, num_heads, mlp_ratio)
            for _ in range(num_layers)
        ])
        
        self.stage_boundaries = [4, 8, 12]
    
    def get_stage(self, layer_idx):
        """根据 layer index (0-based) 返回 stage (1, 2, 3)"""
        if layer_idx < self.stage_boundaries[0]:
            return 1
        elif layer_idx < self.stage_boundaries[1]:
            return 2
        else:
            return 3
    
    def forward(self, cam_tokens, lidar_tokens, traj_tokens):
        """
        Args:
            cam_tokens: (B, N_c, D)
            lidar_tokens: (B, N_l, D)
            traj_tokens: (B, N_t, D)
        
        Returns:
            cam_tokens, lidar_tokens, traj_tokens: 更新后的特征
        """
        B = cam_tokens.shape[0]
        num_cam = cam_tokens.shape[1]
        num_lidar = lidar_tokens.shape[1]
        num_traj = traj_tokens.shape[1]
        device = cam_tokens.device
        
        x = torch.cat([cam_tokens, lidar_tokens, traj_tokens], dim=1)  # (B, N_total, D)
        
        masks = {
            1: create_stage_mask(num_cam, num_lidar, num_traj, 1, device),
            2: create_stage_mask(num_cam, num_lidar, num_traj, 2, device),
            3: create_stage_mask(num_cam, num_lidar, num_traj, 3, device),
        }
        
        for layer_idx, block in enumerate(self.blocks):
            stage = self.get_stage(layer_idx)
            mask = masks[stage]
            x = block(x, mask)
        
        cam_out = x[:, :num_cam]
        lidar_out = x[:, num_cam:num_cam + num_lidar]
        traj_out = x[:, num_cam + num_lidar:]
        
        return cam_out, lidar_out, traj_out


if __name__ == "__main__":
    model = HierarchicalFusionTransformer(dim=384, num_heads=6, num_layers=12)
    
    B, N_c, N_l, N_t, D = 2, 100, 80, 6, 384
    cam = torch.randn(B, N_c, D)
    lidar = torch.randn(B, N_l, D)
    traj = torch.randn(B, N_t, D)
    
    cam_out, lidar_out, traj_out = model(cam, lidar, traj)
    
    print(f"Input shapes:  cam {cam.shape}, lidar {lidar.shape}, traj {traj.shape}")
    print(f"Output shapes: cam {cam_out.shape}, lidar {lidar_out.shape}, traj {traj_out.shape}")

