#三分支
import torch
import torch.nn as nn
from typing import Optional, Tuple

from .attention import attention, flash_attention
#from .attention import attention
from torch.nn.init import trunc_normal_
from torch.nn.init import trunc_normal_, constant_, xavier_normal_
import matplotlib
import os

import matplotlib.pyplot as plt


class WanRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class TrimodalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: Tuple[int, int] = (-1, -1),
        qk_norm: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.save_counter = 0
        # 只保留两路 pos
        #self.pos1 = nn.Parameter(torch.zeros(1, 261, dim))
        self.pos1 = nn.Parameter(torch.zeros(1, 1024, dim))
        self.pos2 = nn.Parameter(torch.zeros(1, 512, dim))
        self.pos3 = nn.Parameter(torch.zeros(1, 8, dim))

        trunc_normal_(self.pos1, std=0.02)
        trunc_normal_(self.pos2, std=0.02)
        trunc_normal_(self.pos3, std=0.02)

        # 两路 QKV
        self.q1, self.k1, self.v1 = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.q2, self.k2, self.v2 = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.q3, self.k3, self.v3 = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

        # 两路 output projection
        self.o1 = nn.Linear(dim, dim)
        self.o2 = nn.Linear(dim, dim)
        self.o3 = nn.Linear(dim, dim)

        if qk_norm:
            self.norm_q1, self.norm_k1 = WanRMSNorm(dim, eps=eps), WanRMSNorm(dim, eps=eps)
            self.norm_q2, self.norm_k2 = WanRMSNorm(dim, eps=eps), WanRMSNorm(dim, eps=eps)
            self.norm_q3, self.norm_k3 = WanRMSNorm(dim, eps=eps), WanRMSNorm(dim, eps=eps)
        else:
            self.norm_q1 = self.norm_k1 = nn.Identity()
            self.norm_q2 = self.norm_k2 = nn.Identity()
            self.norm_q3 = self.norm_k3 = nn.Identity()
        
        self.input_norm1 = nn.LayerNorm(dim, eps=eps) 
        self.input_norm2 = nn.LayerNorm(dim, eps=eps)
        self.input_norm3 = nn.LayerNorm(dim, eps=eps)  
        
        self._init_weights()

    def _init_weights(self):
        for m in [self.q1, self.k1, self.v1, self.q2, self.k2, self.v2, self.q3, self.k3, self.v3]:
            xavier_normal_(m.weight)
            if m.bias is not None:
                constant_(m.bias, 0)

        # zero-init 输出层：保证残差更稳
        for o in [self.o1, self.o2,self.o3]:
            constant_(o.weight, 0)
            if o.bias is not None:
                constant_(o.bias, 0)

    def _qkv(self, x, q_proj, k_proj, v_proj, norm_q, norm_k):
        b, l, _ = x.shape
        h, d = self.num_heads, self.head_dim
        q = norm_q(q_proj(x)).view(b, l, h, d)
        k = norm_k(k_proj(x)).view(b, l, h, d)
        v = v_proj(x).view(b, l, h, d)
        return q, k, v

    @staticmethod
    def _ensure_lens(x, lens):
        b, l = x.shape[0], x.shape[1]
        if lens is None:
            return torch.full((b,), l, device=x.device, dtype=torch.long)
        return lens.to(device=x.device, dtype=torch.long)

    def _save_attention_map(self, q, k, n1, n2):
        """
        计算并保存 Attention Map 图片
        """
        try:
            # 取 Batch 中的第一个样本进行可视化
            # q, k shape: [B, N, H, D] -> 取 index 0 -> [N, H, D] -> permute -> [H, N, D]
            q_vis = q[0].permute(1, 0, 2)
            k_vis = k[0].permute(1, 0, 2)
            
            # 计算 Attention Score: (Q @ K^T) / sqrt(d)
            scale = self.head_dim ** -0.5
            # [H, N, N]
            attn_scores = torch.matmul(q_vis, k_vis.transpose(-2, -1)) * scale
            attn_probs = attn_scores.softmax(dim=-1)
            
            # 对多头取平均，得到 [N, N]
            attn_map = attn_probs.mean(dim=0).detach().cpu().numpy()
            
            # --- 绘图 ---
            plt.figure(figsize=(10, 10))
            plt.imshow(attn_map, cmap='viridis', interpolation='nearest')
            plt.colorbar()
            
            # 画红线分割两个模态 (x1 和 x2 的边界)
            # 垂直线
            plt.axvline(x=n1 - 0.5, color='red', linestyle='--', linewidth=1.5)
            # 水平线
            plt.axhline(y=n1 - 0.5, color='red', linestyle='--', linewidth=1.5)
            
            plt.title(f"Attention Map (Avg Heads)\nN1={n1}, N2={n2}")
            plt.xlabel("Key Token Index")
            plt.ylabel("Query Token Index")
            
            # 保存文件
            filename = f"attn_step_{self.save_counter}.png"
            filepath = os.path.join("/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/attn_map", filename)
            plt.savefig(filepath)
            plt.close() # 关闭图像释放内存
            
            # print(f"Saved: {filepath}") # 调试用
            
        except Exception as e:
            print(f"Error saving attention map: {e}")

    def _save_traj_to_img_attention(self, q, k, n1, n2):
        """
        可视化 8 个轨迹点 (x2) 对 图像空间 (x1) 的注意力分布
        """
        try:
            # 1. 准备数据 (取 Batch 0)
            # q, k shape: [B, N_total, H, D] -> [H, N_total, D]
            q_vis = q[0].permute(1, 0, 2)
            k_vis = k[0].permute(1, 0, 2)
            
            # 2. 计算完整的 Attention Matrix [H, N_total, N_total]
            # 必须计算完整的 softmax，因为轨迹点不仅关注图像，也关注其他轨迹点
            scale = self.head_dim ** -0.5
            attn_scores = torch.matmul(q_vis, k_vis.transpose(-2, -1)) * scale
            attn_probs = attn_scores.softmax(dim=-1)
            
            # 3. 对 Head 取平均 -> [N_total, N_total]
            attn_map_avg = attn_probs.mean(dim=0)
            
            # 4. 切片提取感兴趣区域
            # Rows: 轨迹点 (从 n1 到 n1+n2)
            # Cols: 图像点 (从 0 到 n1)
            # Shape: [8, 1024]
            traj_attn = attn_map_avg[n1:n1+n2, :n1] 
            
            # 5. Reshape 回图像空间 [8, 16, 64]
            # 假设 n1=1024 是 16x64
            h_img, w_img = 16, 64
            if n1 != h_img * w_img:
                # 如果尺寸不对，打印警告并尝试自动推导方形或保持 1D
                print(f"Warning: n1 ({n1}) != 16*64. Visualization might be wrong.")
                # 简单的 fallback，防止报错
                vis_data = traj_attn.detach().cpu().numpy() 
                is_2d = False
            else:
                vis_data = traj_attn.reshape(n2, h_img, w_img).detach().cpu().numpy()
                is_2d = True

            # 6. 绘图 (8个子图)
            # 创建 4行2列 的布局，适应 8 个点
            fig, axes = plt.subplots(4, 2, figsize=(16, 12))
            fig.suptitle(f'Trajectory Points Attention to Image (Step {self.save_counter})', fontsize=16)
            
            # 统一色阶范围，方便对比不同点的关注强度
            vmin, vmax = vis_data.min(), vis_data.max()

            for i in range(n2):
                ax = axes.flat[i]
                if is_2d:
                    # 绘制热力图，使用 'jet' 或 'viridis' 颜色
                    im = ax.imshow(vis_data[i], cmap='jet', vmin=vmin, vmax=vmax, aspect='auto')
                    ax.set_title(f'Traj Point {i}')
                    ax.axis('off') # 隐藏坐标轴刻度
                else:
                    ax.plot(vis_data[i])
                    ax.set_title(f'Traj Point {i} (1D)')

            # 添加颜色条
            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) # [left, bottom, width, height]
            fig.colorbar(im, cax=cbar_ax)
            
            # 保存
            filename = f"traj_attn_step_{self.save_counter}.png"
            filepath = os.path.join('/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/attn_map', filename)
            plt.savefig(filepath, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error saving trajectory attention: {e}")
            import traceback
            traceback.print_exc()

    def forward(
        self,
        x1: torch.Tensor,                 # [B, N1, C]
        x2: torch.Tensor, 
        x3: torch.Tensor,          # [B, N2, C]
        seq_lens_total: Optional[torch.Tensor] = None,
        seq_lens1: Optional[torch.Tensor] = None,
        seq_lens2: Optional[torch.Tensor] = None,
        seq_lens3: Optional[torch.Tensor] = None,
    ):
        b, n1, c = x1.shape
        n2 = x2.shape[1]
        n3 = x3.shape[1]

        x1_res = x1
        x2_res = x2
        x3_res = x3
        # pos（建议按实际 token 数切片，避免 n1/n2 不是固定值时报错）
        x1 = x1 + self.pos1[:, :n1, :]
        x2 = x2 + self.pos2[:, :n2, :]
        x3 = x3 + self.pos3[:, :n3, :]

        x1 = self.input_norm1(x1)
        x2 = self.input_norm2(x2)
        x3 = self.input_norm3(x3)

        q1, k1, v1 = self._qkv(x1, self.q1, self.k1, self.v1, self.norm_q1, self.norm_k1)
        q2, k2, v2 = self._qkv(x2, self.q2, self.k2, self.v2, self.norm_q2, self.norm_k2)
        q3, k3, v3 = self._qkv(x3, self.q3, self.k3, self.v3, self.norm_q3, self.norm_k3)

        q_cat = torch.cat([q1, q2, q3], dim=1)
        k_cat = torch.cat([k1, k2, k3], dim=1)
        v_cat = torch.cat([v1, v2, v3], dim=1)
        
        #self._save_attention_map(q_cat, k_cat, n1, n2)
        #self._save_traj_to_lidar_attention(q_cat, k_cat, n1, n2)

        #self.save_counter = self.save_counter + 1

        if seq_lens_total is None:
            l1 = self._ensure_lens(x1, seq_lens1)
            l2 = self._ensure_lens(x2, seq_lens2)
            l3 = self._ensure_lens(x3, seq_lens3)
            k_lens = l1 + l2 + l3
        else:
            k_lens = seq_lens_total.to(device=x1.device, dtype=torch.long)

        attn_out = attention(
        #attn_out =  flash_attention(
            q=q_cat, #[B,1544,8,48]
            k=k_cat,  #[B,1544,8,48]
            v=v_cat,   #[B,1544,8,48]
            k_lens=k_lens,
            window_size=self.window_size,
        )  # [B, N1+N2, H, D]

        out1_h = attn_out[:, :n1, :, :].float()
        out2_h = attn_out[:, n1:n1+n2, :, :].float()
        out3_h = attn_out[:, n1+n2:n1+n2+n3, :, :].float()
        
        out1 = self.o1(out1_h.flatten(2))  # [B, N1, C]
        out2 = self.o2(out2_h.flatten(2))  # [B, N2, C]
        out3 = self.o3(out3_h.flatten(2)) 
        
        out1 = out1 + x1_res
        out2 = out2 + x2_res
        out3 = out3 + x3_res

        return out1, out2, out3
