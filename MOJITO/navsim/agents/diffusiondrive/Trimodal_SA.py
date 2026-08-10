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
       
        #self.pos1 = nn.Parameter(torch.zeros(1, 261, dim))
        self.pos1 = nn.Parameter(torch.zeros(1, 1024, dim))
        self.pos2 = nn.Parameter(torch.zeros(1, 512, dim))
        self.pos3 = nn.Parameter(torch.zeros(1, 8, dim))

        trunc_normal_(self.pos1, std=0.02)
        trunc_normal_(self.pos2, std=0.02)
        trunc_normal_(self.pos3, std=0.02)

     
        self.q1, self.k1, self.v1 = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.q2, self.k2, self.v2 = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.q3, self.k3, self.v3 = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)

      
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
