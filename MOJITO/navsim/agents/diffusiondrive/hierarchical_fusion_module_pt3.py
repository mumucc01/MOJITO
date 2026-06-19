#version v120


import torch
import torch.nn as nn
import sys
from typing import Dict, List, Optional, Tuple
from functools import partial
import numpy as np
import timm
import time
from timm.models.layers import Mlp
from torch.nn.init import trunc_normal_, constant_, xavier_normal_
import torch.distributed as dist
from navsim.agents.diffusiondrive.modules.multimodal_loss import LossComputer
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.agents.diffusiondrive.modules.dinov3_backbone import DINOv3Backbone
from navsim.agents.diffusiondrive.Trimodal_SA import TrimodalSelfAttention
from navsim.agents.diffusiondrive.diffusion_utils.sampling import dpm_sampler
from navsim.agents.diffusiondrive.diffusion_utils.sde import VPSDE_linear
import logging
from pointnet2_ops import pointnet2_utils
from navsim.agents.diffusiondrive.uni3d_config import Uni3DConfig

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number) 
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1,2).contiguous()
    return fps_data

# https://github.com/Strawberry-Eat-Mango/PCT_Pytorch/blob/main/util.py 
def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim = -1, largest=False, sorted=False)
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm;
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist    


class PatchDropout(nn.Module):
    """ 
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token
        logging.info("patch dropout prob is {}".format(prob))

    def forward(self, x):
        # if not self.training or self.prob == 0.:
        #     return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size

    def forward(self, xyz):
        '''
            input: B N 3
            ---------------------------
            output: B G M 3
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = fps(xyz, self.num_group) # B G 3
        # knn to get the neighborhood
        # _, idx = self.knn(xyz, center) # B G M
        idx = knn_point(self.group_size, xyz, center) # B G M
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points
        idx = idx + idx_base
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        #neighborhood_color = color.view(batch_size * num_points, -1)[idx, :]
        #neighborhood_color = neighborhood_color.view(batch_size, self.num_group, self.group_size, 3).contiguous()

        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)

        features = neighborhood
        return neighborhood, center, features

class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

class PTv3FeatureExtractor(nn.Module):
    def __init__(self, point_transformer, freeze_backbone, config: Uni3DConfig):
        super().__init__()
        from easydict import EasyDict
        
        self.trans_dim = config.pc_feat_dim # 384
        self.embed_dim = config.embed_dim # 1024
        self.group_size = config.group_size # 64
        self.num_group = config.num_group # 1024
        # grouper
        self.group_divider = Group(num_group = self.num_group, group_size = self.group_size)
        # define the encoder
        self.encoder_dim =  config.pc_encoder_dim # 256
        self.encoder = Encoder(encoder_channel = self.encoder_dim) #512
       
        # bridge encoder and transformer
        self.encoder2trans = nn.Linear(self.encoder_dim,  self.trans_dim) # 512 --> 384
        
        self.encoder_norm = nn.LayerNorm(self.encoder_dim)
        
        # bridge transformer and clip embedding
        #self.trans2embed = nn.Linear(self.trans_dim,  self.embed_dim)
        #self.cls_token = nn.Parameter(torch.zeros(1, 1, self.trans_dim)) #384
        #self.cls_pos = nn.Parameter(torch.randn(1, 1, self.trans_dim))

        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, self.trans_dim)
        )  
        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(config.patch_dropout) if config.patch_dropout > 0. else nn.Identity()
        self.visual = point_transformer

        num_blocks = len(self.visual.blocks)
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.trans_dim) for _ in range(num_blocks)
        ])

        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self): #v63

        """Freeze backbone network."""
        modules_to_freeze = [
            self.visual,
            self.pos_embed,
            self.encoder2trans,
        ]
        
        for module in modules_to_freeze:
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
        
        for i in range(len(self.visual.blocks)):
            block = self.visual.blocks[i]
            for param in block.norm2.parameters():
                param.requires_grad = True

            for param in block.mlp.parameters():
                param.requires_grad = True

            for param in block.norm1.parameters():
                param.requires_grad = True
            for param in block.attn.proj.parameters():
                param.requires_grad = True

    def get_layer_feature(self, center, features, layer_idx):
        # divide the point cloud in the same form. This is important
        rank = dist.get_rank() if dist.is_initialized() else 0
        if layer_idx==0:
        # encoder the input cloud patches
            group_input_tokens = self.encoder(features)  #  B G N [B,512,512]
            # with open(f'/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/new_output{rank}.txt', 'a') as f:
            #     f.write(f'layer_idx {layer_idx}, group_input_tokens1 max is {group_input_tokens.max()}\n')
            group_input_tokens = self.encoder2trans(group_input_tokens) #[B,512,384]
            # with open(f'/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/new_output{rank}.txt', 'a') as f:
            #     f.write(f'layer_idx {layer_idx}, group_input_tokens2 max is {group_input_tokens.max()}\n')
            # prepare cls

            # add pos embedding
            pos = self.pos_embed(center) #[B,512,384]
            # final input
            
            # transformer
            x = group_input_tokens + pos
            # x = x.half()
           
            # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
            x = self.patch_dropout(x)

            features = self.visual.pos_drop(x) #[B,512,384]

        # ModuleList not support forward
        #for i, blk in enumerate(self.visual.blocks):
        x = self.visual.blocks[layer_idx](features)

        x = self.layer_norms[layer_idx](x)

        return x

    

class TimestepEmbedder(nn.Module):
    """Timestep encoder."""
    
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(max_period)) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def modulate(x, shift, scale):
    """AdaLN modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DINOv3FeatureExtractor(nn.Module):
    """DINOv3 feature extractor."""
    
    def __init__(
        self, 
        weights_path: str,
        patch_size: int = 16,
        num_layers: int = 12,
        freeze_backbone: bool = True
    ):
        super().__init__()
        
        self.backbone = DINOv3Backbone(
            weights_path=weights_path,
            patch_size=patch_size,
            out_channels=(512, 512, 512, 512)
        )
        
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.layer_indices = list(range(num_layers))

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(384) for _ in range(num_layers)
        ])

        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self): #v63
        for param in self.backbone.parameters():  #dino freeze
            param.requires_grad = False
        

        for i, block in enumerate(self.backbone.raw_model.blocks):
            for name, param in block.norm2.named_parameters():
                param.requires_grad = True
            for name, param in block.mlp.named_parameters():
                param.requires_grad = True
            for name, param in block.ls2.named_parameters():
                param.requires_grad = True

 
    


    def get_layer_feature(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        output = self.backbone.raw_model.get_intermediate_layers(
            x,
            layer_idx=layer_idx,
            n=self.layer_indices,
            return_class_token=False,
            norm=False
        )

        #output = self.layer_norms[layer_idx](output)
        #output = torch.clamp(output, min=-20.0, max=20.0)

        return output


class HierarchicalFusionBlock(nn.Module):
    """Trimodal fusion block."""
    
    def __init__(
        self, 
        dim: int = 384, 
        num_heads: int = 8, 
        dropout: float = 0.1, 
        mlp_ratio: float = 4.0
    ):
        super().__init__()
        
        self.dim = dim
        
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout, batch_first=True)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True)
        )
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp1 = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=approx_gelu, 
            drop=0
        )
        
        self.trimodal_attn = TrimodalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            window_size=(-1, -1),
            qk_norm=True
        )
        
        self.norm3 = nn.LayerNorm(dim)
        self.mlp2 = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0
        )
    
    def forward(
        self, 
        traj_feature: torch.Tensor,
        camera_feature: torch.Tensor,
        lidar_feature: torch.Tensor,
        time_cond: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(time_cond).chunk(6, dim=1)
        
        modulated_traj = modulate(self.norm1(traj_feature), shift_msa, scale_msa)
        traj_feature = traj_feature + gate_msa.unsqueeze(1) * self.self_attn(
            modulated_traj, modulated_traj, modulated_traj, 
            key_padding_mask=attn_mask
        )[0]
        
        modulated_traj = modulate(self.norm2(traj_feature), shift_mlp, scale_mlp)
        traj_feature = traj_feature + gate_mlp.unsqueeze(1) * self.mlp1(modulated_traj)
        
        fused_camera, fused_lidar, fused_traj = self.trimodal_attn(
            x1=camera_feature,
            x2=lidar_feature,
            x3=traj_feature
        )
        
        fused_traj = fused_traj + self.mlp2(self.norm3(fused_traj))
        
        return fused_camera, fused_lidar, fused_traj


class FinalLayer(nn.Module):
    """Final output layer."""
    
    def __init__(self, hidden_size: int, num_tokens: int = 8, output_dim_per_token: int = 3):
        super().__init__()
        self.num_tokens = num_tokens
        self.output_dim_per_token = output_dim_per_token
        
        self.norm_final = nn.LayerNorm(hidden_size)
        
        self.proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 2, bias=True),
            nn.GELU(approximate="tanh"),
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_dim_per_token, bias=True)
        )
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )
    
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(y).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class DiffusionTrajectoryPredictor(nn.Module):
    """Diffusion trajectory predictor."""
    
    def __init__(
        self,
        dinov3_extractor_camera: DINOv3FeatureExtractor,
        ptv3_extractor_lidar: PTv3FeatureExtractor,
        num_traj_points: int = 8,
        coord_dim: int = 3,
        hidden_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        model_type: str = "x_start"
    ):
        super().__init__()
        
        self.num_traj_points = num_traj_points
        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.model_type = model_type
        
        self.dinov3_extractor_camera = dinov3_extractor_camera
        self.ptv3_extractor_lidar = ptv3_extractor_lidar
        
        self.sde = VPSDE_linear()
        
        self.traj_proj = nn.Linear(coord_dim, hidden_dim)
        self.traj_pos_embed = nn.Parameter(torch.zeros(1, num_traj_points, hidden_dim))
        trunc_normal_(self.traj_pos_embed, std=0.02)
        
        self.time_embedder = TimestepEmbedder(hidden_dim)
  

        self.fusion_blocks = nn.ModuleList([
            HierarchicalFusionBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio
            )
            for _ in range(num_layers)
        ])
        
        self.final_layer = FinalLayer(
            hidden_size=hidden_dim,
            num_tokens=num_traj_points,
            output_dim_per_token=coord_dim
        )


    def forward(
        self,
        noisy_traj: torch.Tensor,
        t: torch.Tensor,
        camera_feature: torch.Tensor,
        lidar_point_cloud: Dict,
        status_encoding: torch.Tensor
    ) -> torch.Tensor:
        
        B = noisy_traj.shape[0]
        
        traj_feature = self.traj_proj(noisy_traj)
        time_emb = self.time_embedder(t)
        cond = status_encoding + time_emb
        
        attn_mask = torch.zeros((B, self.num_traj_points), dtype=torch.bool, device=noisy_traj.device)

        lidar_feature = lidar_point_cloud['lidar_feature'] 
        center = lidar_point_cloud['lidar_center']

        # rank = dist.get_rank() if dist.is_initialized() else 0
        #     f.write('training new sample' + '\n')
           
        for layer_idx in range(self.num_layers):
            
            camera_feature = self.dinov3_extractor_camera.get_layer_feature(
                camera_feature, 
                layer_idx
            )
            # with open(f'/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/new_output{rank}.txt', 'a') as f:
            #     f.write(f'layer_idx {layer_idx}, camera feature max is {camera_feature.max()}\n')
            lidar_feature = self.ptv3_extractor_lidar.get_layer_feature( #[B,5000,384]
                center,
                lidar_feature, 
                layer_idx
            )
            # with open(f'/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/new_output{rank}.txt', 'a') as f:
            #     f.write(f'layer_idx {layer_idx}, lidar feature max is {lidar_feature.max()}\n')
            camera_feature, lidar_feature, traj_feature = self.fusion_blocks[layer_idx](
                traj_feature=traj_feature,
                camera_feature=camera_feature,
                lidar_feature=lidar_feature,
                time_cond=cond,
                attn_mask=attn_mask
            )
            # with open(f'/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/new_output{rank}.txt', 'a') as f:
            #     f.write(f'layer_idx {layer_idx}, camera feature max is {camera_feature.max()}, lidar feature max is {lidar_feature.max()}, traj_feature max is {traj_feature.max()} \n')
        output = self.final_layer(traj_feature, cond)

        if self.model_type == "score":
            std = self.sde.marginal_prob_std(t)[:, None, None]
            return output / (std + 1e-6)
        else:
            return output


class HierarchicalFusionModule(nn.Module):
    """Hierarchical fusion module."""
    
    def __init__(
        self,
        dinov3_weights_path: str,
        num_traj_points: int = 8,
        coord_dim: int = 3,
        hidden_dim: int = 384,
        num_layers: int = 12,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        model_type: str = "x_start",
        lidar_in_channels: int = 3,
        ptv3_grid_sizes: List[float] = None
    ):
        super().__init__()
        
        self.num_traj_points = num_traj_points
        self.coord_dim = coord_dim
        
        self.dinov3_extractor_camera = DINOv3FeatureExtractor(
            weights_path=dinov3_weights_path,
            patch_size=patch_size,
            num_layers=num_layers,
            freeze_backbone=True
        )
        

        self.ptv3_extractor_lidar = PTv3FeatureExtractor(
            point_transformer = timm.create_model(Uni3DConfig.pc_model, checkpoint_path=None, drop_path_rate=Uni3DConfig.drop_path_rate),
            freeze_backbone=True,
            config = Uni3DConfig
        )
        self.ptv3_extractor_lidar = self.ptv3_extractor_lidar.to("cuda")
        
        self.diffusion_model = DiffusionTrajectoryPredictor(
            dinov3_extractor_camera=self.dinov3_extractor_camera,
            ptv3_extractor_lidar=self.ptv3_extractor_lidar,
            num_traj_points=num_traj_points,
            coord_dim=coord_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            mlp_ratio=mlp_ratio,
            model_type=model_type
        )
        
        self.sde = VPSDE_linear()
        
        config = TransfuserConfig()
        self.loss_computer = LossComputer(config)
    
    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]
        
        odo_info_fut_x = 2 * (odo_info_fut_x + 1.2) / 56.9 - 1
        odo_info_fut_y = 2 * (odo_info_fut_y + 20) / 46 - 1
        odo_info_fut_head = 2 * (odo_info_fut_head + 2) / 3.9 - 1
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]
        
        odo_info_fut_x = (odo_info_fut_x + 1) / 2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1) / 2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1) / 2 * 3.9 - 2
        
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)
    
    def forward_train(
        self,
        camera_image: torch.Tensor,
        lidar_point_cloud: Dict,
        status_encoding: torch.Tensor,
        gt_trajectory: torch.Tensor,
        eps: float = 1e-3
    ) -> Dict[str, torch.Tensor]:
        
        bs = camera_image.shape[0]
        device = camera_image.device
        
        t = torch.rand(bs, device=device) * (1 - eps) + eps
        z = torch.randn_like(gt_trajectory['trajectory'], device=device).float()
        
        gt_norm = self.norm_odo(gt_trajectory['trajectory'])
        mean, std = self.sde.marginal_prob(gt_norm, t)
        std = std.view(-1, *([1] * (len(gt_norm.shape) - 1)))
        noisy_traj = mean + std * z
        noisy_traj = torch.clamp(noisy_traj, min=-1, max=1).float()
        
        pred = self.diffusion_model(
            noisy_traj=noisy_traj,
            t=t,
            camera_feature=camera_image,
            lidar_point_cloud=lidar_point_cloud,
            status_encoding=status_encoding
        )
        
        denoised_traj = self.denorm_odo(pred)
        denoised_traj[..., 2] = denoised_traj[..., 2].tanh() * np.pi
        
        trajectory_loss_dict = {}
        trajectory_loss = self.loss_computer(denoised_traj, gt_trajectory)
        trajectory_loss_dict[f"trajectory_loss_0"] = trajectory_loss
        
        return {
            "trajectory": denoised_traj,
            "trajectory_loss": trajectory_loss,
            "trajectory_loss_dict": trajectory_loss_dict
        }
    
    def forward_test(
        self,
        camera_image: torch.Tensor,
        lidar_point_cloud: Dict,
        status_encoding: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
            bs = camera_image.shape[0]
            device = camera_image.device
            
            xT = torch.randn(bs, self.num_traj_points, self.coord_dim, device=device)
            xT = torch.clamp(xT, min=-1, max=1).float()
            


            x0 = dpm_sampler(
                model=self.diffusion_model,
                x_T=xT,
                other_model_params={
                    "camera_feature": camera_image,
                    "lidar_point_cloud": lidar_point_cloud,
                    "status_encoding": status_encoding,
                },
                diffusion_steps=2, 
                dpm_solver_params={
                    "correcting_xt_fn": None,
                },
                model_wrapper_params={
                    "classifier_fn": None,
                    "guidance_scale": 0.5,
                    "guidance_type": "uncond"
                },
            )
            
            trajectory = self.denorm_odo(x0)
            trajectory[..., 2] = trajectory[..., 2].tanh() * np.pi
            
            return {"trajectory": trajectory}
    
    def forward(
        self,
        camera_image: torch.Tensor,
        lidar_point_cloud: Dict,
        status_encoding: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        if self.training:
            assert gt_trajectory is not None
            return self.forward_train(camera_image, lidar_point_cloud, status_encoding, gt_trajectory)
        else:
            return self.forward_test(camera_image, lidar_point_cloud, status_encoding)





    # def forward_test(
    #     self,
    #     camera_image: torch.Tensor,
    #     lidar_point_cloud: Dict,
    #     language_token: torch.Tensor,
    #     status_encoding: torch.Tensor,
    # ) -> Dict[str, torch.Tensor]:
        
    #     bs = camera_image.shape[0]
    #     device = camera_image.device

    #     class _TrajOnlyModel(torch.nn.Module):
    #         def __init__(self, parent):
    #             super().__init__()
    #             self.parent = parent
    #             self.model_type = "x_start"

    #         def forward(self, x, t, **kwargs):
    #             sigma, _ = self.parent.noise(t)
    #             mdlm_sigma = self.parent._process_sigma(sigma[:, None])
    #             lang_tokens = kwargs["language_feature"]
    #             traj_pred, _ = self.parent.diffusion_model(
    #                 noisy_traj=x,
    #                 t=t,
    #                 mdlm_sigma=mdlm_sigma,
    #                 camera_feature=kwargs["camera_feature"],
    #                 lidar_point_cloud=kwargs["lidar_point_cloud"],
    #                 language_feature=lang_tokens,
    #                 status_encoding=kwargs["status_encoding"],
    #             )
    #             return traj_pred

    #     traj_model = _TrajOnlyModel(self)
    #     traj_x_T = torch.randn(bs, self.num_traj_points, self.coord_dim, device=device)
    #     traj_x_T = torch.clamp(traj_x_T, min=-1, max=1).float()
        
    #     lang_tokens = language_token.long().clamp(0, self.vocab_size - 1)

    #     traj_x_0 = dpm_sampler(
    #         model=traj_model,
    #         x_T=traj_x_T,
    #         other_model_params={
    #             "camera_feature": camera_image,
    #             "lidar_point_cloud": lidar_point_cloud,
    #             "status_encoding": status_encoding,
    #             "language_feature": lang_tokens,
    #         },
    #         #diffusion_steps=10,
    #         diffusion_steps=2,
    #         dpm_solver_params={
    #             "correcting_xt_fn": None,
    #         },
    #         model_wrapper_params={
    #             "classifier_fn": None,
    #             "guidance_scale": 0.5,
    #             "guidance_type": "uncond",
    #         },
    #     )

    #     # Decode language once after trajectory sampling.
    #     t_decode = torch.full((bs,), 1e-3, device=device)
    #     sigma_decode, _ = self.noise(t_decode)
    #     mdlm_sigma_decode = self._process_sigma(sigma_decode[:, None])
    #     _, lang_logits = self.diffusion_model(
    #         noisy_traj=traj_x_0,
    #         t=t_decode,
    #         mdlm_sigma=mdlm_sigma_decode,
    #         camera_feature=camera_image,
    #         lidar_point_cloud=lidar_point_cloud,
    #         language_feature=lang_tokens,
    #         status_encoding=status_encoding,
    #     )
    #     lang_state = lang_logits.argmax(dim=-1).long().clamp(0, self.vocab_size - 1)

    #     trajectory = self.denorm_odo(traj_x_0)
    #     trajectory[..., 2] = trajectory[..., 2].tanh() * np.pi
        
    #     return {
    #         "trajectory": trajectory,
    #         "language_tokens": lang_state
    #     }
    
    # def forward(
    #     self,
    #     camera_image: torch.Tensor,
    #     lidar_point_cloud: Dict,
    #     language_token: torch.Tensor,
    #     status_encoding: torch.Tensor,
    #     gt_trajectory: Optional[torch.Tensor] = None
    # ) -> Dict[str, torch.Tensor]:
        
    #     if self.training:
    #         assert gt_trajectory is not None
    #         return self.forward_train(
    #             camera_image,
    #             lidar_point_cloud,
    #             language_token,
    #             status_encoding,
    #             gt_trajectory,
    #         )
    #     else:
    #         return self.forward_test(
    #             camera_image,
    #             lidar_point_cloud,
    #             language_token,
    #             status_encoding
    #         )

