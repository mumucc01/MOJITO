################# Ours ###################
from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import copy
import math
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
from navsim.agents.diffusiondrive.diffusion_planner_config import DP_Config
from navsim.mojito_paths import DIFFUSION_PLANNER_ROOT, pretrained_path

diffusion_planner_path = str(DIFFUSION_PLANNER_ROOT)
if diffusion_planner_path not in sys.path:
    sys.path.append(diffusion_planner_path)

from diffusion_planner.model.diffusion_utils.sde import VPSDE_linear
from diffusion_planner.model.diffusion_planner import Diffusion_Planner_Decoder
from navsim.agents.diffusiondrive.visualization_czj import save_as_8bit, visualize_noise_grid_on_lidar
from navsim.agents.diffusiondrive.trimodal_fusion import Trimodal_Fusion

from navsim.agents.diffusiondrive.hierarchical_fusion_module_pt3 import HierarchicalFusionModule
import torchvision
from torchvision.transforms import v2



def make_transform_for_tensor():
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([normalize])


def _rank_prefix():
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                return f"[rank{torch.distributed.get_rank()}]"
        except Exception:
            pass
        return "[rank?]"

@torch.no_grad()
def _check(name: str, t: torch.Tensor, *, stop: bool = False):
    """Print dtype/shape/finite/min/max; optionally raise on non-finite values."""
    if t is None:
        print(_rank_prefix(), f"{name}: None")
        return True
    finite = torch.isfinite(t).all().item()
    x = torch.nan_to_num(t.detach().float(), nan=0.0, posinf=0.0, neginf=0.0)
    mn = x.min().item() if x.numel() else float("nan")
    mx = x.max().item() if x.numel() else float("nan")
    if (not finite):
        print(_rank_prefix(), f"NON-FINITE {name}: dtype={t.dtype}, shape={tuple(t.shape)}, min={mn:.6g}, max={mx:.6g}")
        if stop:
            raise FloatingPointError(f"Non-finite detected at {name}")
        return False
    return True

class V2TransfuserModel(nn.Module):
    """Torch module for Transfuser."""

    def __init__(self, config: TransfuserConfig):
        """
        Initializes TransFuser torch module.
        :param config: global config dataclass of TransFuser.
        """

        super().__init__()

        self._config = config
        #self._backbone = TransfuserBackbone(config)

        self._bev_downscale = nn.Conv2d(384, DP_Config().hidden_dim, kernel_size=1)
        #self._status_encoding = nn.Linear(4 + 2 + 2, DP_Config().hidden_dim)
        self._status_encoding = nn.Linear(4 + 2 + 2, 384)
    

        self._trajectory_head = HierarchicalFusionModule(
            dinov3_weights_path=pretrained_path("dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"),
            num_traj_points=8,
            coord_dim=3,
            hidden_dim=384,
            num_layers=12,
            num_heads=8,
            #dropout=0.1,
            dropout=0.2,
            mlp_ratio=4.0,
            patch_size=16,
            model_type="x_start"
        ).to("cuda")
        
        self.bev_proj = nn.Sequential(
            #*linear_relu_ln(256, 1, 1,320),
            *linear_relu_ln(3072, 1, 1,3136),
        )

        self._query_splits = [
            config.num_bounding_boxes,
        ]

        channel = config.bev_features_channels
        self.relu = nn.ReLU(inplace=True)
        self.c5_conv = nn.Conv2d(384, channel, (1, 1))
        self.up_conv5 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        self.up_conv4 = nn.Conv2d(channel, channel, (3, 3), padding=1)
        # lateral
        self._keyval_embedding = nn.Embedding(8**2 + 1, config.tf_d_model)
        #self._keyval_embedding = nn.Embedding(16**2+1, config.tf_d_model)
        
        self._query_embedding = nn.Embedding(sum(self._query_splits), config.tf_d_model)

        tf_decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.tf_d_model,
            nhead=config.tf_num_head,
            dim_feedforward=config.tf_d_ffn,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        
        #self.keyval_ln = nn.LayerNorm(config.tf_d_model)
        #self.query_ln  = nn.LayerNorm(config.tf_d_model)

        self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, config.tf_num_layers)

        self._bev_semantic_head = nn.Sequential(
            nn.Conv2d(
                config.bev_features_channels,
                config.bev_features_channels,
                kernel_size=(3, 3),
                stride=1,
                padding=(1, 1),
                bias=True,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                config.bev_features_channels,
                config.num_bev_classes,
                kernel_size=(1, 1),
                stride=1,
                padding=0,
                bias=True,
            ),
            nn.Upsample(
                size=(config.lidar_resolution_height // 2, config.lidar_resolution_width),
                mode="bilinear",
                align_corners=False,
            ),
        )

        self.upsample = nn.Upsample(
                scale_factor=config.bev_upsample_factor, mode="bilinear", align_corners=False
            )
        self.upsample2 = nn.Upsample(
            size=(
                config.lidar_resolution_height // config.bev_down_sample_factor,
                config.lidar_resolution_width // config.bev_down_sample_factor,
            ),
            mode="bilinear",
            align_corners=False,
        )
        self._agent_head = AgentHead(
            num_agents=config.num_bounding_boxes,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
        )

        hidden_in_mlp = 256 + 1024
        hidden_out_mlp = 64
        self.bev_mlp = nn.Sequential(
            nn.Linear(hidden_in_mlp, 512),
            nn.GELU(),
            nn.Linear(512, hidden_out_mlp),
        )

    def top_down(self, x):
        p5 = self.relu(self.c5_conv(x))
        p4 = self.relu(self.up_conv5(self.upsample(p5)))
        p3 = self.relu(self.up_conv4(self.upsample2(p4)))

        return p3

    def _to_device(self, pc: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in pc.items()}

    def forward(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        camera_feature: torch.Tensor = features["camera_feature"].to("cuda") #[B,C,H,W]
        lidar_feature: torch.Tensor = features["lidar_feature"].to("cuda")
        lidar_center: torch.Tensor = features["lidar_center"].to("cuda")

        lidar_point_cloud = {"lidar_feature" : lidar_feature, "lidar_center" : lidar_center}

        status_feature: torch.Tensor = features["status_feature"].to("cuda") # [8] 
       
        
        transform = make_transform_for_tensor() 
        camera_feature = transform(camera_feature) 
        
        #save_as_8bit(np.array(lidar_feature[0,0,...].cpu()),"/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/lidar_feature.png")

        batch_size = status_feature.shape[0]

        status_encoding = self._status_encoding(status_feature) #[B,8] --> [B,256]

        #output: Dict[str, torch.Tensor] = {"bev_semantic_map": bev_semantic_map}
        output: Dict[str, torch.Tensor] = {}
        #self._trajectory_head = self._trajectory_head.to("cuda")
        trajectory = self._trajectory_head(camera_feature, lidar_point_cloud, status_encoding, targets)
        output.update(trajectory)

        return output

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, BoundingBox2DIndex.size()),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, 1),
        )


    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., BoundingBox2DIndex.POINT] = agent_states[..., BoundingBox2DIndex.POINT].tanh() * 32
        agent_states[..., BoundingBox2DIndex.HEADING] = agent_states[..., BoundingBox2DIndex.HEADING].tanh() * np.pi

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}

class DiffMotionPlanningRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        ego_fut_ts=8,
        ego_fut_mode=20,
        if_zeroinit_reg=True,
    ):
        super(DiffMotionPlanningRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        #self.plan_cls_branch = nn.Sequential(
        #    *linear_relu_ln(embed_dims, 1, 2),
        #    nn.Linear(embed_dims, 1),
        #)
        self.plan_reg_branch = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(),
            nn.Linear(embed_dims, ego_fut_ts * 3),
        )
        #self.plan_reg_branch = nn.Sequential(
        #    nn.Linear(embed_dims, embed_dims),
        #    nn.LeakyReLU(0.01),
        #    nn.Linear(embed_dims, embed_dims),
        #    nn.LeakyReLU(0.01),
        #    nn.Linear(embed_dims, ego_fut_ts * 3),
        #)
        self.if_zeroinit_reg = False

        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_reg:
            nn.init.constant_(self.plan_reg_branch[-1].weight, 0)
            nn.init.constant_(self.plan_reg_branch[-1].bias, 0)

        bias_init = bias_init_with_prob(0.01)
        #nn.init.constant_(self.plan_cls_branch[-1].bias, bias_init)
    def forward(
        self,
        traj_feature,
    ):
        bs, ego_fut_mode, _ = traj_feature.shape #[B,1,256]
        decoder_pos = gen_sineembed_for_position_decoder(traj_feature).float()
        traj_feature = traj_feature + decoder_pos
        # 6. get final prediction
        #traj_feature = traj_feature.view(bs, ego_fut_mode,-1)
        #plan_cls = self.plan_cls_branch(traj_feature).squeeze(-1)

        traj_delta = self.plan_reg_branch(traj_feature)
        plan_reg = traj_delta.reshape(bs,ego_fut_mode, self.ego_fut_ts, 3)

        return plan_reg, None

class ModulationLayer(nn.Module):

    def __init__(self, embed_dims: int, condition_dims: int):
        super(ModulationLayer, self).__init__()
        self.if_zeroinit_scale=False
        self.embed_dims = embed_dims
        self.scale_shift_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(condition_dims, embed_dims*2),
        )
        self.init_weight()

    def init_weight(self):
        if self.if_zeroinit_scale:
            nn.init.constant_(self.scale_shift_mlp[-1].weight, 0)
            nn.init.constant_(self.scale_shift_mlp[-1].bias, 0)

    def forward(
        self,
        traj_feature,
        time_embed,
        global_cond=None,
        global_img=None,
    ):
        if global_cond is not None:
            global_feature = torch.cat([
                    global_cond, time_embed
                ], axis=-1)
        else:
            global_feature = time_embed
        if global_img is not None:
            global_img = global_img.flatten(2,3).permute(0,2,1).contiguous()
            global_feature = torch.cat([
                    global_img, global_feature
                ], axis=-1)
        
        scale_shift = self.scale_shift_mlp(global_feature)
        scale,shift = scale_shift.chunk(2,dim=-1)
        traj_feature = traj_feature * (1 + scale) + shift
        return traj_feature

class CustomTransformerDecoderLayer(nn.Module):
    def __init__(self, 
                 num_poses,
                 d_model,
                 d_ffn,
                 config,
                 ):
        super().__init__()
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.cross_bev_attention = GridSampleCrossBEVAttention(
            config.tf_d_model,
            config.tf_num_head,
            num_points=num_poses,
            config=config,
            in_bev_dims=256,
        )
        self.cross_agent_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.cross_ego_attention = nn.MultiheadAttention(
            config.tf_d_model,
            config.tf_num_head,
            dropout=config.tf_dropout,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(config.tf_d_model, config.tf_d_ffn),
            nn.ReLU(),
            nn.Linear(config.tf_d_ffn, config.tf_d_model),
        )
        self.norm1 = nn.LayerNorm(config.tf_d_model)
        self.norm2 = nn.LayerNorm(config.tf_d_model)
        self.norm3 = nn.LayerNorm(config.tf_d_model)
        self.time_modulation = ModulationLayer(config.tf_d_model,256)
        self.task_decoder = DiffMotionPlanningRefinementModule(
            embed_dims=config.tf_d_model,
            ego_fut_ts=num_poses,
            ego_fut_mode=20,
        )

    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        traj_feature = self.cross_bev_attention(traj_feature,noisy_traj_points,bev_feature,bev_spatial_shape)
        traj_feature = traj_feature + self.dropout(self.cross_agent_attention(traj_feature, agents_query,agents_query)[0])
        traj_feature = self.norm1(traj_feature)
        
        # traj_feature = traj_feature + self.dropout(self.self_attn(traj_feature, traj_feature, traj_feature)[0])

        # 4.5 cross attention with  ego query
        traj_feature = traj_feature + self.dropout1(self.cross_ego_attention(traj_feature, ego_query,ego_query)[0])
        traj_feature = self.norm2(traj_feature)
        
        # 4.6 feedforward network
        traj_feature = self.norm3(self.ffn(traj_feature))
        # 4.8 modulate with time steps
        traj_feature = self.time_modulation(traj_feature, time_embed,global_cond=None,global_img=global_img) #[B,20,256]
        
        # 4.9 predict the offset & heading
        poses_reg, poses_cls = self.task_decoder(traj_feature) #bs,20,8,3; bs,20
        poses_reg[...,:2] = poses_reg[...,:2] + noisy_traj_points
        poses_reg[..., StateSE2Index.HEADING] = poses_reg[..., StateSE2Index.HEADING].tanh() * np.pi

        return poses_reg, poses_cls
def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class CustomTransformerDecoder(nn.Module):
    def __init__(
        self, 
        decoder_layer, 
        num_layers,
        norm=None,
    ):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
    
    def forward(self, 
                traj_feature, 
                noisy_traj_points, 
                bev_feature, 
                bev_spatial_shape, 
                agents_query, 
                ego_query, 
                time_embed, 
                status_encoding,
                global_img=None):
        poses_reg_list = []
        poses_cls_list = []
        traj_points = noisy_traj_points
        for mod in self.layers:
            poses_reg, poses_cls = mod(traj_feature, traj_points, bev_feature, bev_spatial_shape, agents_query, ego_query, time_embed, status_encoding,global_img)
            poses_reg_list.append(poses_reg)
            poses_cls_list.append(poses_cls)
            traj_points = poses_reg[...,:2].clone().detach()
        return poses_reg_list, poses_cls_list

class TrajectoryHead(nn.Module):
    """Trajectory prediction head."""

    def __init__(self, num_poses: int, d_ffn: int, d_model: int, plan_anchor_path: str,config: TransfuserConfig):
        """
        Initializes trajectory head.
        :param num_poses: number of (x,y,θ) poses to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(TrajectoryHead, self).__init__()
        self.transfuser_config = config
        self._num_poses = num_poses
        self._d_model = d_model
        self._d_ffn = d_ffn
        self.diff_loss_weight = 2.0
        self.ego_fut_mode = 20

        self.diffusion_scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_schedule="scaled_linear",
            prediction_type="sample",
        )
        self.VPSDE_linear = VPSDE_linear()

        plan_anchor = np.load(plan_anchor_path)

        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        ) # 20,8,2
        self.plan_anchor_encoder = nn.Sequential(
            #*linear_relu_ln(d_model, 1, 1,512),
            *linear_relu_ln(d_model, 1, 1,512),
            nn.Linear(d_model, d_model),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Linear(d_model * 4, d_model),
        )

        diff_decoder_layer = CustomTransformerDecoderLayer(
            num_poses=num_poses,
            d_model=d_model,
            d_ffn=d_ffn,
            config=config,
        )
        #self.diff_decoder = CustomTransformerDecoder(diff_decoder_layer, 2)
        self.dp_config = DP_Config()
        self.dp_decoder =  Diffusion_Planner_Decoder(self.dp_config)
        self.loss_computer = LossComputer(config)

        self.task_decoder = DiffMotionPlanningRefinementModule( #1280-->80*3
            embed_dims= self.dp_config.output_dim,
            ego_fut_ts= 8,
            ego_fut_mode=1,
        )

    def gaussian_noise_constrained(self, mask, noise_init, x_min, y_min, ppm):
        """
         (x,y,heading)  x,y  mask （），heading 。

        Args:
            mask: torch.Tensor, shape [H,W] or [B,H,W], 1=, 0=
            noise_init: torch.Tensor, shape [B,T,3]，
            x_min, y_min: float, BEV （meter）
            ppm: float, pixels per meter

        Returns:
            out: torch.Tensor, shape [B,T,3]，（ x,y ）
        """
        device = noise_init.device

        noise_init_ = noise_init.unsqueeze(1)  # [B,1,T,3]
        B, P, T, D = noise_init_.shape
        assert D >= 3, f"noise_init last dim should be >=3, got {D}"

        if mask.dim() == 2:
            H, W = mask.shape
            mask_b = mask.to(torch.int).unsqueeze(0).expand(B, H, W).contiguous()
        elif mask.dim() == 3:
            mask_b = mask.to(torch.int)
            Bm, H, W = mask_b.shape
            assert Bm == B, f"mask batch {Bm} != noise batch {B}"
        else:
            raise ValueError(f"mask dim must be 2 or 3, got {mask.shape}")

        out = noise_init_.clone()  # [B,1,T,3]

        # xy_meter: [B,1,T,2]
        xy_meter = self.denorm_odo(noise_init_)[..., :2]
        x = xy_meter[..., 0]  # [B,1,T]
        y = xy_meter[..., 1]  # [B,1,T]

        ix = ((x - x_min) * ppm).long()  # [B,1,T]
        iy = ((y - y_min) * ppm).long()  # [B,1,T]

        in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)  # [B,1,T]
        valid = (~in_bounds).clone()
        
        #valid = torch.zeros((B, P, T), dtype=torch.bool, device=device)

        if in_bounds.any():
            b_idx, p_idx, t_idx = torch.nonzero(in_bounds, as_tuple=True)  # [M]
            iy_ib = iy[b_idx, p_idx, t_idx]
            ix_ib = ix[b_idx, p_idx, t_idx]
            valid[b_idx, p_idx, t_idx] = (mask_b[b_idx, iy_ib, ix_ib] == 1)

        invalid = ~valid  # [B,1,T]

        for b in range(B):
            invalid_b = invalid[b]  # [1,T] (P=1)
            if not invalid_b.any():
                continue

            feas_y, feas_x = torch.nonzero(mask_b[b] == 1, as_tuple=True)  # [N]
            if feas_y.numel() == 0:
                raise ValueError(f"Batch {b}: mask has no feasible cells (mask==1).")

            feas_x_meter = x_min + (feas_x.float() + 0.5) / ppm
            feas_y_meter = y_min + (feas_y.float() + 0.5) / ppm
            feas_centers = torch.stack([feas_x_meter, feas_y_meter], dim=-1)  # [N,2]

            feas_origin_dist2 = (feas_centers ** 2).sum(dim=-1)  # [N]

            p_i, t_i = torch.nonzero(invalid_b, as_tuple=True)  # [M_b], [M_b]
            xy_invalid = xy_meter[b, p_i, t_i]  # [M_b,2]

            diff = xy_invalid.unsqueeze(1) - feas_centers.unsqueeze(0)
            dist2 = (diff ** 2).sum(dim=-1)

            min_dist2, _ = dist2.min(dim=1)

            eps = 1e-8
            same_min = (dist2 - min_dist2.unsqueeze(1)).abs() < eps
            big = 1e12
            score = torch.where(
                same_min,
                feas_origin_dist2.unsqueeze(0).expand_as(dist2),
                torch.full_like(dist2, big),
            )

            best_idx = score.argmin(dim=1)              # [M_b]
            nearest_centers = feas_centers[best_idx]    # [M_b,2] meter

            nearest_noise_xy = self.norm_odo(nearest_centers)

            nearest_noise_xy = nearest_noise_xy.to(dtype=out.dtype, device=out.device)
            out[b, p_i, t_i, 0:2] = nearest_noise_xy

        out = out.squeeze(1)
        return out

    def norm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = 2*(odo_info_fut_x + 1.2)/56.9 -1
        odo_info_fut_y = 2*(odo_info_fut_y + 20)/46 -1
        odo_info_fut_head = 2*(odo_info_fut_head + 2)/3.9 -1
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)

    def denorm_odo(self, odo_info_fut):
        odo_info_fut_x = odo_info_fut[..., 0:1]
        odo_info_fut_y = odo_info_fut[..., 1:2]
        odo_info_fut_head = odo_info_fut[..., 2:3]

        odo_info_fut_x = (odo_info_fut_x + 1)/2 * 56.9 - 1.2
        odo_info_fut_y = (odo_info_fut_y + 1)/2 * 46 - 20
        odo_info_fut_head = (odo_info_fut_head + 1)/2 * 3.9 - 2
        return torch.cat([odo_info_fut_x, odo_info_fut_y, odo_info_fut_head], dim=-1)


    def forward(self, image_feature, lidar_feature, status_encoding, targets=None) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""
        if self.training:
            return self.forward_train(image_feature, status_encoding,targets)
        else:
            return self.forward_test(image_feature, status_encoding)


    def forward_train(self, image_feature, lidar_feature, status_encoding, targets=None, eps=1e-3) -> Dict[str, torch.Tensor]:
        bs = image_feature.shape[0]
        device = image_feature.device
       
        t = torch.rand(bs, device=device) * (1 - eps) + eps # [B,]
        z = torch.randn_like(targets['trajectory'], device=device).float() # [B, T, 3]
        mean, std = self.VPSDE_linear.marginal_prob(self.norm_odo(targets['trajectory']),t)
        std = std.view(-1, *([1] * (len(targets['trajectory'].shape)-1)))
        xT = mean + std * z 
        xT = xT.to(device=device)
        xT = torch.clamp(xT, min=-1, max=1)
        lidar_mask = (lidar_feature != 0).float()
        lidar_mask = 1 - lidar_mask #[B,1,256,256]
        #xT = self.gaussian_noise_constrained(lidar_mask.squeeze(1), xT, self.transfuser_config.lidar_min_x, self.transfuser_config.lidar_min_y, self.transfuser_config.pixels_per_meter)

        ego_fut_mode = xT.shape[1] #future_len = 8
        traj_pos_embed = gen_sineembed_for_position(xT,hidden_dim=64).float() #[B,8,64]
        traj_pos_embed = traj_pos_embed.flatten(-2) #[B,512]
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)  #[B,8,32]
        
        traj_feature = self.dp_decoder(traj_feature, image_feature, status_encoding, t)
        # lidar_feature : [B, 261 ,384]

        #denoised_traj = self.dp_decoder(traj_feature, bev_feature, lidar_feature, status_encoding, t)
        denoised_traj, _ = self.task_decoder(traj_feature) #[B,1,3072]
        #poses_reg_list['score'] = poses_reg_list['score'] + xT.unsqueeze(1)
        denoised_traj = self.denorm_odo(denoised_traj) #[B,1,8,3]
        denoised_traj[..., 2] = denoised_traj[..., 2].tanh() * np.pi
        trajectory_loss_dict = {}
        ret_traj_loss = 0
        #for idx, (poses_reg) in enumerate(zip(poses_reg_list)):
        trajectory_loss = self.loss_computer(denoised_traj, targets) #/Diffusion_Drive/DiffusionDrive/navsim/agents/diffusiondrive/modules/multimodal_loss.py P131
        trajectory_loss_dict[f"trajectory_loss_{0}"] = trajectory_loss
        ret_traj_loss = trajectory_loss


        return {"trajectory":denoised_traj.squeeze(1),"trajectory_loss":ret_traj_loss,"trajectory_loss_dict":trajectory_loss_dict}

    def forward_test(self, image_feature,lidar_feature,status_encoding, eps=1e-3) -> Dict[str, torch.Tensor]:
        bs = image_feature.shape[0]
        device = image_feature.device
        image_feature = image_feature.to("cuda")
        lidar_feature = lidar_feature.to("cuda")
        status_encoding = status_encoding.to("cuda")
        xT = torch.randn(bs,8,3, device = "cuda").float()
        xT = torch.clamp(xT, min=-1, max=1)
        lidar_mask = (lidar_feature != 0).float()
        lidar_mask = 1 - lidar_mask
        #xT = self.gaussian_noise_constrained(lidar_mask.squeeze(1), xT, self.transfuser_config.lidar_min_x, self.transfuser_config.lidar_min_y, self.transfuser_config.pixels_per_meter)
        

        ego_fut_mode = xT.shape[1] #future_len = 8
        traj_pos_embed = gen_sineembed_for_position(xT,hidden_dim=64) #[B,8,64]
        traj_pos_embed = traj_pos_embed.flatten(-2) #[B,512]
        traj_feature = self.plan_anchor_encoder(traj_pos_embed)
        traj_feature = traj_feature.view(bs,ego_fut_mode,-1)  #[B,8,32]
        
      
        
       
        traj_feature = self.dp_decoder(traj_feature, image_feature, status_encoding, t=None)
        
        denoised_traj, _ = self.task_decoder(traj_feature)
        #poses_reg_list['score'] = poses_reg_list['score'] + xT.unsqueeze(1)
        denoised_traj = self.denorm_odo(denoised_traj)
        denoised_traj[..., 2] = denoised_traj[..., 2].tanh() * np.pi #[B,1,8,3]
        return {"trajectory":denoised_traj.squeeze(1)} #[B,8,3]
        #return {"trajectory":poses_reg_list['score'].squeeze(1)}