import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
from MOJITO.navsim.agents.transfuser.img_transformer import ImageFeatureTransformer
from MOJITO.navsim.agents.transfuser.transfuser_config import TransfuserConfig
from MOJITO.navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
import torch.nn.functional as F 
import os

from torchvision.utils import save_image
from PIL import Image
import numpy as np
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.modeling.torch_module_wrapper import TorchModuleWrapper
from nuplan.planning.training.preprocessing.target_builders.ego_trajectory_target_builder import (
    EgoTrajectoryTargetBuilder,
)
from diffusion_planner.feature_builders.pluto_feature_builder import PlutoFeatureBuilder
trajectory_sampling = TrajectorySampling(num_poses=8, time_horizon=8, interval_length=1)

def save_first_batch_as_image(tensor, save_path):
    """
     [B, C, H, W] Tensorbatch
    
    Args:
        tensor (torch.Tensor): Tensor， [B, C, H, W]
        save_path (str): （，：'/path/to/save/image.png'）
    """
    if tensor.dim() != 4:
        raise ValueError(f"Tensor4 [B, C, H, W]，: {tensor.dim()}")
    
    if tensor.size(1) not in [1, 3]:
        raise ValueError(f"13，: {tensor.size(1)}")
    
    first_batch = tensor[0]
    save_image(first_batch, save_path)
    
    print(f": {save_path}")

class Diffusion_Planner(TorchModuleWrapper):
    def __init__(self, config,
                feature_builder: PlutoFeatureBuilder = PlutoFeatureBuilder(),):
        super().__init__(
            feature_builders=[feature_builder],
            target_builders=[EgoTrajectoryTargetBuilder(trajectory_sampling)],
            future_trajectory_sampling=trajectory_sampling,
        )

        backbone_config = TransfuserConfig()
        self.device = config.device
        #self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)
        self.image_backbone = TransfuserBackbone(backbone_config).to(self.device)
        self.img_transformer = ImageFeatureTransformer()

    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    def forward(self, inputs, sensor_image):
        #sensor_img:[B,N=3/8,3,H,W]
        sensor_image = sensor_image.to(self.device)
        sensor_image = F.interpolate(
            sensor_image.flatten(0, 1),  # [B*N, C, H, W]
            size=(540, 960),             
            mode='bilinear',
            align_corners=False
        ).unflatten(0, (sensor_image.size(0), sensor_image.size(1)))

        if sensor_image.size(1) == 6:
            forward_left = sensor_image[:, 1, :, :, :]
            forward_center = sensor_image[:, 0, :, :, :]
            forward_right = sensor_image[:, 2, :, :, :]
            backward_left = sensor_image[:, 1, :, :, :]
            backward_center = sensor_image[:, 0, :, :, :]
            backward_right = sensor_image[:, 2, :, :, :]
            sensor_image = torch.cat([forward_left, forward_center, forward_right, backward_left, backward_center, backward_right], dim=3)  # [B, C, H, W*3] shape=[1,3,540,2880]
        else:
            sensor_image = torch.cat(sensor_image.unbind(1), dim=3)
        
        #save_first_batch_as_image(sensor_image, "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/image.jpg")
            
        _, _, _, encoder_outputs = self.image_backbone(sensor_image, sensor_image) #[1,512,17,180]
        encoder_outputs = self.img_transformer(encoder_outputs) #[1,512,256]
        
        decoder_outputs = self.decoder(encoder_outputs, inputs)

        return encoder_outputs, decoder_outputs





class Diffusion_Planner_Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.encoder = Encoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize embedding MLP:
        nn.init.normal_(self.encoder.pos_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.neighbor_encoder.type_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.speed_limit_emb.weight, std=0.02)
        nn.init.normal_(self.encoder.lane_encoder.traffic_emb.weight, std=0.02)

    def forward(self, inputs):

        encoder_outputs = self.encoder(inputs)

        return encoder_outputs
    

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.decoder.dit.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.decoder.dit.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)

    def forward(self, encoder_outputs, inputs):

        decoder_outputs = self.decoder(encoder_outputs, inputs)
        
        return decoder_outputs
