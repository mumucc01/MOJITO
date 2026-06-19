
import torch
import torch.nn as nn

from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
from DiffusionDrive.navsim.agents.transfuser.img_transformer import ImageFeatureTransformer, ImageFeatureTransformer_Raw
from DiffusionDrive.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder

from DiffusionDrive.navsim.agents.transfuser.transfuser_config import TransfuserConfig
from DiffusionDrive.navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
import torch.nn.functional as F 
import os

from torchvision.utils import save_image
from PIL import Image
import numpy as np

def save_first_batch_as_image(tensor, save_path):
    """
    将 [B, C, H, W] 维度的Tensor中的第一个batch保存为图片
    
    Args:
        tensor (torch.Tensor): 输入Tensor，形状为 [B, C, H, W]
        save_path (str): 图片保存路径（包含文件名和扩展名，如：'/path/to/save/image.png'）
    """
    if tensor.dim() != 4:
        raise ValueError(f"输入Tensor应为4维 [B, C, H, W]，当前维度为: {tensor.dim()}")
    
    if tensor.size(1) not in [1, 3]:
        raise ValueError(f"通道数应为1或3，当前通道数为: {tensor.size(1)}")
    
    first_batch = tensor[0]
    
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_image(first_batch, save_path)
    
    print(f"图片已保存至: {save_path}")

class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()

        backbone_config = TransfuserConfig()
        self.device = config.device 
        #self.encoder = Diffusion_Planner_Encoder(config)
        self.decoder = Diffusion_Planner_Decoder(config)
        self.image_backbone = TransfuserBackbone(backbone_config).to(self.device)
        #self.feature_processor = TransfuserFeatureBuilder(backbone_config)
        #self.img_transformer = ImageFeatureTransformer(query_len = config.query_len, d_model = config.hidden_dim)
        self.img_transformer = ImageFeatureTransformer_Raw()

    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    def forward(self, inputs, sensor_image, lidar_points):
        #sensor_image: [B,3,256, 1024] tensor
        #lidar_points: [B, 1, 256, 256] tensor

        #sensor_lidar_processed = self.feature_processor._get_lidar_feature_batch(lidar_points).to(self.device) #torch.Size([B, 1, 256, 256])
        #sensor_image_processed = self.feature_processor._get_camera_feature_batch(sensor_image).to(self.device) #torch.Size([B, 3, 256, 1024])
      
        

        bev_feature_upscale, bev_feature, _ = self.image_backbone(sensor_image, lidar_points) #[1,512,8,8]
        encoder_outputs = self.img_transformer(bev_feature) #[1,512,d_model]
       
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
