import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from diffusion_planner.model.module.encoder import Encoder
from diffusion_planner.model.module.decoder import Decoder
import sys
#diffusiondrive2_path = "/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/Diffusion-Planner/DiffusionDrive"
#if diffusiondrive2_path not in sys.path:
#    sys.path.append(diffusiondrive2_path)
#from DiffusionDrive.navsim.agents.transfuser.img_transformer import  ImageFeatureTransformer
#from DiffusionDrive.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder
#from diffusion_planner.model.mix_mlp import TokenSemAligner
#from DiffusionDrive.navsim.agents.transfuser.transfuser_config import TransfuserConfig
#from DiffusionDrive.navsim.agents.transfuser.transfuser_config_raster import TransfuserConfig_Raster
#from DiffusionDrive.navsim.agents.transfuser.transfuser_backbone import TransfuserBackbone
import torch.nn.functional as F 
import os

from torchvision.utils import save_image
from PIL import Image
import numpy as np

'''

def save_first_batch_as_image(tensor, save_path):
    """
    将 [B, C, H, W] 维度的Tensor中的第一个batch保存为图片
    
    Args:
        tensor (torch.Tensor): 输入Tensor，形状为 [B, C, H, W]
        save_path (str): 图片保存路径（包含文件名和扩展名，如：'/path/to/save/image.png'）
    """
    # 检查输入Tensor的维度
    if tensor.dim() != 4:
        raise ValueError(f"输入Tensor应为4维 [B, C, H, W]，当前维度为: {tensor.dim()}")
    
    # 检查通道数（支持1通道灰度图或3通道RGB图）
    if tensor.size(1) not in [1, 3]:
        raise ValueError(f"通道数应为1或3，当前通道数为: {tensor.size(1)}")
    
    # 获取第一个batch
    first_batch = tensor[0]  # 形状变为 [C, H, W]
    
    # 确保目录存在
    #os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 保存图片
    save_image(first_batch, save_path)
    
    print(f"图片已保存至: {save_path}")

def save_as_rgb(tensor, save_path):
    """
    将三个通道合并为RGB图像并保存
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = tensor[0]  # [3, 256, 1024]
    
    # 转换为HWC格式 [256, 1024, 3]
    if torch.is_tensor(data):
        img = data.permute(1, 2, 0).cpu().detach().numpy()
    else:
        img = data.transpose(1, 2, 0)
    
    # 标准化到[0, 1]范围
    img_min = img.min()
    img_max = img.max()
    if img_max > img_min:
        img = (img - img_min) / (img_max - img_min)
    
    plt.figure(figsize=(12, 3))
    plt.imshow(img)
    plt.title('RGB Visualization')
    plt.axis('off')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
    print(f"RGB图像已保存到: {save_path}")

class Diffusion_Planner(nn.Module):
    def __init__(self, config):
        super().__init__()

        backbone_config = TransfuserConfig()
        self.device = config.device 
        #self.encoder_dp = Diffusion_Planner_Encoder(config).to(self.device)
        self.decoder = Diffusion_Planner_Decoder(config).to(self.device)
        self.image_backbone = TransfuserBackbone(backbone_config).to(self.device)
      
        self.img_transformer = ImageFeatureTransformer()
        self.projector = TokenSemAligner()
        

    @property
    def sde(self):
        return self.decoder.decoder.sde
    
    def forward(self, inputs, sensor_image, lidar_bev, lidar_mask):
        
       
        #encoder_outputs_dp = self.encoder_dp(inputs) #[B,107,192]

        #bev_feature_upscale, bev_feature, _ = self.image_backbone(sensor_image, lidar_points) #[1,256,16,16] 
        bev_feature = self.image_backbone(sensor_image, lidar_bev) #[B,256,16,16] 
        
        bev_feature = self.img_transformer(bev_feature).contiguous() #[B,256,192]
        encoder_outputs_prediceted = self.projector(bev_feature) #[B,107,192]
        decoder_outputs = self.decoder(encoder_outputs_prediceted, inputs, lidar_mask)

        return  encoder_outputs_prediceted, decoder_outputs

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
'''

class Diffusion_Planner_Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.decoder = Decoder(config)
        self.initialize_weights()

    def initialize_weights(self):
    # Initialize transformer layers:
        def _basic_init(m):
            #if hasattr(m, "bias") and m.bias is None:
            #    print("bias is None in:", type(m))
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
            #nn.init.normal_(block.adaLN_modulation[-1].weight, std=1e-4)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0) 

        # Zero-out output layers:
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].weight, 0)
        nn.init.constant_(self.decoder.dit.final_layer.proj[-1].bias, 0)
        #nn.init.normal_(self.decoder.dit.final_layer.proj[-1].weight, std=1e-4)
        #nn.init.zeros_(self.decoder.dit.final_layer.proj[-1].bias)
        #nn.init.normal_(self.decoder.dit.final_layer.adaLN_modulation[-1].weight, std=1e-4)
        #nn.init.zeros_(self.decoder.dit.final_layer.adaLN_modulation[-1].bias)

    def forward(self, noise, image_feature, status_encoding, t):
        
        x0 = self.decoder(noise, image_feature, status_encoding, t)
        
        return x0
