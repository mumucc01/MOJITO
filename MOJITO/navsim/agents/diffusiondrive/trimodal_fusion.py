import sys
import os
import os.path as osp
import math
import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
import logging

sys.path.insert(0, osp.abspath(osp.join(osp.dirname(__file__), "dinov3")))
import hubconf
from navsim.agents.diffusiondrive.modules.dinov3_adapter import DINOv3_Adapter
from navsim.agents.diffusiondrive.modules.dinov3_backbone import DINOv3Backbone
from navsim.agents.diffusiondrive.transfuser_config import TransfuserConfig
from navsim.mojito_paths import pretrained_path



    #return 



class Trimodal_Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        weights = pretrained_path("dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth")
        #    weights_path=weights, 
        #    patch_size=16,
        #    out_channels=(512, 512, 512, 512)
        #)

        self.lidar_dinov3 = DINOv3Backbone(
            weights_path=weights, 
            patch_size=16,
            out_channels=(512, 512, 512, 512)
        )
       

    def forward(self,layer_idx: int):
        return 
    
    def process_joint_attention(self, lidar_feature, layer_idx): 
        #img_token = self.img_dinov3(image_feature, layer_idx) #[1037,379]
        lidar_token = self.lidar_dinov3(lidar_feature, layer_idx) #[261,379]

        return  lidar_token
    
    def pre_process_img_lidar(self,img_feature,lidar_feature):
        return 

    