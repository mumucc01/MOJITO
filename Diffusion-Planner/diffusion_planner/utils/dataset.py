import os
from torch.utils.data import Dataset
import numpy as np
from diffusion_planner.utils.train_utils import openjson, opendata
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import torch
import os
import time 
import torch.nn.functional as F
from multiprocessing import Pool
from functools import partial
from diffusion_planner.model.lidar_mask import Lidar_Mask
from DiffusionDrive.navsim.agents.transfuser.transfuser_config import TransfuserConfig
from DiffusionDrive.navsim.agents.transfuser.transfuser_features import TransfuserFeatureBuilder

from torchvision.transforms.functional import resize as torch_resize
from torchvision.io import read_image
from turbojpeg import TurboJPEG
import concurrent.futures


def visualize_noise_grid_on_lidar(
    lidar_feature: torch.Tensor,
    noise: torch.Tensor,
    x_min: float,
    y_min: float,
    ppm: float,
    save_path: str,
):
    """
    lidar_feature: [1,H,W] 或 [H,W]，0/1，可行区域 mask
    noise: [B, P, T, 2]，这里你是 [1,1,T,2]
    x_min, y_min, ppm: 必须和 Lidar_CZJ.get_feasible_voxel_feature 里的配置一致
    save_path: 保存 PNG 的路径
    """
    if lidar_feature.dim() == 3:
        lidar_feature = lidar_feature.squeeze(0)
    assert lidar_feature.dim() == 2, f"lidar_feature 应该是 [H,W]，现在是 {lidar_feature.shape}"

    device = lidar_feature.device
    H, W = lidar_feature.shape

    pts_noise = noise[0, 0]  # [T,2]

    mean_xy = torch.tensor([10.0, 0.0], device=device)
    std_xy  = torch.tensor([20.0, 20.0], device=device)

    #pts_meter = pts_noise * std_xy + mean_xy  # [T,2]
    pts_meter = pts_noise
    x = pts_meter[:, 0]
    y = pts_meter[:, 1]

    ix = ((x - x_min) * ppm).long()
    iy = ((y - y_min) * ppm).long()

    in_bounds = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix = ix[in_bounds]
    iy = iy[in_bounds]

    bg = lidar_feature.detach().cpu().numpy().astype(np.float32)   # [H,W]
    bg_img = (bg * 255).astype(np.uint8)                           # [H,W]

    rgb = np.stack([bg_img, bg_img, bg_img], axis=-1)              # [H,W,3]

    ix_np = ix.cpu().numpy()
    iy_np = iy.cpu().numpy()
    rgb[iy_np, ix_np, 0] = 255  # R
    rgb[iy_np, ix_np, 1] = 0    # G
    rgb[iy_np, ix_np, 2] = 0    # B

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb, origin='lower')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved noise visualization to: {save_path}")



jpeg_decoder = TurboJPEG()


def fast_read_jpeg_numpy(path):
    with open(path, 'rb') as f:
        jpeg_bytes = f.read()
    img = jpeg_decoder.decode(jpeg_bytes)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_single_image_numpy(sensor_path, rel_path):
    full_path = os.path.join(sensor_path, rel_path)
    return fast_read_jpeg_numpy(full_path)

def batch_read_images_parallel_numpy(sensor_path, path_array, executor):
    B, V = path_array.shape
    imgs = [[None]*V for _ in range(B)]
    futures = {}
    
    for i in range(B):
        for j in range(V):
            futures[executor.submit(load_single_image_numpy, sensor_path, path_array[i,j])] = (i,j)

    for fut in concurrent.futures.as_completed(futures):
        i,j = futures[fut]
        imgs[i][j] = fut.result()
    
    imgs_array = np.array(imgs)
    
    return imgs_array

class DiffusionPlannerData(Dataset):
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        backbone_config = TransfuserConfig()
        self.feature_processor = TransfuserFeatureBuilder(backbone_config)
        self.lidar_mask = Lidar_Mask()
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
        # 'map_name':'us-nv-las-vegas-strip' \ 'token.npy' : '306767aa721651c5'
        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        lanes = data['lanes'] #[B,70,20,12]
        lanes_speed_limit = data['lanes_speed_limit'] #(70,1)
        lanes_has_speed_limit = data['lanes_has_speed_limit'] #(70,1)

        route_lanes = data['route_lanes'] #[25,20,12]
        route_lanes_speed_limit = data['route_lanes_speed_limit'] #[25,1]
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']  #[25,1]

        static_objects = data['static_objects'] #[5,10]
        sensor_image_path = data['sensor_image_path'] 
        lidar_points = data["lidar_points"][:3,:]
        
        mask_ego_region = (lidar_points[0] >= -0.25) & (lidar_points[0] <= 0.25) & \
            (lidar_points[1] >= -0.25) & (lidar_points[1] <= 0.25)

        lidar_points[2][mask_ego_region] = 0
        
        camera_keys = ['CAM_F0', 'CAM_L0', 'CAM_R0']

        try:
            sensor_image_path = [
                [frame_dict[cam] for cam in camera_keys]
                for frame_dict in sensor_image_path
            ]
        except KeyError as e:
            print(f"[Warning] Missing camera key: {e} at index {idx}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self)) 
        # F0,L0,R0,L1,R1,L2,R2,B0
        sensor_image_path_array = np.array(sensor_image_path, dtype=str) #(1,3)
        #sensor_image_path_array = sensor_image_path_array[-1:, [0, 1, 2, 5, 6, 7]] #[1,6]
        #sensor_image_path_array = sensor_image_path_array[0, [0, 1, 2]]
        sensor_root = '/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/sensor_imgs'
        #sensor_root = '/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/sensor_blobs'
        sensor_image_path_array = batch_read_images_parallel_numpy(sensor_root, sensor_image_path_array, self.executor) #[1,3,3,1080,1920]
        sensor_image_path_array = sensor_image_path_array.squeeze() #[3,3,1080,1920]
        
        lidar_bev = self.feature_processor._get_lidar_feature(lidar_points)
        lidar_points = torch.from_numpy(lidar_points).float()
        lidar_mask = self.lidar_mask.get_feasible_voxel_feature(lidar_points)
        
        #visualize_noise_grid_on_lidar(
        #    lidar_mask,
        #    torch.zeros(1, 1, 80, 2),
        #    -50,
        #    -50,
        #    4,
        #   "/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/AD-Eccv/DiffusionPlanner_v37/color_noise2.png"
        #)
        sensor_image_processed = self.feature_processor._get_camera_feature(sensor_image_path_array)
      
        data = {
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
            "sensor_image" : sensor_image_processed,
            "lidar_bev": lidar_bev,
            "lidar_mask": lidar_mask
        }

        return tuple(data.values())