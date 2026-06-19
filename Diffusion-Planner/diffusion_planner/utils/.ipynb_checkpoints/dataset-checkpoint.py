import os
from torch.utils.data import Dataset
import numpy as np
from diffusion_planner.utils.train_utils import openjson, opendata
import torchvision.transforms.functional as TF
from concurrent.futures import ThreadPoolExecutor
import cv2
from torchvision import transforms
from PIL import Image
import torch
import os
import time 
import torch.nn.functional as F
from multiprocessing import Pool
from functools import partial

from torchvision.transforms.functional import resize as torch_resize
from torchvision.io import read_image
from turbojpeg import TurboJPEG
import concurrent.futures

jpeg_decoder = TurboJPEG()


def fast_read_jpeg(path):
    with open(path, 'rb') as f:
        jpeg_bytes = f.read()
    img = jpeg_decoder.decode(jpeg_bytes)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # CxHxW
    return img_tensor
def load_single_image(sensor_path, rel_path):
    full_path = os.path.join(sensor_path, rel_path)
    return fast_read_jpeg(full_path)

def batch_read_images_parallel(sensor_path, path_array,executor):
    B, V = path_array.shape
    imgs = [[None]*V for _ in range(B)]
    futures = {}
    '''
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        futures = {}
        for i in range(B):
            for j in range(V):
                futures[executor.submit(load_single_image, sensor_path, path_array[i,j])] = (i,j)

        for fut in concurrent.futures.as_completed(futures):
            i,j = futures[fut]
            imgs[i][j] = fut.result()
    '''
    for i in range(B):
        for j in range(V):
            futures[executor.submit(load_single_image, sensor_path, path_array[i,j])] = (i,j)

    for fut in concurrent.futures.as_completed(futures):
        i,j = futures[fut]
        imgs[i][j] = fut.result()
    for i in range(B):
        imgs[i] = torch.stack(imgs[i], dim=0)  # V x C x H x W
    imgs_tensor = torch.stack(imgs, dim=0)  # B x V x C x H x W
    return imgs_tensor

class DiffusionPlannerData(Dataset):
    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))
        # 'map_name':'us-nv-las-vegas-strip' \ 'token.npy' : '306767aa721651c5'
        ego_current_state = data['ego_current_state']
        ego_agent_future = data['ego_agent_future']

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num] #[32,21,11]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num] #[10,80,3]

        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']
        sensor_image_path = data['sensor_image_path']

        camera_keys = ['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_L1', 'CAM_R1', 'CAM_L2', 'CAM_R2', 'CAM_B0']

        try:
            sensor_image_path = [
                [frame_dict[cam] for cam in camera_keys]
                for frame_dict in sensor_image_path
            ]
        except KeyError as e:
            print(f"[Warning] Missing camera key: {e} at index {idx}. Skipping sample.")
            return self.__getitem__((idx + 1) % len(self)) 
        # F0,L0,R0,L1,R1,L2,R2,B0
        sensor_image_path_array = np.array(sensor_image_path, dtype=str)
        #sensor_image_path_array = sensor_image_path_array[-1:, [0, 1, 2, 5, 6, 7]] #[1,6]
        sensor_image_path_array = sensor_image_path_array[-1:, [0, 1, 2]]
        sensor_root = '/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-past-current-trainval/nuplan-v1.1/sensor_blobs'
        #sensor_root = '/lpai/volumes/base-3da-ali-sh-mix/zhangxc/DiffusionPlanner/nuplan_dataset/nuplan-v1.1/sensor_blobs'
        sensor_image_path_array = batch_read_images_parallel(sensor_root, sensor_image_path_array, self.executor) #[1,6,3,1080,1920]
        sensor_image_path_array = sensor_image_path_array.squeeze(0)
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
            "sensor_image" : sensor_image_path_array
        }

        return tuple(data.values())