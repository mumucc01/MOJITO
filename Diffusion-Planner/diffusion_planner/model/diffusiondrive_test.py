import torch
import torch.nn as nn
import sys
sys.path.append("/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v36/Diffusion-Planner/DiffusionDrive")

import numpy.typing as npt
import pytest
from matplotlib import axes, cm
from PIL import Image
from pyquaternion import Quaternion
import imageio
from typing import IO, Any, ByteString, Dict, List, Optional, Tuple, Union

from DiffusionDrive.navsim.agents.transfuser.transfuser_config import TransfuserConfig

from DiffusionDrive.navsim.agents.transfuser.transfuser_model_test import TransfuserModel
import torch.nn.functional as F 
import os

from torchvision.utils import save_image
from PIL import Image
import numpy as np

from nuplan.database.utils.geometry import view_points


def render_image(
        points,
        canvas_size: Tuple[int, int] = (1001, 1001),
        view: npt.NDArray[np.float64] = np.array([[10, 0, 0, 500], [0, 10, 0, 500], [0, 0, 10, 0]]),
        color_dim: int = 2,
    ) -> Image.Image:
        """
        Renders pointcloud to an array with 3 channels appropriate for viewing as an image. The image is color coded
        according the color_dim dimension of points (typically the height).
        :param canvas_size: (width, height). Size of the canvas on which to render the image.
        :param view: <np.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param color_dim: The dimension of the points to be visualized as color. Default is 2 for height.
        :return: A Image instance.
        """
        # Apply desired transformation to the point cloud. (height is here considered independent of the view).
        heights = points[2, :]
        points = view_points(points[:3, :], view, normalize=False)
        points[2, :] = heights

        mask = np.ones(points.shape[1], dtype=bool)  # type: ignore
        mask = np.logical_and(mask, points[0, :] < canvas_size[0] - 1)  #canvas_size:tuple (1000,1000)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[1, :] < canvas_size[1] - 1)
        mask = np.logical_and(mask, points[1, :] > 0)
        points = points[:, mask]

        # Scale color_values to be between 0 and 255.
        color_values = points[color_dim, :]
        color_values = 255.0 * (color_values - np.amin(color_values)) / (np.amax(color_values) - np.amin(color_values))

        # Rounds to ints and generate colors that will be used in the image.
        points = np.int16(np.round(points[:2, :]))
        color_values = np.int16(np.round(color_values))
        cmap = [cm.jet(i / 255, bytes=True)[:3] for i in range(256)]

        # Populate canvas, use maximum color_value for each bin
        render = np.tile(np.expand_dims(np.zeros(canvas_size, dtype=np.uint8), axis=2), [1, 1, 3])  # type: ignore
        color_value_array: npt.NDArray[np.float64] = -1 * np.ones(canvas_size, dtype=float)  # type: ignore [1001,1001,3]
        for (col, row), color_value in zip(points.T, color_values.T):
            if color_value > color_value_array[row, col]:
                color_value_array[row, col] = color_value
                render[row, col] = cmap[color_value]

        return Image.fromarray(render)



def height_distribution(heights, bins=10):
    heights = np.array(heights)

    h_min = heights.min()
    h_max = heights.max()

    bin_edges = np.linspace(h_min, h_max, bins + 1)

    counts, _ = np.histogram(heights, bins=bin_edges)

    percentages = counts / counts.sum() * 100

    print(f"Height range: min={h_min:.4f}, max={h_max:.4f}")
    print("----- Height Distribution (10 bins) -----")
    for i in range(bins):
        print(f"Bin {i+1}: [{bin_edges[i]:.4f}, {bin_edges[i+1]:.4f}] "
              f"→ {counts[i]} points ({percentages[i]:.2f}%)")

    return bin_edges, counts, percentages

def count_in_range(values, low, high):
    """
    values: 一维 numpy 数组或 torch tensor
    low: 区间下界
    high: 区间上界（包含 high）
    返回：数量、占比
    """
    if not isinstance(values, np.ndarray):
        values = values.cpu().numpy()

    mask = (values >= low) & (values <= high)
    count = mask.sum()

    percentage = count / len(values) * 100

    print(f"Range [{low}, {high}] → {count} points ({percentage:.2f}%)")

    return count, percentage

def save_gray_image(tensor, save_path="gray.png"):
    tensor = tensor.squeeze(0).squeeze(0)

    img = tensor.cpu().numpy()

    if img.dtype != np.uint8:
        img = img - img.min()
        if img.max() > 0:
            img = img / img.max()
        img = (img * 255).astype(np.uint8)

    imageio.imwrite(save_path, img)
    print("Saved:", save_path)



def save_tensor_as_pngs(tensor, save_dir="output_png"):
    os.makedirs(save_dir, exist_ok=True)

    tensor = tensor.squeeze(0)

    for i in range(tensor.shape[0]):
        channel = tensor[i]  # [128, 256]

        arr = channel.cpu().detach().numpy()

        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).astype(np.uint8)

        save_path = os.path.join(save_dir, f"channel_{i}.png")
        imageio.imwrite(save_path, arr)

        print(f"Saved: {save_path}")


device = 'cuda'

#path = "/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v36/scene-init/scene3/lidar_point.pt"

#data = torch.load(path)



#lidar = data["lidar_points"] 


lidar_point ="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v36/scene-init/scene3/lidar_point.pt"
lidar_point = torch.load(lidar_point, weights_only=False)
print(f"1 shape is {lidar_point.shape}")
lidar_point_process = render_image(lidar_point)

lidar_array = np.array(lidar_point_process)
print(f"3 shape is {lidar_array.shape}")
Lidar_image_white = Image.fromarray(lidar_array)
Lidar_image_white.save('/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v36/scene-init/scene3/pointcloud_image.png')
count_in_range(lidar_point[2, :],-10,0.2)


#print(lidar_point.shape)    

#save_gray_image(lidar_array,'/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v36/scene-init/scene2/lidar.jpg')
''' 
features = {'camera_feature': sensor, 'lidar_feature': lidar}
backbone_config = TransfuserConfig()
       
My_TransfuserModel = TransfuserModel(backbone_config).to("cuda") 

image_backbone_ckpt = '/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS' 

backbone_state_dict = torch.load(image_backbone_ckpt, map_location=device)['state_dict']  # 先加载到CPU

# 键名转换
backbone_state_dict = {
    k.replace("agent._transfuser_model.", ""): v
    for k, v in backbone_state_dict.items()
    if "agent._transfuser_model." in k  # 只处理相关键
}

# 加载状态字典
missing_keys, unexpected_keys = My_TransfuserModel.load_state_dict(backbone_state_dict, strict=False)

# 打印加载结果
if missing_keys:
    print(f"Missing keys in image_backbone: {missing_keys}")
if unexpected_keys:
    print(f"Unexpected keys in image_backbone: {unexpected_keys}")
      
    

bev_semantic_map = My_TransfuserModel(features)

print(f"维度是{bev_semantic_map.shape}")
save_tensor_as_pngs(bev_semantic_map,'/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v36/Diffusion-Planner/channels_bev_map')

'''