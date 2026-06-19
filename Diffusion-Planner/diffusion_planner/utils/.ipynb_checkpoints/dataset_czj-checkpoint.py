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

def fast_read_jpeg_numpy(path):
    with open(path, 'rb') as f:
        jpeg_bytes = f.read()
    img = jpeg_decoder.decode(jpeg_bytes)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img 

data = opendata("/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/nuplan-dataset-img-lidar-current-100w/npz_data/sg-one-north_0003bf21e7b756a9.npz")
lidar_points = data["lidar_points"]
print(type(lidar_points))
print(lidar_points.shape)

test_img_path = "/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/image.jpg"
ours= np.array(fast_read_jpeg_numpy(test_img_path))[:,:,1]
raw = np.array(Image.open(test_img_path))[:,:,1]
print(f"ours shape is {ours.shape}, raw shape is {raw.shape}")
print(ours)
print("************")
print(raw)