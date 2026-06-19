import torch
import random
import numpy as np
#from mmengine import fileio

import io
import os
import json

def openjson(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def opendata(path):
    with open(path, 'rb') as file:
        npz_data = np.load(file, allow_pickle=True)
    return npz_data

def set_seed(CUR_SEED):
    random.seed(CUR_SEED)
    np.random.seed(CUR_SEED)
    torch.manual_seed(CUR_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_epoch_mean_loss(epoch_loss):
    epoch_mean_loss = {}
    for current_loss in epoch_loss:
        for key, value in current_loss.items():
            if key in epoch_mean_loss:
                epoch_mean_loss[key].append(value if isinstance(value, (int, float)) else value.item())
            else:
                epoch_mean_loss[key] = [value if isinstance(value, (int, float)) else value.item()]


    for key, values in epoch_mean_loss.items():
        epoch_mean_loss[key] = np.mean(np.array(values))

    return epoch_mean_loss

def save_model(model, optimizer, scheduler, save_path, epoch, train_loss, wandb_id, ema):
    """
    """
    save_model = {'epoch': epoch + 1, 
                  'model': model.state_dict(), 
                  'ema_state_dict': ema.state_dict(),
                  'optimizer': optimizer.state_dict(), 
                  'schedule': scheduler.state_dict(), 
                  'loss': train_loss,
                  'wandb_id': wandb_id}

    with io.BytesIO() as f:
        torch.save(save_model, f)
        fileio.put(f.getvalue(), f'{save_path}/model_epoch_{epoch+1}_trainloss_{train_loss:.4f}.pth')
        fileio.put(f.getvalue(), f"{save_path}/latest.pth")

def resume_model(path: str, model, optimizer, scheduler, ema, device):
    """
    """
    #path = os.path.join(path, 'model.pth')
    ckpt = fileio.get(path)
    with io.BytesIO(ckpt) as f:
        ckpt = torch.load(f)

    # load model
    try:
        model.load_state_dict(ckpt['model'],strict=False)
    except:
        model.load_state_dict(ckpt, strict=False)       
    print("Model load done")
    
    need_load_external = True
    if 'model' in ckpt:
        has_internal_backbone = any(k.startswith('module.image_backbone') 
                            for k in ckpt['model'].keys())
    if has_internal_backbone:
        print("Found image_backbone weights in checkpoint, skipping external loading")
        need_load_external = False

    if need_load_external:
        device = next(model.parameters()).device
        image_backbone = model.module.image_backbone
        image_backbone_ckpt = '/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/AD-Eccv/DiffusionPlanner_v37/Diffusion-Planner/DiffusionDrive/diffusiondrive_navsim_88p1_PDMS'
        print("Loading image_backbone weights from:", image_backbone_ckpt)

        try:
            backbone_state_dict = torch.load(image_backbone_ckpt, map_location=device)['state_dict']
            
            backbone_state_dict = {
                k.replace("agent._transfuser_model._backbone.", ""): v
                for k, v in backbone_state_dict.items()
                if "agent._transfuser_model._backbone." in k
            }
            
            missing_keys, unexpected_keys = image_backbone.load_state_dict(backbone_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in image_backbone: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in image_backbone: {unexpected_keys}")
        except Exception as e:
            print(f"Failed to load image_backbone weights: {str(e)}")
    
    # load optimizer
    try:
        optimizer.load_state_dict(ckpt['optimizer'])
        print("Optimizer load done")
    except:
        print("no pretrained optimizer found")
            
    # load schedule
    try:
        scheduler.load_state_dict(ckpt['schedule'])
        print("Schedule load done")
    except:
        print("no schedule found,")
    
    # load step
    try:
        init_epoch = ckpt['epoch']
        print("Step load done")
    except:
        init_epoch = 0
   
    # Load wandb id
    try:
        wandb_id = ckpt['wandb_id']
        print("wandb id load done")
    except:
        wandb_id = None
   

    try:
        #ema.ema.load_state_dict(ckpt['ema_state_dict'],strict=False)
        ema.ema.load_state_dict(ckpt['ema_state_dict'])
        ema.ema.eval()
        for p in ema.ema.parameters():
            p.requires_grad_(False)

        print("ema load done")
    except:
        print('no ema shadow found')
    
    return model, optimizer, scheduler, init_epoch, wandb_id, ema


