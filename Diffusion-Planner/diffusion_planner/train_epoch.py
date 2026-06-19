from tqdm import tqdm
import torch
from torch import nn

from diffusion_planner.utils.data_augmentation import StatePerturbation   
from diffusion_planner.utils.train_utils import get_epoch_mean_loss
from diffusion_planner.utils import ddp
from diffusion_planner.loss import diffusion_loss_func


def train_epoch(data_loader, model, optimizer, args, ema, epoch, aug: StatePerturbation=None):
    
    epoch_loss = []

    model.train()
    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            '''
            data structure in batch: Tuple(Tensor) 

            ego_current_state,
            ego_future_gt,

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            '''

            # prepare data
            inputs = {
                'ego_current_state': batch[0].to(args.device), #[2048,80,dim]

                'neighbor_agents_past': batch[2].to(args.device),
                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),

                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),

                'static_objects': batch[10].to(args.device),
                'senor_image': batch[11].to(args.device).to(torch.float32), #[B,T=1,3,3,1080,1920]
                'lidar_bev': batch[12].to(args.device),
                "lidar_mask" : batch[13].to(args.device)
            }

            ego_future = batch[1].to(args.device)
            neighbors_future = batch[3].to(args.device)
            # Normalize to ego-centric
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            ego_future = torch.cat( #[B,80,4]
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = torch.cat( #[B,0,80,4]
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
            neighbors_future[mask] = 0.
            inputs = args.observation_normalizer(inputs)
           
            # call the mdoel
            optimizer.zero_grad()
            loss = {}
            #/DiffusionPlanner/Diffusion-Planner/diffusion_planner/loss.py P8
            loss, _ = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                args.state_normalizer,
                loss,
                args.diffusion_model_type
            )

        
            loss['loss'] =   loss['neighbor_prediction_loss'] +  args.alpha_planning_loss * loss['ego_planning_loss']
            total_loss = loss['loss'].item()

            # loss backward
            loss['loss'].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()
            
           
            data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))

            epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    return epoch_mean_loss, epoch_mean_loss['loss']