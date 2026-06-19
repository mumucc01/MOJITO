import os
import torch
import argparse
from torch import optim
from timm.utils import ModelEma
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from diffusion_planner.model.diffusion_planner import Diffusion_Planner

from diffusion_planner.utils.train_utils import set_seed, save_model, resume_model
from diffusion_planner.utils.normalizer import ObservationNormalizer, StateNormalizer
from diffusion_planner.utils.lr_schedule import CosineAnnealingWarmUpRestarts
from diffusion_planner.utils.tb_log import TensorBoardLogger as Logger
from diffusion_planner.utils.data_augmentation import StatePerturbation
from diffusion_planner.utils.dataset import DiffusionPlannerData
from diffusion_planner.utils import ddp

from diffusion_planner.train_epoch import train_epoch


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name (default: "diffusion-planner-training")', default="v29")
    parser.add_argument('--save_dir', type=str, help='save dir for model ckpt', default="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/Diffusion-Planner/training_log")

    # Data
    parser.add_argument('--train_set', type=str, help='path to train data', default=None)
    parser.add_argument('--train_set_list', type=str, help='data list of train data', default=None)

    parser.add_argument('--future_len', type=int, help='number of time point', default=80)
    parser.add_argument('--time_len', type=int, help='number of time point', default=21)

    parser.add_argument('--agent_state_dim', type=int, help='past state dim for agents', default=11)
    parser.add_argument('--agent_num', type=int, help='number of agents', default=32)

    parser.add_argument('--static_objects_state_dim', type=int, help='state dim for static objects', default=10)
    parser.add_argument('--static_objects_num', type=int, help='number of static objects', default=5)

    parser.add_argument('--lane_len', type=int, help='number of lane point', default=20)
    parser.add_argument('--lane_state_dim', type=int, help='state dim for lane point', default=12)
    parser.add_argument('--lane_num', type=int, help='number of lanes', default=70)

    parser.add_argument('--route_len', type=int, help='number of route lane point', default=20)
    parser.add_argument('--route_state_dim', type=int, help='state dim for route lane point', default=12)
    parser.add_argument('--route_num', type=int, help='number of route lanes', default=25)
    
    # DataLoader parameters
    parser.add_argument('--augment_prob', type=float, help='augmentation probability', default=0.5)
    parser.add_argument('--normalization_file_path', default='/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_v10/Diffusion-Planner/normalization.json', help='filepath of normalizaiton.json', type=str)
    parser.add_argument('--use_data_augment', default=True, type=boolean)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin-mem', action='store_true', help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem', help='')
    parser.set_defaults(pin_mem=True)
    
    # Training
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=600)
    parser.add_argument('--save_utd', type=int, help='save frequency', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size (default: 2048)', default=11000)
    #parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    parser.add_argument('--learning_rate', type=float, help='learning rate (default: 5e-4)', default=5e-4)
    parser.add_argument('--warm_up_epoch', type=int, help='number of warm up', default=5)
    parser.add_argument('--encoder_drop_path_rate', type=float, help='encoder drop out rate', default=0.1)
    parser.add_argument('--decoder_drop_path_rate', type=float, help='decoder drop out rate', default=0.1)

    parser.add_argument('--alpha_planning_loss', type=float, help='coefficient of planning loss (default: 1.0)', default=1.0)
    parser.add_argument('--query_len', type=float, help='scene_query_len', default=107) 
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')

    parser.add_argument('--use_ema', default=True, type=boolean)
    # Model
    parser.add_argument('--encoder_depth', type=int, help='number of encoding layers', default=3)
    parser.add_argument('--decoder_depth', type=int, help='number of decoding layers', default=3)
    parser.add_argument('--num_heads', type=int, help='number of multi-head', default=8)
    parser.add_argument('--hidden_dim', type=int, help='hidden dimension', default=192)
    parser.add_argument('--diffusion_model_type', type=str, help='type of diffusion model [x_start, score]', choices=['score', 'x_start'], default='x_start')

    # decoder
    parser.add_argument('--predicted_neighbor_num', type=int, help='number of neighbor agents to predict', default=0)
    parser.add_argument('--resume_model_path', type=str, help='path to resume model', default="/mnt/volumes/base-3da-ali-sh-mix/chengzhijing/DiffusionPlanner_tem_czj/Diffusion-Planner/checkpoints/model.pth")

    parser.add_argument('--use_wandb', default=False, type=boolean)
    parser.add_argument('--notes', default='', type=str)

    # distributed training parameters
    parser.add_argument('--ddp', default=True, type=boolean, help='use ddp or not')
    parser.add_argument('--port', default='22324', type=str, help='port')

    args = parser.parse_args()

    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    
    return args

def model_training(args):

    # init ddp
    # global_rank, rank, _ = ddp.ddp_setup_universal(True, args)
    #local_rank = int(os.environ["LOCAL_RANK"])
    if args.ddp:
        dist.init_process_group(backend='nccl')
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        #print(f"world size is {world_size}")
    else:
        global_rank, local_rank, world_size = 0, 0, 1

    if global_rank == 0:
        # Logging
        print("------------- {} -------------".format(args.name))
        print("Batch size: {}".format(args.batch_size))
        print("Learning rate: {}".format(args.learning_rate))
        print("Use device: {}".format(args.device))

        if args.resume_model_path is not None:
            from datetime import datetime
            time = datetime.now()
            time = time.strftime("%Y-%m-%d-%H:%M:%S")

            save_path = f"{args.save_dir}/{args.name}/{time}/"
            os.makedirs(save_path, exist_ok=True)
            #save_path = args.resume_model_path
        else:
            from datetime import datetime
            time = datetime.now()
            time = time.strftime("%Y-%m-%d-%H:%M:%S")

            save_path = f"{args.save_dir}/{args.name}/{time}/"
            os.makedirs(save_path, exist_ok=True)

        # Save args
        args_dict = vars(args)
        args_dict = {k: v if not isinstance(v, (StateNormalizer, ObservationNormalizer)) else v.to_dict() for k, v in args_dict.items() }

        from mmengine.fileio import dump
        dump(args_dict, os.path.join(save_path, 'args.json'), file_format='json', indent=4)
    else:
        save_path = None

    # set seed
    set_seed(args.seed + global_rank)

    # training parameters
    train_epochs = args.train_epochs
    batch_size = args.batch_size
    
    # set up data loaders #args.agent_num:32  args.predicted_neighbor_num:10  args.future_len:80
    aug = StatePerturbation(augment_prob=args.augment_prob, device=args.device) if args.use_data_augment else None
    train_set = DiffusionPlannerData(args.train_set, args.train_set_list, args.agent_num, args.predicted_neighbor_num, args.future_len)
    train_sampler = DistributedSampler(train_set, num_replicas=ddp.get_world_size(), rank=global_rank, shuffle=True)
    train_loader = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size//ddp.get_world_size(), num_workers=12, pin_memory=args.pin_mem, drop_last=True)
   
    if global_rank == 0:
        print("Dataset Prepared: {} train data\n".format(len(train_set)))

    if args.ddp:
        torch.distributed.barrier()

    # set up model
    diffusion_planner = Diffusion_Planner(args)
    diffusion_planner = diffusion_planner.to(local_rank if args.device == 'cuda' else args.device)

    if args.ddp:
        #diffusion_planner = DDP(diffusion_planner, device_ids=[rank])
        diffusion_planner = DDP(
            diffusion_planner,
            device_ids=[local_rank],
            find_unused_parameters=True
        )

    if args.use_ema:
        model_ema = ModelEma(
            diffusion_planner,
            decay=0.999,
            device=args.device,
        )
    
   

    if global_rank == 0:
        print("Model Params: {}".format(sum(p.numel() for p in ddp.get_model(diffusion_planner, args.ddp).parameters())))

    # optimizer
    #params = [{'params': ddp.get_model(diffusion_planner, args.ddp).parameters(), 'lr': args.learning_rate}]

    #optimizer = optim.AdamW(params)
    image_backbone = diffusion_planner.module.image_backbone
    for param in image_backbone.parameters():
        param.requires_grad_(False)

    trainable_params = filter(lambda p: p.requires_grad, ddp.get_model(diffusion_planner, args.ddp).parameters())
    params = [{'params': trainable_params, 'lr': args.learning_rate}]

    optimizer = optim.AdamW(params)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, train_epochs, args.warm_up_epoch)

    if args.resume_model_path is not None:
        print(f"Model loaded from {args.resume_model_path}")
        diffusion_planner, optimizer, scheduler, init_epoch, wandb_id, model_ema = resume_model(args.resume_model_path, diffusion_planner, optimizer, scheduler, model_ema, args.device)
    else:
        init_epoch = 0
        wandb_id = None

    for name, param in diffusion_planner.module.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    # logger
    wandb_logger = Logger(args.name, args.notes, args, wandb_resume_id=wandb_id, save_path=save_path, rank=global_rank) 

    if args.ddp:
        torch.distributed.barrier()

    # begin training
    for epoch in range(init_epoch, train_epochs):
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{train_epochs}")
        train_loss, train_total_loss = train_epoch(train_loader, diffusion_planner, optimizer, args, model_ema, aug)
        


        if global_rank == 0:
            lr_dict = {'lr': optimizer.param_groups[0]['lr']}
            wandb_logger.log_metrics({f"train_loss/{k}": v for k, v in train_loss.items()}, step=epoch+1)
            wandb_logger.log_metrics({f"lr/{k}": v for k, v in lr_dict.items()}, step=epoch+1)

            if (epoch+1) % args.save_utd == 0:
                # save model at the end of epoch
                save_model(diffusion_planner, optimizer, scheduler, save_path, epoch, train_total_loss, wandb_logger.id, model_ema.ema)
                print(f"Model saved in {save_path}\n")

        scheduler.step()
        train_sampler.set_epoch(epoch + 1)

if __name__ == "__main__":

    args = get_args()
    
    # Run
    model_training(args)
