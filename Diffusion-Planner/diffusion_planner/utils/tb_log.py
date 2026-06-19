import os
from torch.utils.tensorboard import SummaryWriter

import wandb

class TensorBoardLogger():
    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, rank=0):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """              
        self.args = args
        self.writer = None
        self.id = None
        
        if rank == 0:
            os.environ["WANDB_MODE"] = "online" if args.use_wandb else "offline"

            wandb_writer = wandb.init(project='Diffusion-Planner', 
                name=run_name, 
                notes=notes,
                resume="allow",
                id = wandb_resume_id,
                sync_tensorboard=True,
                dir=f'{save_path}')
            wandb.config.update(args)
            self.id = wandb_writer.id
            
            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    def log_metrics(self, metrics: dict, step: int):
       """
       metrics (dict):
       step (int, optional): epoch or step
       """
       if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self):
       if self.writer is not None:
            self.writer.close()