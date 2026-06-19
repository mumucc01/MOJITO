from torch.optim.lr_scheduler import SequentialLR, LinearLR, MultiplicativeLR

def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1):
    assert epoch >= warm_up_epoch
    T_warmup = warm_up_epoch
    
    warmup_scheduler = LinearLR(optimizer, start_factor=start_factor, total_iters=warm_up_epoch - 1)
    fixed_scheduler = MultiplicativeLR(optimizer, lr_lambda=lambda epoch: 1.0)
    
    scheduler = SequentialLR(optimizer, 
                             schedulers=[warmup_scheduler, fixed_scheduler], 
                             milestones=[T_warmup])
    
    return scheduler
'''
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

def CosineAnnealingWarmUpRestarts(optimizer, epoch, warm_up_epoch, start_factor=0.1, eta_min=1e-4):
    
    assert epoch >= warm_up_epoch
    
    # 预热阶段：线性增加学习率
    warmup_scheduler = LinearLR(optimizer, 
                               start_factor=start_factor, 
                               total_iters=warm_up_epoch-1)
    
    # 主训练阶段：余弦退火
    cosine_scheduler = CosineAnnealingLR(optimizer, 
                                        T_max=epoch - warm_up_epoch, 
                                        eta_min=eta_min)
    
    scheduler = SequentialLR(optimizer, 
                            schedulers=[warmup_scheduler, cosine_scheduler], 
                            milestones=[warm_up_epoch])
    
    return scheduler
'''