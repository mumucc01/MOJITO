import sys
import os
import os.path as osp
import math
import torch
import torch.nn as nn
import torchvision.ops as ops
import torch.nn.functional as F
import logging


def _apply_compat_patch():
    """
    Patch DeepSpeed / legacy PyTorch AMP compatibility.
    Strips the PyTorch 2.4+ 'device_type' argument when needed.
    """
    try:
        if not hasattr(torch, 'amp'):
            import types
            torch.amp = types.ModuleType('torch.amp')
            sys.modules['torch.amp'] = torch.amp

        from torch.cuda.amp import custom_fwd as _old_fwd
        from torch.cuda.amp import custom_bwd as _old_bwd

        def _smart_fwd(*args, **kwargs):
            kwargs.pop('device_type', None)
            return _old_fwd(*args, **kwargs)

        def _smart_bwd(*args, **kwargs):
            kwargs.pop('device_type', None)
            if not args:
                return lambda func: _old_bwd(func)
            return _old_bwd(*args, **kwargs)

        if not hasattr(torch.amp, 'custom_fwd'):
            setattr(torch.amp, 'custom_fwd', _smart_fwd)
            setattr(torch.amp, 'custom_bwd', _smart_bwd)

    except Exception as e:
        print(f"[DINOv3Backbone] Warning: AMP compatibility patch failed: {e}")

def _setup_env_path():
    """Set project root and DINOv3 submodule paths for hubconf import."""
    current_dir = osp.dirname(osp.abspath(__file__))
    
    project_root = osp.abspath(osp.join(current_dir, "../../../../"))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    dinov3_repo = osp.abspath(osp.join(current_dir, "../../dinov3"))
    if os.path.exists(dinov3_repo) and dinov3_repo not in sys.path:
        sys.path.insert(0, dinov3_repo)

_setup_env_path()
import hubconf
from navsim.agents.diffusiondrive.modules.dinov3_adapter import DINOv3_Adapter


class DINOv3Backbone(nn.Module):
    def __init__(self, 
                 model_name='dinov3_vits16plus',
                 weights_path=None,
                 patch_size=16,
                 out_indices=None,
                 out_channels=(64, 128, 256, 512),   
                 freeze=True,
                 **kwargs):        
        super().__init__()
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.out_channels = out_channels

        if hasattr(hubconf, model_name):
            build_func = getattr(hubconf, model_name)
            self.raw_model = build_func(pretrained=False)
            # print(self.model)
        else:
            raise ValueError(f"Model {model_name} not found in dinov3/hubconf.py")

        if weights_path:
            print(f"Loading weights from {weights_path}")
            assert os.path.exists(weights_path)
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint.get('model', checkpoint.get('teacher', checkpoint))
        
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

            msg = self.raw_model.load_state_dict(state_dict, strict=False)
            print(f"[DINOv3] Weights loaded: {msg}")
        else:
            print(f"Training Dino From Scratch")
            self.raw_model.init_weights()

        self.num_blocks = len(self.raw_model.blocks)
        self.embed_dim = self.raw_model.embed_dim

        self.adapter = DINOv3_Adapter(
            backbone=self.raw_model
        )



    def forward(self, x, layer_idx):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            tuple([B, C, H/4, W/4], [B, C, H/8, W/8], ...)
        """
        outputs = self.adapter(x,layer_idx)
        
        return outputs



if __name__ == "__main__":
    print("="*40)
    print("🚀 测试 DINOv3 Iterative Execution")
    print("="*40)
    
    weights = "/lpai/volumes/mind-vla-ali-sh-mix/liuxuetao/code/work/Diffusion_Drive/DiffusionDrive/navsim/agents/diffusiondrive copy/dinov3/weights/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth"

    try:
        model = DINOv3Backbone(
            weights_path=weights, 
            patch_size=16,
            out_channels=(512, 512, 512, 512)
        )
        model.eval()
        model.cuda()
        
        H_img, W_img = 256, 1024
        B = 2
        dummy_input = torch.randn(B, 3, H_img, W_img).cuda().to(torch.bfloat16)
        print(f"📥 输入形状: {dummy_input.shape}")

        with torch.autocast("cuda", torch.bfloat16):
            with torch.no_grad():
                outputs = model(dummy_input)

        print("\n📤 输出特征尺寸:")
        for i, out in enumerate(outputs):
            print(f"  Stage {i+1}: {out.shape}")
            
        assert outputs[0].shape[-2:] == (H_img // 4, W_img // 4)
        assert outputs[1].shape[-2:] == (H_img // 8, W_img // 8)
        print("\n✅ 尺寸验证通过！")


    except Exception as e:
        import traceback
        traceback.print_exc()