
import torch
import torch.nn as nn
from collections import OrderedDict
import re

def load_ptv3_pretrained_weights(
    model: nn.Module,
    pretrained_path: str,
    strict: bool = False,
    verbose: bool = True
):
    """
     PTv3
    
    Args:
        model:  PTv3FeatureExtractor
        pretrained_path:
        strict: （False ）
        verbose:
    
    Returns:
    """
    # ========================================
    # ========================================
    print(f"📦 Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        pretrained_state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        pretrained_state_dict = checkpoint['model']
    else:
        pretrained_state_dict = checkpoint
    
    # ========================================
    # ========================================
    
    stage_block_mapping = create_stage_block_mapping()
    
    # ========================================
    # ========================================
    new_state_dict = OrderedDict()
    
    for old_key, param in pretrained_state_dict.items():
        if should_skip_key(old_key):
            if verbose:
                print(f"  ⏭️  Skip: {old_key}")
            continue
        
        new_key = convert_key(old_key, stage_block_mapping)
        
        if new_key is None:
            if verbose:
                print(f"  ❌ Cannot map: {old_key}")
            continue
        
        if new_key in model.state_dict():
            if param.shape == model.state_dict()[new_key].shape:
                new_state_dict[new_key] = param
                if verbose:
                    print(f"  ✅ {old_key} -> {new_key}")
            else:
                if verbose:
                    print(f"  ⚠️  Shape mismatch: {old_key} {param.shape} vs {model.state_dict()[new_key].shape}")
        else:
            if verbose:
                print(f"  ❓ Key not in model: {new_key}")
    
    # ========================================
    # ========================================
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=strict)
    
    # ========================================
    # ========================================
    total_pretrained = len(pretrained_state_dict)
    total_loaded = len(new_state_dict)
    total_model = len(model.state_dict())
    
    print("\n" + "="*80)
    print("📊 Loading Summary:")
    print(f"  - Pretrained params: {total_pretrained}")
    print(f"  - Successfully loaded: {total_loaded}")
    print(f"  - Model total params: {total_model}")
    print(f"  - Coverage: {total_loaded/total_model*100:.2f}%")
    print(f"  - Missing keys: {len(missing_keys)}")
    print(f"  - Unexpected keys: {len(unexpected_keys)}")
    print("="*80 + "\n")
    
    if verbose and len(missing_keys) > 0:
        print("⚠️  Missing keys (will use random init):")
        for key in missing_keys[:10]:
            print(f"    - {key}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")
    
    return {
        'loaded': total_loaded,
        'total': total_model,
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys
    }


def create_stage_block_mapping():
    """
     stage/block  block index
    
    :
    - enc0: 2 blocks  -> blocks 0-1
    - enc1: 2 blocks  -> blocks 2-3
    - enc2: 2 blocks  -> blocks 4-5
    - enc3: 6 blocks  -> blocks 6-11
    - enc4: 2 blocks  -> (，)
    
    : {(stage_idx, block_idx): new_block_idx}
    """
    mapping = {}
    
    # enc0: 2 blocks
    mapping[(0, 0)] = 0
    mapping[(0, 1)] = 1
    
    # enc1: 2 blocks
    mapping[(1, 0)] = 2
    mapping[(1, 1)] = 3
    
    # enc2: 2 blocks
    mapping[(2, 0)] = 4
    mapping[(2, 1)] = 5
    
    # enc3: 6 blocks
    mapping[(3, 0)] = 6
    mapping[(3, 1)] = 7
    mapping[(3, 2)] = 8
    mapping[(3, 3)] = 9
    mapping[(3, 4)] = 10
    mapping[(3, 5)] = 11
    
    
    return mapping


def should_skip_key(key: str) -> bool:
    """Whether to skip a weight key."""
    skip_patterns = [
        'seg_head',
        'dec.',
        'enc.enc4',
        'down.',
    ]
    
    for pattern in skip_patterns:
        if pattern in key:
            return True
    return False


def convert_key(old_key: str, stage_block_mapping: dict) -> str:
    """
     key
    
    Examples:
        module.backbone.embedding.stem.conv.weight
        -> ptv3_extractor_lidar.input_proj.0.weight
        
        module.backbone.enc.enc0.block0.cpe.0.weight
        -> ptv3_extractor_lidar.blocks.0.cpe.conv.weight
        
        module.backbone.enc.enc3.block2.attn.qkv.weight
        -> ptv3_extractor_lidar.blocks.8.attn.qkv.weight
    """
    # ========================================
    # ========================================
    if 'embedding.stem' in old_key:
        # module.backbone.embedding.stem.conv.weight
        # -> ptv3_extractor_lidar.input_proj.0.weight
        suffix = old_key.split('embedding.stem.')[-1]
        
        if 'conv.weight' in suffix:
            return 'input_proj.0.weight'
        elif 'norm.weight' in suffix:
            return 'input_proj.1.weight'
        elif 'norm.bias' in suffix:
            return 'input_proj.1.bias'
        elif 'norm.running_mean' in suffix:
            return 'input_proj.1.running_mean'
        elif 'norm.running_var' in suffix:
            return 'input_proj.1.running_var'
        elif 'norm.num_batches_tracked' in suffix:
            return 'input_proj.1.num_batches_tracked'
        else:
            return None
    
    # ========================================
    # ========================================
    match = re.match(r'module\.backbone\.enc\.enc(\d+)\.block(\d+)\.(.*)', old_key)
    
    if match:
        stage_idx = int(match.group(1))
        block_idx = int(match.group(2))
        suffix = match.group(3)
        
        if (stage_idx, block_idx) not in stage_block_mapping:
            return None
        
        new_block_idx = stage_block_mapping[(stage_idx, block_idx)]
        
        new_suffix = convert_block_suffix(suffix)
        
        if new_suffix is None:
            return None
        
        return f'blocks.{new_block_idx}.{new_suffix}'
    
    return None


def convert_block_suffix(suffix: str) -> str:
    """
     block
    
    Examples:
        cpe.0.weight -> cpe.conv.weight
        norm1.0.weight -> norm1.weight
        mlp.0.fc1.weight -> mlp.fc1.weight
    """
    if suffix.startswith('cpe.'):
        # cpe.0.weight -> cpe.conv.weight
        # cpe.1.weight -> cpe.linear.weight
        # cpe.2.weight -> cpe.norm.weight
        parts = suffix.split('.')
        if parts[1] == '0':  # spconv
            return 'cpe.conv.' + '.'.join(parts[2:])
        elif parts[1] == '1':  # linear
            return 'cpe.linear.' + '.'.join(parts[2:])
        elif parts[1] == '2':  # norm
            return 'cpe.norm.' + '.'.join(parts[2:])
    
    elif suffix.startswith('norm1.') or suffix.startswith('norm2.'):
        # norm1.0.weight -> norm1.weight
        parts = suffix.split('.')
        return parts[0] + '.' + '.'.join(parts[2:])
    
    elif suffix.startswith('mlp.'):
        # mlp.0.fc1.weight -> mlp.fc1.weight
        parts = suffix.split('.')
        return 'mlp.' + '.'.join(parts[2:])
    
    elif suffix.startswith('attn.'):
        return suffix
    
    else:
        return None


# ========================================
# ========================================

def example_usage():
    """Usage example."""
    from  navsim.agents.diffusiondrive.hierarchical_fusion_module_pt3 import PTv3FeatureExtractor
    
    model = PTv3FeatureExtractor(
        in_channels=5,
        hidden_dim=384,
        num_layers=12,
        num_heads=8,
    )
    
    load_info = load_ptv3_pretrained_weights(
        model=model,
        pretrained_path='/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/DiffusionDrive/navsim/agents/diffusiondrive/pointcept/models/point_transformer_v3/model_best.pth',
        strict=False,
        verbose=True
    )
    
    print(f"\n✅ Successfully loaded {load_info['loaded']}/{load_info['total']} parameters")
    
    for name, param in model.named_parameters():
        if not any(x in name for x in ['input_proj', 'blocks']):
            continue
        if name not in load_info['missing_keys']:
            param.requires_grad = False
    
    print("\n🔒 Frozen pretrained parameters")
    
    return model


if __name__ == "__main__":
    example_usage()