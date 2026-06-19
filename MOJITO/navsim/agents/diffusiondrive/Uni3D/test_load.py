import torch

model_path = "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/needed/uni3d_b.pt"

try:
    checkpoint = torch.load(model_path, map_location='cpu')
    
    print("=" * 80)
    print("Model file info:")
    print("=" * 80)
    
    print(f"1. : {type(checkpoint)}")
    print(f"2. : {isinstance(checkpoint, dict)}")
    
    if isinstance(checkpoint, dict):
        print(f"3. : {list(checkpoint.keys())}")
        print(f"4. : {len(checkpoint)}")
        
        print("\n" + "=" * 80)
        print("Detailed content analysis:")
        print("=" * 80)
        
        for key, value in checkpoint.items():
            print(f"\n: {key}")
            print(f"  : {type(value)}")
            
            if torch.is_tensor(value):
                print(f"  : {value.shape}")
                print(f"  : {value.dtype}")
                print(f"  : {value.device}")
                print(f"  5: {value.flatten()[:5] if value.numel() > 5 else value}")
            elif isinstance(value, dict):
                print(f"  : {list(value.keys())}")
                print(f"  : {len(value)}")
                if key == 'state_dict' or key == 'model_state_dict':
                    print(f"\n  : {len(value)}")
                    count = 0
                    for param_name, param_value in value.items():
                        if count < 5:
                            print(f"    {param_name}: {param_value.shape}")
                            count += 1
                        else:
                            print(f"    ...  {len(value)-5} ")
                            break
            elif isinstance(value, list):
                print(f"  : {len(value)}")
                if len(value) > 0:
                    print(f"  : {type(value[0])}")
            elif isinstance(value, (int, float, str, bool)):
                print(f"  : {value}")
            else:
                print(f"  : {value}")
                
    elif torch.is_tensor(checkpoint):
        print(f"3. : {checkpoint.shape}")
        print(f"4. : {checkpoint.dtype}")
        print(f"5. : [{checkpoint.min().item():.6f}, {checkpoint.max().item():.6f}]")
        print(f"6. : {checkpoint.mean().item():.6f}")
        print(f"7. : {checkpoint.std().item():.6f}")
        
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("\n" + "=" * 80)
        print("Model parameter statistics:")
        print("=" * 80)
        
        total_params = 0
        for name, param in state_dict.items():
            if torch.is_tensor(param):
                param_count = param.numel()
                total_params += param_count
                print(f"{name}: {param.shape} (: {param_count:,})")
        
        print(f"\n: {total_params:,}")
        print(f": {total_params * 4 / 1024**2:.2f} MB (float32)")
        
    if isinstance(checkpoint, dict):
        common_keys = ['epoch', 'model_state_dict', 'optimizer_state_dict', 
                      'loss', 'args', 'config', 'best_val_loss']
        for key in common_keys:
            if key in checkpoint:
                print(f"\n '{key}': {checkpoint[key]}")
    
except Exception as e:
    print(f": {e}")
    import traceback
    traceback.print_exc()