"""Compare Uni3D checkpoint encoder keys vs current model."""
import torch

ckpt_path = "/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/needed/uni3d_model.pt"
ckpt = torch.load(ckpt_path, map_location='cpu')
sd = ckpt.get('module', ckpt.get('state_dict', ckpt))

print("=" * 80)
print("Uni3D checkpoint encoder keys:")
print("=" * 80)
for k in sorted(sd.keys()):
    if 'encoder' in k.lower() or 'first_conv' in k.lower() or 'second_conv' in k.lower():
        print(f"  {k}: {sd[k].shape}")

print("\n" + "=" * 80)
print("Uni3D checkpoint point_encoder keys (first 50):")
print("=" * 80)
for i, k in enumerate(sorted(sd.keys())):
    if k.startswith('point_encoder.'):
        print(f"  {k}: {sd[k].shape}")

print("\n" + "=" * 80)
print("All keys (first 80):")
print("=" * 80)
for i, k in enumerate(sorted(sd.keys())):
    if i < 80:
        print(f"  {k}: {sd[k].shape}")
    else:
        print(f"  ... total {len(sd)} keys")
        break

print("\n" + "=" * 80)
print("Current model encoder keys:")
print("=" * 80)
from navsim.agents.diffusiondrive.hierarchical_fusion_module_pt3 import PTv3FeatureExtractor, Uni3DConfig
import timm
model = PTv3FeatureExtractor(
    point_transformer=timm.create_model(Uni3DConfig.pc_model, checkpoint_path=None, drop_path_rate=Uni3DConfig.drop_path_rate),
    freeze_backbone=False,
    config=Uni3DConfig
)
for k, v in sorted(model.state_dict().items()):
    if 'encoder' in k.lower() or 'first_conv' in k.lower() or 'second_conv' in k.lower():
        print(f"  {k}: {v.shape}")

print("\n" + "=" * 80)
print("Match analysis:")
print("=" * 80)
sd_transformed = {k.replace('point_encoder.', ''): v for k, v in sd.items()}
encoder_keys_ckpt = {k: v for k, v in sd_transformed.items() if k.startswith('encoder.')}
encoder_keys_model = {k: v for k, v in model.state_dict().items() if k.startswith('encoder.')}

print(f"\nCheckpoint encoder keys ({len(encoder_keys_ckpt)}):")
for k in sorted(encoder_keys_ckpt.keys()):
    print(f"  {k}: {encoder_keys_ckpt[k].shape}")

print(f"\nModel encoder keys ({len(encoder_keys_model)}):")
for k in sorted(encoder_keys_model.keys()):
    print(f"  {k}: {encoder_keys_model[k].shape}")

print(f"\n:")
for k in sorted(encoder_keys_model.keys()):
    if k in encoder_keys_ckpt:
        match = "Shape match" if encoder_keys_ckpt[k].shape == encoder_keys_model[k].shape else f"❌ : ckpt={encoder_keys_ckpt[k].shape} vs model={encoder_keys_model[k].shape}"
        print(f"  {k}: {match}")
    else:
        print(f"  {k}: ❌ checkpoint ")

for k in sorted(encoder_keys_ckpt.keys()):
    if k not in encoder_keys_model:
        print(f"  {k}: ⚠️  (ckpt shape: {encoder_keys_ckpt[k].shape})")