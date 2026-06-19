from dataclasses import dataclass

from navsim.mojito_paths import pretrained_path


@dataclass
class Uni3DConfig:
    pc_model = "eva02_small_patch14_224"
    pretrained_pc: str = pretrained_path("uni3d_b.pt")
    #drop_path_rate = 0.0
    drop_path_rate = 0.1
    pc_feat_dim = 384
    embed_dim = 1024
    group_size = 64
    num_group = 512
    pc_encoder_dim = 512
    patch_dropout = 0