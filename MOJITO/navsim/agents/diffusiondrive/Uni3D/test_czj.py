import torch
import models.uni3d as models
from utils import utils
from types import SimpleNamespace


def make_args():
    """Minimal args config needed to init create_uni3d model."""
    args = SimpleNamespace()

    args.model = "create_uni3d"
    args.pc_model = "eva02_small_patch14_224"
    args.pc_feat_dim = 384
    args.embed_dim = 1024

    args.npoints = 10000
    args.num_group = 512
    args.group_size = 64
    args.pc_encoder_dim = 512

    # dummy device
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.pretrained_pc="/lpai/volumes/base-3da-ali-sh-mix/chengzhijing/Diffusion_Drive/needed/uni3d_model.pt"
    args.drop_path_rate = 0.0
    args.patch_dropout = 0
    args.precision = "amp"
    args.use_embed = False
    args.clip_model = None

    return args


def main():
    args = make_args()

    print("=> Init model...")
    model = getattr(models, args.model)(args=args).to(args.device)
    model.eval()

    # -----------------------------
    # pc: [B, N, 3]
    # rgb: [B, N, 3]
    # -----------------------------
    B = 2
    N = args.npoints

    pc = torch.randn(B, N, 3).to(args.device)
    rgb = torch.rand(B, N, 3).to(args.device)

    print("pc shape:", pc.shape)
    print("rgb shape:", rgb.shape)

    # -----------------------------
    # -----------------------------
    feature = torch.cat((pc, rgb), dim=-1)     # [B, N, 6]
    print("feature shape:", feature.shape)

    # -----------------------------
    # -----------------------------
    print("=> Running encode_pc...")
    pc_features = utils.get_model(model).encode_pc(feature)

    print("encode_pc output shape:", pc_features.shape)
    print("DONE.")


if __name__ == "__main__":
    main()
