import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """Standard MLP-Mixer Block"""
    def __init__(self, num_tokens, hidden_dim, token_mlp_dim, channel_mlp_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.token_mixing = nn.Sequential(
            nn.Linear(num_tokens, token_mlp_dim),
            nn.GELU(),
            nn.Linear(token_mlp_dim, num_tokens)
        )
        self.channel_mixing = nn.Sequential(
            nn.Linear(hidden_dim, channel_mlp_dim),
            nn.GELU(),
            nn.Linear(channel_mlp_dim, hidden_dim)
        )

    def forward(self, x):
        # x: [B, N, C]
        y = self.norm1(x)
        y = y.transpose(1, 2)            # [B, C, N]
        y = self.token_mixing(y)
        y = y.transpose(1, 2)            # [B, N, C]
        x = x + y                        # token mixing
        z = self.channel_mixing(self.norm2(x))  # channel mixing
        x = x + z
        return x


class TokenSemAligner(nn.Module):
    """
    Token Semantic Aligner
     [B,256,192]  [B,107,192]  token

    ：
    -  learnable  embedding
    - /
    - MLP-Mixer
    -
    """

    def __init__(
        self,
        in_tokens=256,
        out_tokens=107,
        hidden_dim=192,
        token_mlp_dim=256,
        channel_mlp_dim=768,
        depth=3
    ):
        super().__init__()

        # === learnable embeddings ===
        self.input_pos_embed = nn.Parameter(torch.randn(1, in_tokens, hidden_dim)) #[B,256,192]
        self.semantic_embed = nn.Parameter(torch.randn(1, out_tokens, hidden_dim)) #[B,107,192]

        # === semantic conditioning: project semantic priors onto input tokens ===
        self.semantic_to_input = nn.Linear(out_tokens, in_tokens, bias=False) # 107-->256

        # === token downsample (256→107) ===
        self.downsample = nn.Linear(in_tokens, out_tokens)

        # === mixer blocks after projection ===
        self.blocks = nn.ModuleList([
            MixerBlock(
                num_tokens=out_tokens,
                hidden_dim=hidden_dim,
                token_mlp_dim=token_mlp_dim,
                channel_mlp_dim=channel_mlp_dim
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        """
        x: [B, 256, 192]
        return: [B, 107, 192]
        """
        B = x.size(0)

        x = x + self.input_pos_embed

        sem = self.semantic_embed.expand(B, -1, -1).permute(0, 2, 1)      # [B,192,107]
        sem_proj = self.semantic_to_input(sem).permute(0, 2, 1)           # [B,256,192]
        x = x + sem_proj

        x = x.permute(0, 2, 1)  # [B,192,256]
        x = self.downsample(x)  # [B,192,107]
        x = x.permute(0, 2, 1)  # [B,107,192]

        x = x + self.semantic_embed

        #  Mixer blocks
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

#if __name__ == "__main__":
#    projector = TokenSemAligner()
#    x = torch.randn(2, 256, 192)     # batch=2
#    y = projector(x)
#    print(y.shape)  # torch.Size([2, 107, 192])
