import torch
import torch.nn as nn
import math

def sinusoidal_embedding(timesteps, dim):
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000) * torch.arange(half, device=timesteps.device).float() / (half - 1)
    )
    args = timesteps.float()[:, None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return emb

class TransformerCond(nn.Module):
    def __init__(self, nfeats, hidden_dim=512, depth=8, nheads=8, cond_dim=768):
        super().__init__()
        self.input_proj = nn.Linear(nfeats, hidden_dim)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(cond_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nheads, batch_first=True, dim_feedforward=hidden_dim*4
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.out_proj = nn.Linear(hidden_dim, nfeats)

    def forward(self, x, timesteps, text_embed):
        # x: [B, T, D]
        h = self.input_proj(x)

        # diffusion time embedding
        t_emb = sinusoidal_embedding(timesteps, h.shape[-1])
        h = h + self.time_proj(t_emb)[:, None, :]

        # add text conditioning
        c = self.text_proj(text_embed)[:, None, :]  # [B,1,H]
        h = h + c

        h = self.encoder(h)
        return self.out_proj(h)  # predict noise, same shape as input
