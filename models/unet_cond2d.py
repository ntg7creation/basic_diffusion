# models/unet_cond2d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: [B] int/long
    returns:  [B, dim]
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb


class FiLM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(in_dim, out_dim)
        self.to_shift = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, emb: torch.Tensor):
        # x: [B, C, H, W], emb: [B, E]
        s = self.to_scale(emb).unsqueeze(-1).unsqueeze(-1)
        b = self.to_shift(emb).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + s) + b


class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out, time_dim, use_film=True):
        super().__init__()
        self.use_film = use_film
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, c_out)
        self.norm2 = nn.GroupNorm(8, c_out)
        if use_film:
            self.film1 = FiLM(time_dim, c_out)
            self.film2 = FiLM(time_dim, c_out)
        self.act = nn.SiLU()

    def forward(self, x, t_emb):
        x = self.conv1(x)
        x = self.norm1(x)
        if self.use_film:
            x = self.film1(x, t_emb)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)
        if self.use_film:
            x = self.film2(x, t_emb)
        x = self.act(x)
        return x


class Down(nn.Module):
    def __init__(self, c_in, c_out, time_dim):
        super().__init__()
        self.block = ConvBlock(c_in, c_out, time_dim)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        skip = x
        x = self.pool(x)
        return x, skip


class Up(nn.Module):
    def __init__(self, c_in, c_out, time_dim):
        super().__init__()
        self.block = ConvBlock(c_in + c_out, c_out, time_dim)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t_emb)
        return x


class UNetCond2D(nn.Module):
    """
    Super small U-Net:
      - conditioning: concat a broadcasted text feature map (Ccond) to input
      - timestep embedding via FiLM inside blocks
    """
    def __init__(self, in_channels=1, cond_channels=8, base_channels=64, time_dim=128, out_channels=1):
        super().__init__()
        self.in_channels = in_channels
        self.cond_channels = cond_channels
        c0 = base_channels

        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # encoder
        self.down1 = Down(in_channels + cond_channels, c0, time_dim)
        self.down2 = Down(c0, c0 * 2, time_dim)
        self.mid   = ConvBlock(c0 * 2, c0 * 2, time_dim)

        # decoder
        self.up1 = Up(c0 * 2, c0, time_dim)
        self.up2 = ConvBlock(c0, c0, time_dim)
        self.out = nn.Conv2d(c0, out_channels, 1)

    def forward(self, x, timesteps, cond_map):
        """
        x:          [B, 1, H, W]
        timesteps:  [B]  (int64)
        cond_map:   [B, Ccond, H, W]  (broadcasted text embedding)
        """
        t_emb = sinusoidal_embedding(timesteps, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([x, cond_map], dim=1)  # concat cond channels

        x, s1 = self.down1(x, t_emb)  # -> c0
        x, s2 = self.down2(x, t_emb)  # -> c0*2
        x = self.mid(x, t_emb)
        x = self.up1(x, s2, t_emb)
        x = self.up2(x, t_emb)
        x = self.out(x)
        return x
