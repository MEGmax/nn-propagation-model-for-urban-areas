import math
from typing import Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------
# Embeddings
# ------------------------
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal (Fourier) timestep embedding. Returns vector of size dim."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] (long / int)
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # pad if odd
            emb = F.pad(emb, (0, 1))
        return emb  # [B, dim]


class FreqEmbed(nn.Module):
    """Simple MLP mapping scalar frequency -> embedding vector."""
    def __init__(self, emb_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, freq: torch.Tensor) -> torch.Tensor:
        # freq: [B, 1] or [B]
        if freq.dim() == 1:
            freq = freq.unsqueeze(-1)
        return self.mlp(freq)  # [B, emb_dim]


# ------------------------
# Building blocks
# ------------------------
def conv3x3(in_ch, out_ch, stride=1):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=stride)


class ResBlock(nn.Module):
    """
    Residual block that injects an embedding (t+freq) via a linear layer and addition.
    Uses GroupNorm -> SiLU -> Conv pattern.
    """
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.emb_dim = emb_dim

        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = conv3x3(in_ch, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = conv3x3(out_ch, out_ch)

        self.emb_proj = nn.Linear(emb_dim, out_ch)
        self.activation = nn.SiLU()

        if in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        emb: [B, emb_dim]
        """
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        # add embedding (FiLM-style)
        emb_out = self.emb_proj(emb).unsqueeze(-1).unsqueeze(-1)  # [B, out_ch, 1, 1]
        h = h + emb_out

        h = self.norm2(h)
        h = self.activation(h)
        h = self.conv2(h)
        return h + self.shortcut(x)


class AttentionBlock(nn.Module):
    """Simple self-attention over spatial tokens in small feature maps."""
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.scale = (ch // num_heads) ** -0.5

        self.to_qkv = nn.Conv1d(ch, ch * 3, kernel_size=1)
        self.to_out = nn.Conv1d(ch, ch, kernel_size=1)
        self.norm = nn.GroupNorm(8, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] -> treat H*W as sequence
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W)  # [B, C, N]
        qkv = self.to_qkv(h)  # [B, 3C, N]
        q, k, v = qkv.chunk(3, dim=1)
        q = q.view(B, self.num_heads, C // self.num_heads, H * W)
        k = k.view(B, self.num_heads, C // self.num_heads, H * W)
        v = v.view(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.einsum("bhnm,bhkm->bhnk", q, k) * self.scale  # [B, heads, N, N]
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhnk,bhkm->bhnm", attn, v)  # [B, heads, dim_head, N]
        out = out.contiguous().view(B, C, H * W)
        out = self.to_out(out)
        out = out.view(B, C, H, W)
        return out + x  # residual


# ------------------------
# Down / Up samples
# ------------------------
class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, use_att: bool = False):
        super().__init__()
        self.res = ResBlock(in_ch, out_ch, emb_dim)
        self.pool = nn.AvgPool2d(2)
        self.att = AttentionBlock(out_ch) if use_att else None

    def forward(self, x, emb):
        x = self.res(x, emb)
        if self.att is not None:
            x = self.att(x)
        p = self.pool(x)
        return x, p


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, use_att: bool = False):
        super().__init__()
        # in_ch is concatenated channels from upsample + skip -> typically 2*out_ch_prev
        self.res = ResBlock(in_ch, out_ch, emb_dim)
        self.att = AttentionBlock(out_ch) if use_att else None

    def forward(self, x, skip, emb):
        # x: [B, C, h, w] upsample to match skip
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, emb)
        if self.att is not None:
            x = self.att(x)
        return x


# ------------------------
# UNet
# ------------------------
class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for predicting noise given (noisy_rss + elevation) and conditioning (timestep, frequency).
    - in_ch: number of input channels (e.g., 2: noisy_rss + elevation)
    - out_ch: output channels (1: predicted noise for RSS)
    - base_ch: base channel width
    - channel_mults: multiples for each downsample stage
    - num_res_blocks: number of ResBlocks per level (kept 1 here for clarity)
    - emb_dim: embedding dimensionality for timestep+freq
    """
    def __init__(
        self,
        in_ch: int = 2,
        out_ch: int = 1,
        base_ch: int = 64,
        channel_mults: Sequence[int] = (1, 2, 4, 8),
        emb_dim: int = 256,
        use_attention: Sequence[bool] = (False, False, False, True),
    ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.emb_dim = emb_dim

        # embeddings
        self.timestep_emb = SinusoidalPosEmb(emb_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.SiLU(),
            nn.Linear(emb_dim * 2, emb_dim),
        )
        self.freq_mlp = FreqEmbed(emb_dim)

        # initial conv
        self.init_conv = conv3x3(in_ch, base_ch)

        # build encoder / decoder
        chs = [base_ch * m for m in channel_mults]
        in_chs = [base_ch] + chs[:-1]
        out_chs = chs

        self.downs = nn.ModuleList()
        for ic, oc, att in zip(in_chs, out_chs, use_attention):
            self.downs.append(Down(ic, oc, emb_dim, use_att=att))

        # middle
        self.mid1 = ResBlock(chs[-1], chs[-1], emb_dim)
        self.mid_att = AttentionBlock(chs[-1]) if use_attention[-1] else None

        # build ups (reverse)
        self.ups = nn.ModuleList()
        rev_chs = list(reversed(chs))
        for i in range(len(rev_chs) - 1):
            in_ch_up = rev_chs[i] + rev_chs[i + 1]  # cat of upsample + skip
            out_ch_up = rev_chs[i + 1]
            att = use_attention[len(rev_chs) - 2 - i] if len(use_attention) > 1 else False
            self.ups.append(Up(in_ch_up, out_ch_up, emb_dim, use_att=att))

        # final blocks
        self.final_res = ResBlock(rev_chs[-1] + base_ch, base_ch, emb_dim)
        self.out_conv = nn.Sequential(
            nn.GroupNorm(8, base_ch),
            nn.SiLU(),
            nn.Conv2d(base_ch, out_ch, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, freq: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_ch, H, W]  (noisy_rss concatenated with elevation)
        t: [B] (timestep indices, int tensor)
        freq: [B] or [B,1] (scalar frequency)
        returns: predicted_noise [B, out_ch, H, W]
        """
        # prepare embedding
        t_emb = self.timestep_emb(t)           # [B, emb_dim]
        t_emb = self.timestep_mlp(t_emb)      # [B, emb_dim]
        f_emb = self.freq_mlp(freq)           # [B, emb_dim]
        emb = t_emb + f_emb                   # fuse by addition (you can try concat + proj)

        # down
        h = self.init_conv(x)
        skips = []
        for d in self.downs:
            skip, h = d(h, emb)
            skips.append(skip)

        # middle
        h = self.mid1(h, emb)
        if self.mid_att is not None:
            h = self.mid_att(h)

        # up
        for up in self.ups:
            skip = skips.pop()
            h = up(h, skip, emb)

        # final combine with first skip
        first_skip = skips.pop(0) if len(skips) > 0 else None
        if first_skip is not None:
            h = torch.cat([h, first_skip], dim=1)
        h = self.final_res(h, emb)
        out = self.out_conv(h)
        return out


# ------------------------
# Smoke test
# ------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, H, W = 2, 128, 128
    in_ch = 2  # noisy_rss + elevation
    model = ConditionalUNet(in_ch=in_ch, out_ch=1, base_ch=48, channel_mults=(1, 2, 4), emb_dim=128).to(device)

    x = torch.randn(B, in_ch, H, W).to(device)
    t = torch.randint(0, 1000, (B,), device=device).long()
    freq = torch.tensor([2.4, 28.0], device=device).unsqueeze(-1).float()  # [B,1]

    out = model(x, t, freq)
    print("out.shape:", out.shape)  # expect [B,1,H,W]
