# Diffusion-based DAIRMap: conditional diffusion model for radio-map prediction.
# Usage: import this module and run train() or sample() as needed.

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os

# -------------------------
# Utilities: Sinusoidal time embedding (like Transformer / DDPM)
# -------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: (B,)
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]  # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb  # (B, dim)

# -------------------------
# Simple Residual block with FiLM conditioning
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None, cond_dim=None, dropout=0.0):
        super().__init__()
        self.time_fc = None
        self.cond_fc = None
        mid_ch = out_ch
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, mid_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        if time_emb_dim is not None:
            self.time_fc = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))
        if cond_dim is not None:
            # cond vector -> FiLM scale and shift
            self.cond_fc = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, out_ch * 2))
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.res_conv = nn.Identity()

        self.activation = nn.SiLU()

    def forward(self, x, t_emb: Optional[torch.Tensor] = None, cond: Optional[torch.Tensor] = None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        # apply time and conditional FiLM (scale, shift)
        if self.time_fc is not None and t_emb is not None:
            tparams = self.time_fc(t_emb)  # (B, out_ch*2)
            scale, shift = tparams.chunk(2, dim=-1)
            scale = scale[:, :, None, None]
            shift = shift[:, :, None, None]
            h = h * (1 + scale) + shift

        if self.cond_fc is not None and cond is not None:
            cparams = self.cond_fc(cond)
            cscale, cshift = cparams.chunk(2, dim=-1)
            cscale = cscale[:, :, None, None]
            cshift = cshift[:, :, None, None]
            h = h * (1 + cscale) + cshift

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        return h + self.res_conv(x)

# -------------------------
# Downsample / Upsample helpers
# -------------------------
class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Conv2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.ConvTranspose2d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)

# -------------------------
# U-Net backbone (time-conditional, FiLM-conditioned on scene)
# -------------------------
class TimeCondUNet(nn.Module):
    def __init__(self, in_ch=1, cond_channels=0, base_ch=64, channel_mults=(1,2,4,8), 
                 num_res_blocks=2, time_emb_dim=256, cond_emb_dim=128):
        """
        in_ch: channels of noisy target (e.g., 1 for RSS map)
        cond_channels: number of channels in conditioning image (elevation + tx + extras)
        """
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim // 2),
            nn.Linear(time_emb_dim // 2, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        ) if False else nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU())

        self.cond_proj = None
        if cond_channels > 0:
            # project cond image to vector per-sample (global cond)
            # we do an average pool + linear to produce conditioning vector
            self.cond_proj = nn.Sequential(
                nn.Conv2d(cond_channels, base_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base_ch, cond_emb_dim),
                nn.SiLU(),
                nn.Linear(cond_emb_dim, cond_emb_dim)
            )

        # input conv
        self.input_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        # encoder
        self.downs = nn.ModuleList()
        self.updowns = nn.ModuleList()
        ch = base_ch
        self.res_blocks_down = nn.ModuleList()
        for mult in channel_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.res_blocks_down.append(ResBlock(ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
                ch = out_ch
            self.downs.append(Downsample(ch))

        # middle
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim)

        # decoder
        self.ups = nn.ModuleList()
        self.res_blocks_up = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_ch * mult
            self.ups.append(Upsample(ch))
            for _ in range(num_res_blocks):
                # when concatenating skip, input channels will be (ch + skip_ch)
                self.res_blocks_up.append(ResBlock(ch + out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
                ch = out_ch

        # output conv
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x_noisy, t, cond_img=None):
        """
        x_noisy: (B, C, H, W) noisy target (RSS)
        t: (B,) timesteps in [0, T)
        cond_img: (B, cond_channels, H, W)
        """
        B = x_noisy.shape[0]
        t_emb = self.time_emb(t)  # (B, time_emb_dim)
        cond_vec = None
        if self.cond_proj and cond_img is not None:
            cond_vec = self.cond_proj(cond_img)  # (B, cond_emb_dim)

        hs = []
        h = self.input_conv(x_noisy)
        # encoder pass: apply ResBlocks then downsample
        idx = 0
        for down in self.downs:
            # each downsample stage may have multiple res blocks
            for _ in range(2):  # num_res_blocks default 2
                h = self.res_blocks_down[idx](h, t_emb, cond_vec)
                hs.append(h)
                idx += 1
            h = down(h)
        # middle
        h = self.mid_block1(h, t_emb, cond_vec)
        h = self.mid_block2(h, t_emb, cond_vec)

        # decoder: upsample and consume skips
        idx_up = 0
        for up in self.ups:
            h = up(h)
            # pop last skip (reverse order)
            skip = hs.pop()
            h = torch.cat([h, skip], dim=1)
            # run the res blocks for this stage
            for _ in range(2):
                h = self.res_blocks_up[idx_up](h, t_emb, cond_vec)
                idx_up += 1

        h = self.out_norm(h)
        h = self.out_act(h)
        out = self.out_conv(h)
        return out  # predicted noise (epsilon) by default

# -------------------------
# Diffusion Scheduler (Betas / Alphas)
# -------------------------
def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class Diffusion:
    def __init__(self, model: nn.Module, timesteps=1000, device='cuda'):
        self.model = model
        self.device = device
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer = lambda name, val: setattr(self, name, val)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x_start, t, noise=None):
        # x_t = sqrt(alpha_cumprod[t])*x0 + sqrt(1 - alpha_cumprod[t]) * noise
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_acp = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_omacp = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_acp * x_start + sqrt_omacp * noise

    def p_losses(self, x_start, cond_img, t):
        """
        Standard epsilon-prediction loss (MSE between true noise and model-predicted noise)
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        pred_noise = self.model(x_noisy, t, cond_img)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x, t_index, cond_img):
        # one reverse step (DDPM)
        betas_t = self.betas[t_index]
        sqrt_one_minus_acp_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = (1. / torch.sqrt(self.alphas[t_index]))
        # model predicts noise
        pred_noise = self.model(x, torch.tensor([t_index], device=self.device).repeat(x.shape[0]), cond_img)
        # compute posterior mean
        model_mean = sqrt_recip_alphas_t * (x - betas_t / sqrt_one_minus_acp_t * pred_noise)
        if t_index == 0:
            return model_mean
        noise = torch.randn_like(x)
        sigma = torch.sqrt(betas_t)
        return model_mean + sigma * noise

    @torch.no_grad()
    def sample(self, cond_img, shape=(1,1,200,200), steps=50):
        """
        Fast sampling loop (can reduce steps to 10-20 for speed; quality may drop)
        cond_img: conditioning image (B, cond_channels, H, W)
        """
        self.model.eval()
        b = cond_img.shape[0]
        x = torch.randn((b, shape[1], shape[2], shape[3]), device=self.device)
        timesteps = np.linspace(0, self.timesteps - 1, steps, dtype=int)[::-1]
        for i, t in enumerate(timesteps):
            t_idx = int(t)
            t_tensor = torch.tensor([t_idx], device=self.device).repeat(b)
            with torch.no_grad():
                x = self.p_sample(x, t_idx, cond_img)
        return x

# -------------------------
# Dataset skeleton
# -------------------------
class RadioMapDataset(Dataset):
    """
    Expect inputs:
      - elevation maps: (H, W) floats (meters) normalized externally
      - cond channels: concatenated channels, e.g. tx heatmap, material codes (C_cond, H, W)
      - target radio map: (H, W) in dB normalized to [-1,1] or similar
    This is a skeleton; adapt to your storage format (numpy files, HDF5, etc.).
    """
    def __init__(self, root_dir, filenames, transform=None):
        super().__init__()
        self.root = root_dir
        self.files = filenames
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        # Example: each file is a dict with 'elevation', 'cond', 'rss'
        data = np.load(os.path.join(self.root, fn), allow_pickle=True).item()
        elev = data['elevation'].astype(np.float32)  # (H,W)
        cond = data.get('cond', None)  # (C_cond,H,W)
        rss = data['rss'].astype(np.float32)  # (H,W)
        # normalize / scale as needed (user choice)
        # Here assume rss is in dB range [-150,-50]. Map to [-1,1] for training stability
        rss_norm = (rss + 100.) / 50.  # example mapping
        rss_norm = np.clip(rss_norm, -1.0, 1.0)
        rss_norm = rss_norm[None, :, :]  # add channel dim

        # cond channels: always present as image-like
        if cond is None:
            cond = np.zeros((1, elev.shape[0], elev.shape[1]), dtype=np.float32)
        # include elevation as first cond channel
        cond_full = np.concatenate([elev[None, :, :].astype(np.float32), cond], axis=0)

        return {
            'rss': torch.from_numpy(rss_norm),
            'cond': torch.from_numpy(cond_full)
        }

# -------------------------
# Training loop (simple)
# -------------------------
def train(
    model: TimeCondUNet,
    dataset: RadioMapDataset,
    device='cuda',
    epochs=20,
    batch_size=8,
    lr=2e-4,
    timesteps=1000,
    save_every=5,
    out_dir='./checkpoints'
):
    os.makedirs(out_dir, exist_ok=True)
    model.to(device)
    diffusion = Diffusion(model, timesteps=timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            rss = batch['rss'].to(device)  # (B,1,H,W)
            cond = batch['cond'].to(device)  # (B,C,H,W)
            bs = rss.shape[0]
            # choose random timesteps for each sample
            t = torch.randint(0, timesteps, (bs,), device=device).long()
            loss = diffusion.p_losses(rss, cond, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * bs
            pbar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch+1} average loss: {avg_loss:.6f}")
        if (epoch + 1) % save_every == 0:
            torch.save(model.state_dict(), os.path.join(out_dir, f"model_epoch{epoch+1}.pt"))
    # final save
    torch.save(model.state_dict(), os.path.join(out_dir, "model_final.pt"))
    return model

# -------------------------
# Sampling example
# -------------------------
@torch.no_grad()
def sample_and_save(model: TimeCondUNet, cond_img: torch.Tensor, device='cuda', steps=50, out_path='sample.npy'):
    model.to(device)
    diffusion = Diffusion(model, timesteps=1000, device=device)
    cond_img = cond_img.to(device)
    shape = (cond_img.shape[0], 1, cond_img.shape[2], cond_img.shape[3])
    samples = diffusion.sample(cond_img, shape=shape, steps=steps)
    # map back to dB if you used normalization -- inverse of dataset transform
    samples_cpu = samples.detach().cpu().numpy()
    np.save(out_path, samples_cpu)
    return samples_cpu

# -------------------------
# Example usage (if run as script)
# -------------------------
if __name__ == '__main__':
    # small smoke test with random tensors
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = TimeCondUNet(in_ch=1, cond_channels=2, base_ch=32, channel_mults=(1,2,4), num_res_blocks=2, time_emb_dim=128, cond_emb_dim=64)
    model.to(device)
    print("Model params:", sum(p.numel() for p in model.parameters())/1e6, "M")

    # create fake dataset
    class FakeDS(Dataset):
        def __len__(self): return 64
        def __getitem__(self, idx):
            rss = np.random.randn(1, 128, 128).astype(np.float32)
            cond = np.random.randn(2, 128, 128).astype(np.float32)
            return {'rss': torch.from_numpy(rss), 'cond': torch.from_numpy(cond)}
    ds = FakeDS()
    # train for 1 epoch just to check
    train(model, ds, device=device, epochs=1, batch_size=8, lr=2e-4, timesteps=200, save_every=1, out_dir='./tmp_ckpt')
    # sampling example
    cond_sample = torch.randn((4,2,128,128))
    samples = sample_and_save(model, cond_sample, device=device, steps=20, out_path='sample.npy')
    print("Saved samples shape:", samples.shape)
