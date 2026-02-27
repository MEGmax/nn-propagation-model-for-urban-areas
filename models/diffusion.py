# Diffusion-based DAIRMap: conditional diffusion model for radio-map prediction.
# Usage: import this module and run train() or sample() as needed.

import math
from typing import Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import os
import json

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
        self.op = nn.AvgPool2d(2)

    def forward(self, x):
        return self.op(x)

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.op = nn.Upsample(scale_factor=2, mode='nearest')

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
            # First resblock takes concatenated input (ch + out_ch)
            self.res_blocks_up.append(ResBlock(ch + out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
            # Subsequent resblocks take output of previous block
            for _ in range(num_res_blocks - 1):
                self.res_blocks_up.append(ResBlock(out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
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

        hs = []  # skip connections
        h = self.input_conv(x_noisy)
        
        # encoder pass: apply ResBlocks then downsample, save skip after resblocks
        idx = 0
        for down in self.downs:
            # Apply all res blocks for this stage
            for _ in range(2):  # num_res_blocks default 2
                h = self.res_blocks_down[idx](h, t_emb, cond_vec)
                idx += 1
            # Save the feature map BEFORE downsampling as skip connection
            hs.append(h)
            # Then downsample
            h = down(h)
        
        # middle
        h = self.mid_block1(h, t_emb, cond_vec)
        h = self.mid_block2(h, t_emb, cond_vec)

        # decoder: upsample and consume skips (pop in reverse order)
        idx_up = 0
        for stage_idx, up in enumerate(self.ups):
            # Upsample first
            h = up(h)
            # Pop the corresponding skip connection (LIFO - last saved, first used)
            skip = hs.pop()
            
            # Ensure spatial dimensions match (handle odd input sizes from nearest upsampling)
            h_h, h_w = h.shape[-2:]
            s_h, s_w = skip.shape[-2:]
            if h_h != s_h or h_w != s_w:
                if h_h < s_h:
                    # Pad with reflection if smaller
                    h = F.pad(h, (0, s_w - h_w, 0, s_h - h_h), mode='reflect')
                else:
                    # Crop if larger
                    h = h[:, :, :s_h, :s_w]
            
            # Concatenate BEFORE first resblock of this stage
            h = torch.cat([h, skip], dim=1)
            # Apply res blocks for this decoder stage
            for block_idx in range(2):
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

    def p_losses(self, x_start, cond_img, t, valid_mask: Optional[torch.Tensor] = None):
        """
        Standard epsilon-prediction loss (MSE between true noise and model-predicted noise)
        """
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        pred_noise = self.model(x_noisy, t, cond_img)

        if valid_mask is None:
            return F.mse_loss(pred_noise, noise)

        # Ensure mask is broadcastable to (B, C, H, W)
        if valid_mask.dim() == 3:
            valid_mask = valid_mask.unsqueeze(1)
        valid_mask = valid_mask.to(device=pred_noise.device, dtype=pred_noise.dtype)

        loss_map = (pred_noise - noise) ** 2
        masked_loss = loss_map * valid_mask
        denom = valid_mask.sum().clamp_min(1e-8)
        return masked_loss.sum() / denom

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
    def sample(self, cond_img, shape=None, steps=50):
        """
        Fast sampling loop (can reduce steps to 10-20 for speed; quality may drop)
        cond_img: conditioning image (B, cond_channels, H, W)
        shape: optional output shape (B, 1, H, W). If None, infers spatial dims from cond_img
        """
        self.model.eval()
        b = cond_img.shape[0]
        
        # Infer shape from conditioning image if not provided
        if shape is None:
            _, _, h, w = cond_img.shape
            shape = (b, 1, h, w)
        
        x = torch.randn(shape, device=self.device)
        timesteps = np.linspace(0, self.timesteps - 1, steps, dtype=int)[::-1]
        for i, t in enumerate(timesteps):
            t_idx = int(t)
            t_tensor = torch.tensor([t_idx], device=self.device).repeat(b)
            with torch.no_grad():
                x = self.p_sample(x, t_idx, cond_img)
        return x

class RadioMapDataset(Dataset):
    """
    Loads input and target tensors from separate directories.
    
    Expects:
      - input_dir: contains input_tensor files (H, W, 2) with channels [elevation, distance]
      - target_dir: contains target_tensor files (H, W, 1) with RSS in dBm
      - mask_dir (optional): contains per-scene mask folder
        {scene_name}/boolean_mask.npy and {scene_name}/rss_null_mask.npy (H, W, 1)
    """
    def __init__(self, input_dir, target_dir, mask_dir=None, transform=None):
        super().__init__()
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get list of input files (assume matching files in target_dir)
        self.input_files = sorted([f for f in os.listdir(input_dir) if f.endswith('_input.npy')])
        self.scene_names = [f.replace('_input.npy', '') for f in self.input_files]

    def __len__(self):
        return len(self.scene_names)

    def __getitem__(self, idx):
        scene_name = self.scene_names[idx]
        
        # Load input tensor (H, W, 2)
        input_tensor = np.load(os.path.join(self.input_dir, f"{scene_name}_input.npy"))
        
        # Load target tensor (H, W, 1)
        target_tensor = np.load(os.path.join(self.target_dir, f"{scene_name}_target.npy"))
        
        # Convert to torch tensors, permute to (C, H, W) format expected by models
        input_tensor = torch.from_numpy(input_tensor).permute(2, 0, 1).float()  # (2, H, W)
        target_tensor = torch.from_numpy(target_tensor).permute(2, 0, 1).float()  # (1, H, W)

        if self.mask_dir is not None:
            scene_mask_dir = os.path.join(self.mask_dir, scene_name)
            boolean_mask = np.load(os.path.join(scene_mask_dir, "boolean_mask.npy"))
            rss_null_mask = np.load(os.path.join(scene_mask_dir, "rss_null_mask.npy"))
            boolean_mask = torch.from_numpy(boolean_mask).permute(2, 0, 1).float()  # (1, H, W)
            rss_null_mask = torch.from_numpy(rss_null_mask).permute(2, 0, 1).float()  # (1, H, W)
        else:
            # Fallback for legacy datasets: no masking
            boolean_mask = torch.zeros_like(target_tensor)
            rss_null_mask = torch.zeros_like(target_tensor)
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'boolean_mask': boolean_mask,
            'rss_null_mask': rss_null_mask
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
            rss = batch['target'].to(device)  # (B,1,H,W)
            cond = batch['input'].to(device)  # (B,C,H,W)
            boolean_mask = batch['boolean_mask'].to(device)
            rss_null_mask = batch['rss_null_mask'].to(device)
            valid_mask = ((boolean_mask == 0) & (rss_null_mask == 0)).float()
            bs = rss.shape[0]
            # choose random timesteps for each sample
            t = torch.randint(0, timesteps, (bs,), device=device).long()
            loss = diffusion.p_losses(rss, cond, t, valid_mask=valid_mask)
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