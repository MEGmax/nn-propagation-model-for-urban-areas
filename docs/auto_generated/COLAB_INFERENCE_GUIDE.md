# Colab Inference Guide - TimeCondUNet Diffusion Model

## Overview
This guide shows how to run inference with your trained `model_final.pt` checkpoint in Google Colab.

## Step 1: Create a New Colab Notebook

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click "New notebook"
3. Copy-paste the cells below in order

## Step 2: Install Dependencies

```python
# Install required packages
!pip install torch torchvision -q
!pip install numpy matplotlib tqdm -q
```

## Step 3: Mount Google Drive & Upload Model

```python
from google.colab import drive
drive.mount('/content/drive')

# Path to your model in Google Drive
MODEL_PATH = '/content/drive/MyDrive/model_final.pt'  # Adjust path as needed
```

Before running, upload `model_final.pt` to your Google Drive (or use the path where you store it).

## Step 4: Define Model Architecture

Copy the TimeCondUNet class and supporting components:

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np

# ===== Time Embedding =====
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

# ===== ResBlock with FiLM Conditioning =====
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
        
        if self.time_fc is not None and t_emb is not None:
            tparams = self.time_fc(t_emb)
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
        h = self.dropout(h)
        return h + self.res_conv(x)

# ===== Downsample/Upsample =====
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

# ===== Main U-Net Model =====
class TimeCondUNet(nn.Module):
    def __init__(self, in_ch=1, cond_channels=3, base_ch=32, channel_mults=(1,2,4), 
                 num_res_blocks=2, time_emb_dim=128, cond_emb_dim=128, timesteps=1000):
        super().__init__()
        self.in_ch = in_ch
        self.cond_channels = cond_channels
        self.base_ch = base_ch
        self.timesteps = timesteps
        self.device = None
        
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU()
        )
        
        self.cond_proj = None
        if cond_channels > 0:
            self.cond_proj = nn.Sequential(
                nn.Conv2d(cond_channels, base_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(base_ch, cond_emb_dim),
                nn.SiLU(),
                nn.Linear(cond_emb_dim, cond_emb_dim)
            )
        
        self.input_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        
        # Encoder
        self.downs = nn.ModuleList()
        self.res_blocks_down = nn.ModuleList()
        ch = base_ch
        
        for mult in channel_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.res_blocks_down.append(ResBlock(ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
                ch = out_ch
            self.downs.append(Downsample(ch))
        
        # Bottleneck
        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim),
            ResBlock(ch, ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim)
        ])
        
        # Decoder
        self.ups = nn.ModuleList()
        self.res_blocks_up = nn.ModuleList()
        
        for mult in reversed(channel_mults):
            out_ch = base_ch * mult
            for i in range(num_res_blocks + 1):
                if i == 0:
                    self.res_blocks_up.append(ResBlock(ch + out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
                else:
                    self.res_blocks_up.append(ResBlock(out_ch, out_ch, time_emb_dim=time_emb_dim, cond_dim=cond_emb_dim))
                ch = out_ch
            self.ups.append(Upsample(ch))
        
        self.final_conv = nn.Conv2d(ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x_noisy, t, cond_img=None):
        t_emb = self.time_emb(t)
        cond_vec = None
        if cond_img is not None and self.cond_proj is not None:
            cond_vec = self.cond_proj(cond_img)
        
        h = self.input_conv(x_noisy)
        skips = [h]
        
        # Encoder
        rb_idx = 0
        for i, down in enumerate(self.downs):
            for _ in range(2):
                h = self.res_blocks_down[rb_idx](h, t_emb, cond_vec)
                rb_idx += 1
                skips.append(h)
            h = down(h)
        
        # Bottleneck
        for block in self.bottleneck:
            h = block(h, t_emb, cond_vec)
        
        # Decoder
        rb_idx = 0
        for i, up in enumerate(self.ups):
            h = up(h)
            num_levels = len(self.channel_mults)
            blocks_per_level = 3
            
            for j in range(blocks_per_level):
                skip_tensor = skips.pop()
                B, C_skip, H_skip, W_skip = skip_tensor.shape
                _, C_h, H_h, W_h = h.shape
                
                if H_h != H_skip or W_h != W_skip:
                    if H_h < H_skip or W_h < W_skip:
                        pad_h = H_skip - H_h
                        pad_w = W_skip - W_h
                        h = F.pad(h, (0, pad_w, 0, pad_h), mode='constant', value=0)
                    elif H_h > H_skip or W_h > W_skip:
                        h = h[:, :, :H_skip, :W_skip]
                
                h = torch.cat([h, skip_tensor], dim=1)
                h = self.res_blocks_up[rb_idx](h, t_emb, cond_vec)
                rb_idx += 1
        
        return self.final_conv(h)
    
    @property
    def channel_mults(self):
        return (1, 2, 4)

# ===== Diffusion Sampling Wrapper =====
class DiffusionSampler:
    def __init__(self, model, timesteps=1000, device='cpu'):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(0.0001, 0.02, timesteps, device=device)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def p_sample(self, x, t, cond_img):
        alpha_bar = self.alpha_bars[t]
        beta = self.betas[t]
        
        t_tensor = torch.tensor([t], device=self.device, dtype=torch.long)
        noise_pred = self.model(x, t_tensor, cond_img)
        
        x_0_pred = (x - (1 - alpha_bar).sqrt() * noise_pred) / alpha_bar.sqrt()
        x_0_pred = torch.clamp(x_0_pred, -1, 1)
        
        if t > 0:
            noise = torch.randn_like(x)
            sigma = beta.sqrt()
            x = x_0_pred * alpha_bar.sqrt() + (1 - alpha_bar).sqrt() * noise
        else:
            x = x_0_pred
        
        return x
    
    def sample(self, cond_img, shape=(1,1,107,107), steps=50):
        self.model.eval()
        b = cond_img.shape[0]
        x = torch.randn((b, shape[1], shape[2], shape[3]), device=self.device)
        timesteps = np.linspace(0, self.timesteps - 1, steps, dtype=int)[::-1]
        
        for t in timesteps:
            t_idx = int(t)
            x = self.p_sample(x, t_idx, cond_img)
        
        return x
```

## Step 5: Load Model and Create Sampler

```python
# Set device
device = torch.device('cpu')

# Load model
model = TimeCondUNet(
    in_ch=1, 
    cond_channels=3,  # elevation + distance + frequency
    base_ch=32,
    channel_mults=(1, 2, 4),
    num_res_blocks=2,
    time_emb_dim=128,
    cond_emb_dim=128,
    timesteps=1000
)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

# Create sampler
sampler = DiffusionSampler(model, timesteps=1000, device=device)
print("✓ Model loaded successfully!")
print(f"  Total parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
```

## Step 6: Create Conditioning Inputs

```python
import matplotlib.pyplot as plt

# Example: Create synthetic conditioning inputs
# Format: (batch, 3, 107, 107) with channels [elevation, distance_norm, frequency_log10]

batch_size = 2
H, W = 107, 107

# Elevation: create a simple dome shape
y = np.linspace(-1, 1, H)
x = np.linspace(-1, 1, W)
yy, xx = np.meshgrid(y, x, indexing='ij')
elevation = np.sqrt(yy**2 + xx**2)
elevation = (1 - elevation / elevation.max())  # normalize to [0, 1]

# Distance from transmitter: place TX at center
tx_y, tx_x = H // 2, W // 2
distance = np.sqrt((yy - yy[tx_y, tx_x])**2 + (xx - xx[tx_y, tx_x])**2)
distance_norm = 1 - (distance / distance.max())  # normalize to [0, 1]

# Frequency: assume 2.4 GHz
frequency_log10 = np.ones((H, W)) * np.log10(2.4)
frequency_log10 = (frequency_log10 - np.log10(1)) / (np.log10(6) - np.log10(1))  # normalize

# Stack into conditioning tensor
cond = np.stack([elevation, distance_norm, frequency_log10], axis=0)  # (3, H, W)
cond = np.repeat(cond[np.newaxis, ...], batch_size, axis=0)  # (batch, 3, H, W)
cond_tensor = torch.from_numpy(cond).float().to(device)

print(f"Conditioning shape: {cond_tensor.shape}")
print(f"  - Elevation range: [{elevation.min():.3f}, {elevation.max():.3f}]")
print(f"  - Distance norm range: [{distance_norm.min():.3f}, {distance_norm.max():.3f}]")
print(f"  - Frequency log10 range: [{frequency_log10.min():.3f}, {frequency_log10.max():.3f}]")
```

## Step 7: Run Inference

```python
# Sampling parameters
steps = 50  # Reduce to 20-30 for faster inference
output_shape = (1, 107, 107)

print(f"\n🎲 Sampling with {steps} steps...")
with torch.no_grad():
    rss_predictions = sampler.sample(cond_tensor, shape=(1,) + output_shape, steps=steps)

print(f"✓ Sampling complete!")
print(f"  Output shape: {rss_predictions.shape}")
print(f"  Range: [{rss_predictions.min():.3f}, {rss_predictions.max():.3f}]")

# Denormalize to dBm
RSS_DB_FLOOR = -100.0
RSS_DB_SCALE = 50.0
rss_dbm = rss_predictions.cpu().numpy() * RSS_DB_SCALE + RSS_DB_FLOOR

print(f"  RSS (dBm) range: [{rss_dbm.min():.1f}, {rss_dbm.max():.1f}]")
```

## Step 8: Visualize Results

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Denormalize
RSS_DB_FLOOR = -100.0
RSS_DB_SCALE = 50.0
rss_dbm = rss_predictions.cpu().numpy() * RSS_DB_SCALE + RSS_DB_FLOOR

# Plot
fig, axes = plt.subplots(batch_size, 2, figsize=(12, 5*batch_size))
if batch_size == 1:
    axes = axes.reshape(1, -1)

for i in range(batch_size):
    # Predicted RSS
    ax = axes[i, 0]
    pred_map = rss_dbm[i, 0, :, :]
    im = ax.imshow(pred_map, cmap='RdYlBu_r', aspect='auto')
    ax.set_title(f'Sample {i+1}: Predicted RSS Map')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='RSS (dBm)')
    
    # Conditioning (elevation)
    ax = axes[i, 1]
    elev = cond_tensor[i, 0, :, :].cpu().numpy()
    im = ax.imshow(elev, cmap='gray', aspect='auto')
    ax.set_title(f'Sample {i+1}: Elevation Map')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    plt.colorbar(im, ax=ax, label='Elevation (norm)')

plt.tight_layout()
plt.savefig('/content/rss_predictions.png', dpi=100, bbox_inches='tight')
print("✓ Visualization saved as rss_predictions.png")
plt.show()
```

## Step 9: Save Results

```python
# Save as numpy file
output_file = '/content/drive/MyDrive/rss_predictions.npy'
np.save(output_file, rss_dbm)
print(f"✓ Results saved to {output_file}")

# Save as CSV (optional)
for i in range(batch_size):
    csv_file = f'/content/drive/MyDrive/rss_sample_{i}.csv'
    np.savetxt(csv_file, rss_dbm[i, 0, :, :], delimiter=',', fmt='%.2f')
    print(f"  - Sample {i+1}: {csv_file}")
```

---

## Key Parameters to Adjust

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `steps` | Number of diffusion steps (quality vs speed) | 20-100 |
| `batch_size` | Process multiple samples at once | 1-4 |
| `H, W` | Must match training grid size | 107×107 |
| `elevation` | Scene geometry conditioning | 0-1 |
| `distance_norm` | Distance from transmitter | 0-1 |
| `frequency_log10` | Frequency band (normalized) | Depends on range |

**Quality Tips**:
- Lower `steps` (20-30) for faster preview inference
- Higher `steps` (50-100) for higher-quality predictions
- Batch multiple samples for efficiency

## Troubleshooting

**Issue**: "RuntimeError: Expected tensor on device X but got tensor on device Y"
- **Fix**: Ensure all tensors are on same device (CPU or CUDA)
```python
cond_tensor = cond_tensor.to(device)
```

**Issue**: "RuntimeError: shape mismatch" during concatenation
- **Fix**: Verify conditioning shape is `(batch, 3, 107, 107)`
```python
print(cond_tensor.shape)  # Should be (batch_size, 3, 107, 107)
```

**Issue**: Model outputs all zeros or NaNs
- **Fix**: Check checkpoint loaded correctly
```python
print(checkpoint.keys())  # Should contain 'model_state' or weights directly
```

**Issue**: Very slow inference on CPU
- **Fix**: Reduce `steps` and `batch_size`
```python
steps = 20  # Minimum viable
batch_size = 1
```
