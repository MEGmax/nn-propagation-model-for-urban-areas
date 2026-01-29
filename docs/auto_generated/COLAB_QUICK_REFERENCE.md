# Quick Reference: Colab Inference Setup

## 5-Minute Setup

### 1. Upload model to Google Drive
- Download `models/checkpoints/model_final.pt` from your local machine
- Upload to Google Drive (e.g., `MyDrive/model_final.pt`)

### 2. Create new Colab Notebook
- Go to [colab.research.google.com](https://colab.research.google.com)
- Click "New notebook"

### 3. Copy-paste these cells in order

**Cell 1: Install & Setup**
```python
!pip install torch torchvision -q

from google.colab import drive
drive.mount('/content/drive')

MODEL_PATH = '/content/drive/MyDrive/model_final.pt'
```

**Cell 2: Copy Model Code**
- Copy the entire `TimeCondUNet` class from [colab_inference_standalone.py](colab_inference_standalone.py) (lines 1-400)
- Run the cell to define the model architecture

**Cell 3: Load Model**
```python
import torch

device = torch.device('cpu')

model = TimeCondUNet(
    in_ch=1, cond_channels=3, base_ch=32,
    channel_mults=(1,2,4), num_res_blocks=2,
    time_emb_dim=128, cond_emb_dim=128, timesteps=1000
)

checkpoint = torch.load(MODEL_PATH, map_location=device)
if 'model_state' in checkpoint:
    model.load_state_dict(checkpoint['model_state'])
else:
    model.load_state_dict(checkpoint)

model = model.to(device).eval()
print(f"✓ Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
```

**Cell 4: Create Conditioning & Sample**
```python
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic conditioning (elevation + distance + frequency)
batch_size = 2
H, W = 107, 107

y = np.linspace(-1, 1, H)
x = np.linspace(-1, 1, W)
yy, xx = np.meshgrid(y, x, indexing='ij')

elevation = np.sqrt(yy**2 + xx**2)
elevation = (1 - elevation / elevation.max())

distance = np.sqrt((yy - yy[H//2, W//2])**2 + (xx - xx[H//2, W//2])**2)
distance_norm = 1 - (distance / distance.max())

frequency_log10 = np.ones((H, W)) * np.log10(2.4)
frequency_log10 = (frequency_log10 - np.log10(1)) / (np.log10(6) - np.log10(1))

cond = np.stack([elevation, distance_norm, frequency_log10], axis=0)
cond = np.repeat(cond[np.newaxis, ...], batch_size, axis=0)
cond_tensor = torch.from_numpy(cond).float().to(device)

print(f"Conditioning shape: {cond_tensor.shape}")

# Diffusion sampling
from torch import nn
import math

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
        with torch.no_grad():
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
        
        for i, t in enumerate(timesteps):
            x = self.p_sample(x, int(t), cond_img)
            if (i % max(1, steps // 10)) == 0:
                print(f"  Step {i+1}/{len(timesteps)}")
        
        return x

sampler = DiffusionSampler(model, device=device)

print("🎲 Sampling RSS maps (50 steps)...")
with torch.no_grad():
    rss_pred = sampler.sample(cond_tensor, shape=(batch_size, 1, 107, 107), steps=50)

RSS_FLOOR, RSS_SCALE = -100.0, 50.0
rss_dbm = rss_pred.cpu().numpy() * RSS_SCALE + RSS_FLOOR

print(f"✓ Complete!")
print(f"  Shape: {rss_dbm.shape}")
print(f"  Range: [{rss_dbm.min():.1f}, {rss_dbm.max():.1f}] dBm")
```

**Cell 5: Visualize & Save**
```python
# Visualize
fig, axes = plt.subplots(batch_size, 2, figsize=(12, 5*batch_size))
if batch_size == 1:
    axes = axes.reshape(1, -1)

for i in range(batch_size):
    # Predicted RSS
    ax = axes[i, 0]
    im = ax.imshow(rss_dbm[i, 0], cmap='RdYlBu_r')
    ax.set_title(f'Sample {i+1}: RSS Map')
    plt.colorbar(im, ax=ax, label='dBm')
    
    # Elevation
    ax = axes[i, 1]
    im = ax.imshow(cond_tensor[i, 0].cpu().numpy(), cmap='gray')
    ax.set_title(f'Sample {i+1}: Elevation')
    plt.colorbar(im, ax=ax, label='norm')

plt.tight_layout()
plt.savefig('/content/rss_results.png', dpi=100, bbox_inches='tight')
print("✓ Saved to rss_results.png")
plt.show()

# Save results
np.save('/content/drive/MyDrive/rss_predictions.npy', rss_dbm)
print("✓ Saved predictions to Google Drive")
```

---

## Key Points

| Parameter | Value | Note |
|-----------|-------|------|
| Input shape | (B, 3, 107, 107) | Batch × 3 channels × 107×107 grid |
| Output shape | (B, 1, 107, 107) | RSS predictions |
| Channel 0 | Elevation | 0-1, scene geometry |
| Channel 1 | Distance norm | 0-1, distance from TX |
| Channel 2 | Frequency log10 | 0-1, normalized frequency |
| Sampling steps | 20-100 | More = better quality, slower |
| Device | CPU (Colab) | No GPU needed |

## Time Estimates

| Batch | Steps | CPU (Colab) |
|-------|-------|-----------|
| 1 | 20 | ~15s |
| 1 | 50 | ~40s |
| 2 | 20 | ~25s |
| 2 | 50 | ~60s |

## Troubleshooting

**Q: "ModuleNotFoundError: No module named 'torch'"**
- A: Run the install cell first: `!pip install torch -q`

**Q: "File not found" for model**
- A: Check the path in `MODEL_PATH` matches your Google Drive location

**Q: Very slow on Colab CPU**
- A: Reduce `steps` to 20 for faster preview inference

**Q: "CUDA out of memory"**
- A: You're on CPU, shouldn't happen. Try reducing `batch_size`.

**Q: Model loading fails with shape error**
- A: Verify checkpoint has 'model_state' key:
```python
checkpoint = torch.load(MODEL_PATH)
print(checkpoint.keys())
```

---

## Next Steps

1. **Use real data**: Replace synthetic conditioning with actual elevation maps + transmitter locations
2. **Batch inference**: Increase `batch_size` to 4-8 for faster throughput
3. **Fine-tune**: Adjust sampling `steps` based on quality/speed tradeoff
4. **Post-process**: Apply smoothing or upsampling to predictions if needed

---

**Files Created**:
- `COLAB_INFERENCE_GUIDE.md` - Detailed walkthrough with explanations
- `colab_inference_standalone.py` - Complete standalone script
- `COLAB_QUICK_REFERENCE.md` - This file
