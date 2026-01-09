import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. Tiny UNet model for diffusion (minimal but functional)
# ------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class TinyUNet(nn.Module):
    def __init__(self, in_channels=2, base=32):
        super().__init__()

        # ---- Encoder ----
        self.enc1 = ConvBlock(in_channels + 1, base)      # 2 input channels + timestep
        self.enc2 = ConvBlock(base, base * 2)

        # ---- Downsampling ----
        self.down = nn.MaxPool2d(2)

        # ---- Bottleneck ----
        self.bottleneck = ConvBlock(base * 2, base * 4)

        # ---- Upsampling ----
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        # ---- Decoder ----
        # concat: bottleneck_up (base*4) + enc2 (base*2)
        self.dec2 = ConvBlock(base * 4 + base * 2, base * 2)

        # concat: dec2_up (base*2) + enc1 (base)
        self.dec1 = ConvBlock(base * 2 + base, base)

        self.final = nn.Conv2d(base, in_channels, 1)

    def forward(self, x, t):

        # time embedding
        t_embed = (t.float() / 1000.0).view(-1, 1, 1, 1)
        t_embed = t_embed.expand(x.shape[0], 1, x.shape[2], x.shape[3])

        xt = torch.cat([x, t_embed], dim=1)

        # ---- Encoder ----
        e1 = self.enc1(xt)             # [B, 32, H, W]
        e2 = self.enc2(self.down(e1))  # [B, 64, H/2, W/2]

        # ---- Bottleneck ----
        b = self.bottleneck(self.down(e2))  # [B, 128, H/4, W/4]

        # ---- Upsample ----
        d2 = self.up(b)                      # [B, 128, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)      # [B, 128+64 = 192, H/2, W/2]
        d2 = self.dec2(d2)                   # [B, 64, H/2, W/2]

        d1 = self.up(d2)                     # [B, 64, H, W]
        d1 = torch.cat([d1, e1], dim=1)      # [B, 64+32 = 96, H, W]
        d1 = self.dec1(d1)                   # [B, 32, H, W]

        return self.final(d1)                # [B, 2, H, W]


# ------------------------------------------------------------
# 2. Dummy input: elevation map + frequency channel
# ------------------------------------------------------------

def make_dummy_input(size=64):
    elev = torch.randn(1, 1, size, size)       # elevation map (khushi can generate )
    freq = torch.ones(1, 1, size, size) * 2.4  # constant 2.4 GHz (distance / wavelength instead of pure freq, normalize this)

    ##can add other channels if needed
    #materials
    #antenna (directionallity not important)
    #max v/wavelength should be stated  
    return torch.cat([elev, freq], dim=1)      # shape: (1, 2, H, W)

# ------------------------------------------------------------
# 3. Simple sampler (20 diffusion steps)
# ------------------------------------------------------------

@torch.no_grad()
def sample(model, steps=20, shape=(1, 2, 64, 64)):
    x = torch.randn(shape)  # Start from pure noise

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t])
        noise_pred = model(x, t_tensor)

        # Simple Euler update (not accurate)
        x = x - noise_pred * 0.1

    return x

# ------------------------------------------------------------
# 4. Main demo
# ------------------------------------------------------------

def main():
    # Create model
    model = TinyUNet()

    # Create dummy input (not actually used in sampling—just showing how you'd use it)
    dummy_input = make_dummy_input(64)

    # Run sampler (generates fake RSS map)
    out = sample(model, steps=20, shape=dummy_input.shape)

    # Take RSS map from channel 0
    rss = out[0, 0].cpu() 

    print("Output shape:", rss.shape)

    # Save result to current directory
    img = rss.cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(img, cmap="viridis")
    plt.colorbar()
    plt.title("RSS Output (Untrained Model)")
    fname = "rss_output.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Saved output image to {fname}")

if __name__ == "__main__":
    main()
