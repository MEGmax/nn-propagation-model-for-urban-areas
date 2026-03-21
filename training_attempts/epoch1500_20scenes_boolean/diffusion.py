from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        exponent = -(math.log(10000) / max(half_dim - 1, 1))
        emb = torch.exp(torch.arange(half_dim, device=device) * exponent)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        self.time_fc = None
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.SiLU()

        if time_emb_dim is not None:
            self.time_fc = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch * 2))

        self.res_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.activation(self.norm1(self.conv1(x)))

        if self.time_fc is not None and t_emb is not None:
            scale, shift = self.time_fc(t_emb).chunk(2, dim=-1)
            h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]

        h = self.activation(self.norm2(self.conv2(h)))
        h = self.dropout(h)
        return h + self.res_conv(x)


class Downsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()
        self.op = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class TimeCondUNet(nn.Module):
    def __init__(
        self,
        noisy_channels: int = 1,
        cond_channels: int = 2,
        base_ch: int = 64,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        out_ch: int = 1,
    ):
        super().__init__()
        self.noisy_channels = noisy_channels
        self.cond_channels = cond_channels
        self.out_ch = out_ch
        self.num_res_blocks = num_res_blocks
        self.base_ch = base_ch
        self.channel_mults = channel_mults
        self.time_emb_dim = time_emb_dim

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        total_in_channels = noisy_channels + cond_channels
        self.input_conv = nn.Conv2d(total_in_channels, base_ch, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        self.res_blocks_down = nn.ModuleList()
        ch = base_ch
        for mult in channel_mults:
            out_channels = base_ch * mult
            for _ in range(num_res_blocks):
                self.res_blocks_down.append(ResBlock(ch, out_channels, time_emb_dim=time_emb_dim))
                ch = out_channels
            self.downs.append(Downsample())

        self.mid_block1 = ResBlock(ch, ch, time_emb_dim=time_emb_dim)
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim=time_emb_dim)

        self.ups = nn.ModuleList()
        self.res_blocks_up = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_channels = base_ch * mult
            self.ups.append(Upsample())
            self.res_blocks_up.append(ResBlock(ch + out_channels, out_channels, time_emb_dim=time_emb_dim))
            for _ in range(num_res_blocks - 1):
                self.res_blocks_up.append(ResBlock(out_channels, out_channels, time_emb_dim=time_emb_dim))
            ch = out_channels

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x_noisy: torch.Tensor, t: torch.Tensor, cond_img: torch.Tensor) -> torch.Tensor:
        if cond_img is None:
            raise ValueError("cond_img is required")
        if x_noisy.shape[0] != cond_img.shape[0] or x_noisy.shape[-2:] != cond_img.shape[-2:]:
            raise ValueError("x_noisy and cond_img must have matching batch and spatial dimensions")
        if x_noisy.shape[1] != self.noisy_channels:
            raise ValueError(
                f"Expected x_noisy to have {self.noisy_channels} channels, got {x_noisy.shape[1]}"
            )
        if cond_img.shape[1] != self.cond_channels:
            raise ValueError(
                f"Expected cond_img to have {self.cond_channels} channels, got {cond_img.shape[1]}"
            )

        t_emb = self.time_emb(t)
        h = self.input_conv(torch.cat([x_noisy, cond_img], dim=1))

        skips: list[torch.Tensor] = []
        down_idx = 0
        for down in self.downs:
            for _ in range(self.num_res_blocks):
                h = self.res_blocks_down[down_idx](h, t_emb)
                down_idx += 1
            skips.append(h)
            h = down(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_block2(h, t_emb)

        up_idx = 0
        for up in self.ups:
            h = up(h)
            skip = skips.pop()

            if h.shape[-2:] != skip.shape[-2:]:
                target_h, target_w = skip.shape[-2:]
                if h.shape[-2] < target_h or h.shape[-1] < target_w:
                    pad_h = target_h - h.shape[-2]
                    pad_w = target_w - h.shape[-1]
                    h = F.pad(h, (0, pad_w, 0, pad_h), mode="reflect")
                else:
                    h = h[:, :, :target_h, :target_w]

            h = torch.cat([h, skip], dim=1)
            for _ in range(self.num_res_blocks):
                h = self.res_blocks_up[up_idx](h, t_emb)
                up_idx += 1

        return self.out_conv(self.out_act(self.out_norm(h)))

    def get_config(self) -> dict:
        return {
            "noisy_channels": self.noisy_channels,
            "cond_channels": self.cond_channels,
            "base_ch": self.base_ch,
            "channel_mults": self.channel_mults,
            "num_res_blocks": self.num_res_blocks,
            "time_emb_dim": self.time_emb_dim,
            "out_ch": self.out_ch,
        }


def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, timesteps)


class Diffusion:
    def __init__(self, model: nn.Module, timesteps: int = 1000, device: str = "cuda"):
        self.model = model
        self.device = device
        self.timesteps = timesteps
        betas = linear_beta_schedule(timesteps).to(device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer = lambda name, value: setattr(self, name, value)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha * x_start + sqrt_one_minus * noise

    def p_losses(
        self,
        x_start: torch.Tensor,
        cond_img: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        pred_noise = self.model(x_noisy, t, cond_img)

        if mask is not None:
            # mask shape: (B, 1, H, W), values 1=keep, 0=exclude
            loss = F.mse_loss(pred_noise * mask, noise * mask, reduction="sum")
            num_valid = mask.sum().clamp(min=1.0)
            return loss / num_valid
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t_index: int, cond_img: torch.Tensor) -> torch.Tensor:
        betas_t = self.betas[t_index]
        sqrt_one_minus_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alpha_t = 1.0 / torch.sqrt(self.alphas[t_index])
        t = torch.full((x.shape[0],), t_index, device=self.device, dtype=torch.long)
        pred_noise = self.model(x, t, cond_img)
        model_mean = sqrt_recip_alpha_t * (x - betas_t / sqrt_one_minus_t * pred_noise)
        if t_index == 0:
            return model_mean
        return model_mean + torch.sqrt(betas_t) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, cond_img: torch.Tensor, shape: Optional[tuple[int, int, int, int]] = None, steps: int = 50) -> torch.Tensor:
        self.model.eval()
        batch_size = cond_img.shape[0]
        if shape is None:
            _, _, height, width = cond_img.shape
            shape = (batch_size, 1, height, width)

        x = torch.randn(shape, device=self.device)
        sample_steps = np.linspace(0, self.timesteps - 1, steps, dtype=int)[::-1]
        for t_index in sample_steps:
            x = self.p_sample(x, int(t_index), cond_img)
        return x


class RadioMapDataset(Dataset):
    def __init__(
        self,
        input_dir: str | os.PathLike[str],
        target_dir: str | os.PathLike[str],
        mask_dir: str | os.PathLike[str] | None = None,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.target_dir = Path(target_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else None
        self.input_files = sorted(self.input_dir.glob("*_input.npy"))
        self.scene_names = [path.stem.replace("_input", "") for path in self.input_files]

    def __len__(self) -> int:
        return len(self.scene_names)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        scene_name = self.scene_names[idx]
        input_tensor = np.load(self.input_dir / f"{scene_name}_input.npy").astype(np.float32)
        target_tensor = np.load(self.target_dir / f"{scene_name}_target.npy").astype(np.float32)

        if input_tensor.ndim != 3 or input_tensor.shape[2] != 2:
            raise ValueError(
                f"Expected input tensor shape (H, W, 2) for {scene_name}, got {input_tensor.shape}"
            )
        if target_tensor.ndim != 3 or target_tensor.shape[2] != 1:
            raise ValueError(
                f"Expected target tensor shape (H, W, 1) for {scene_name}, got {target_tensor.shape}"
            )

        item = {
            "input": torch.from_numpy(input_tensor).permute(2, 0, 1).float(),
            "target": torch.from_numpy(target_tensor).permute(2, 0, 1).float(),
        }

        # Load mask if mask_dir is provided. Mask values of 1 = exclude from loss.
        if self.mask_dir is not None:
            # Try scene-specific mask first, then fall back to a shared mask
            mask_path = self.mask_dir / f"{scene_name}_mask.npy"
            if not mask_path.exists():
                # Try stripping the scene number suffix to find a base mask
                mask_path = self.mask_dir / f"{scene_name.split('_')[0]}_mask.npy"
            if mask_path.exists():
                mask = np.load(mask_path).astype(np.float32)
                if mask.ndim == 2:
                    mask = mask[:, :, np.newaxis]
                # Convert: 1 = exclude → 0 = keep, so we invert for loss weighting
                keep_mask = 1.0 - mask
                item["mask"] = torch.from_numpy(keep_mask).permute(2, 0, 1).float()
            else:
                # No mask found — all pixels contribute
                H, W = target_tensor.shape[:2]
                item["mask"] = torch.ones(1, H, W, dtype=torch.float32)
        else:
            H, W = target_tensor.shape[:2]
            item["mask"] = torch.ones(1, H, W, dtype=torch.float32)

        return item

    def sample_spec(self) -> tuple[int, int]:
        if len(self) == 0:
            raise ValueError("Dataset is empty")
        sample = self[0]
        return sample["input"].shape[0], sample["target"].shape[0]


def save_checkpoint(
    path: Path,
    model: TimeCondUNet,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    normalization_stats: Optional[dict] = None,
    training_config: Optional[dict] = None,
) -> None:
    payload = {
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": model.get_config(),
        "normalization_stats": normalization_stats,
        "training_config": training_config,
    }
    torch.save(payload, path)


def train(
    model: TimeCondUNet,
    dataset: RadioMapDataset,
    device: str = "cuda",
    epochs: int = 20,
    batch_size: int = 8,
    lr: float = 2e-4,
    timesteps: int = 1000,
    save_every: int = 5,
    out_dir: str = "./checkpoints",
    normalization_stats: Optional[dict] = None,
    num_workers: int = 4,
    training_config: Optional[dict] = None,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    model.to(device)
    diffusion = Diffusion(model, timesteps=timesteps, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
    )

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for batch in progress:
            target = batch["target"].to(device)
            cond = batch["input"].to(device)
            mask = batch["mask"].to(device) if "mask" in batch else None
            batch_size_actual = target.shape[0]
            t = torch.randint(0, timesteps, (batch_size_actual,), device=device).long()
            loss = diffusion.p_losses(target, cond, t, mask=mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_size_actual
            progress.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(dataset)
        print(f"Epoch {epoch + 1} average loss: {avg_loss:.6f}")
        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                out_path / f"model_epoch{epoch + 1}.pt",
                model,
                optimizer,
                epoch + 1,
                avg_loss,
                normalization_stats=normalization_stats,
                training_config=training_config,
            )

    save_checkpoint(
        out_path / "model_final.pt",
        model,
        optimizer,
        epochs,
        avg_loss,
        normalization_stats=normalization_stats,
        training_config=training_config,
    )
    return model


@torch.no_grad()
def sample_and_save(
    model: TimeCondUNet,
    cond_img: torch.Tensor,
    device: str = "cuda",
    steps: int = 50,
    out_path: str = "sample.npy",
):
    model.to(device)
    diffusion = Diffusion(model, timesteps=1000, device=device)
    cond_img = cond_img.to(device)
    samples = diffusion.sample(cond_img, shape=(cond_img.shape[0], 1, cond_img.shape[2], cond_img.shape[3]), steps=steps)
    samples_cpu = samples.detach().cpu().numpy()
    np.save(out_path, samples_cpu)
    return samples_cpu