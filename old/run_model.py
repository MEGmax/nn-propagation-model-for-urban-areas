import torch

dummy_input = torch.randn(1, 4, 32, 32)   # elevation + freq
model = YourDiffusionUNet().eval()

out = model(dummy_input, t=torch.tensor([10]))
print(out.shape)

######

import torch.nn.functional as F

# fake 64x64 elevation map
dummy_elev = torch.randn(1, 1, 64, 64)

# fake frequency channel (broadcasted)
dummy_freq = torch.ones(1, 1, 64, 64) * 2.4  # e.g. 2.4 GHz

x0 = torch.cat([dummy_elev, dummy_freq], dim=1)  # shape: (1, 2, 64, 64)

import torch

def sample(model, steps=30, shape=(1, 2, 64, 64)):
    model.eval()
    x = torch.randn(shape)  # start from pure noise

    for t in reversed(range(steps)):
        t_tensor = torch.tensor([t])
        with torch.no_grad():
            noise_pred = model(x, t_tensor)

        # simple Euler step (not accurate but works for demo)
        x = x - noise_pred * 0.1

    return x

