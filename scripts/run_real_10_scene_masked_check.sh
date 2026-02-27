#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

# Required on this macOS setup for Sionna/DrJit runtime resolution.
export DRJIT_LIBLLVM_PATH="${DRJIT_LIBLLVM_PATH:-/opt/anaconda3/lib/libLLVM-14.dylib}"

echo "[1/2] Regenerating model input/target/mask tensors..."
(cd model_input && "$PYTHON_BIN" model_input.py)

if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import torch
PY
then
    echo "ERROR: PyTorch is not available in the active Python environment."
    echo "Use a Python env with torch installed, or run:"
    echo "  pip install torch"
    echo "You can also select a custom interpreter with:"
    echo "  PYTHON_BIN=/path/to/python ./scripts/run_real_10_scene_masked_check.sh"
    exit 1
fi

echo "[2/2] Running masked 10-scene sanity check..."
"$PYTHON_BIN" - <<'PY'
import json
import torch
from torch.utils.data import DataLoader, Subset
from models.diffusion import RadioMapDataset, TimeCondUNet, Diffusion

input_dir = 'model_input/data/training/input'
target_dir = 'model_input/data/training/target'
mask_dir = 'model_input/data/training/masks'

dataset = RadioMapDataset(input_dir, target_dir, mask_dir=mask_dir)
if len(dataset) < 10:
    raise RuntimeError(f"Need at least 10 scenes, found {len(dataset)}")

subset = Subset(dataset, list(range(10)))
loader = DataLoader(subset, batch_size=2, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = TimeCondUNet(
    in_ch=1,
    cond_channels=2,
    base_ch=32,
    channel_mults=(1, 2, 4),
    num_res_blocks=2,
    time_emb_dim=128,
    cond_emb_dim=64,
).to(device)
diffusion = Diffusion(model, timesteps=1000, device=device)

losses = []
valid_fractions = []

for batch in loader:
    target = batch['target'].to(device)
    cond = batch['input'].to(device)
    boolean_mask = batch['boolean_mask'].to(device)
    rss_null_mask = batch['rss_null_mask'].to(device)

    valid_mask = ((boolean_mask == 0) & (rss_null_mask == 0)).float()
    valid_bool = valid_mask > 0.5

    if valid_bool.any() and not torch.isfinite(target[valid_bool]).all():
        raise AssertionError('valid region still includes non-finite target values')

    t = torch.randint(0, 1000, (target.shape[0],), device=device).long()
    loss = diffusion.p_losses(target, cond, t, valid_mask=valid_mask)
    if not torch.isfinite(loss):
        raise AssertionError('loss became non-finite')

    losses.append(float(loss.item()))
    valid_fractions.append(float(valid_mask.mean().item()))

print(json.dumps({
    'checked_scenes': 10,
    'num_batches': len(losses),
    'loss_min': min(losses),
    'loss_max': max(losses),
    'all_finite_losses': all(torch.isfinite(torch.tensor(losses)).tolist()),
    'valid_fraction_mean': sum(valid_fractions) / len(valid_fractions),
}, indent=2))
PY

echo "Done. DRJIT_LIBLLVM_PATH=$DRJIT_LIBLLVM_PATH"
