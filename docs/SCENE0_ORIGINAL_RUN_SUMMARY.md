# Scene0 Original-Architecture Run Summary

## Purpose

This document summarizes the final changes made during the `scene0` single-scene training pass and records the exact configuration that produced a working result.

The final decision was:

- keep the original diffusion architecture and training behavior
- add consistency plumbing so training, inference, visualization, and backtest can use the same diffusion `timesteps`
- preserve the preprocessing improvements that produced stable normalized tensors for `scene0`

## Final Code Changes

### 1. Restored the original diffusion model and training path

Final behavior in:

- [models/diffusion.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/models/diffusion.py)
- [models/model.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/models/model.py)

What this means:

- original `TimeCondUNet` configuration retained
- original linear beta schedule retained
- original DDPM training loop retained
- no experimental dropout
- no data augmentation
- no AdamW / weight decay / grad clipping additions
- no cosine schedule path

### 2. Added checkpointed training config for reproducibility

Updated files:

- [models/model.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/models/model.py)
- [models/diffusion.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/models/diffusion.py)

Training now stores a `training_config` dictionary in each checkpoint payload. The saved fields are:

- `epochs`
- `batch_size`
- `lr`
- `timesteps`
- `save_every`
- `num_workers`
- `device`
- `input_dir`
- `target_dir`
- `stats_file`

### 3. Made inference use the same timesteps as training

Updated file:

- [models/inference.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/models/inference.py)

Behavior now:

- supports `--timesteps`
- if `--timesteps` is omitted, reads `timesteps` from checkpoint `training_config`

### 4. Made backtest use the same timesteps as training

Updated files:

- [backtest/run_backtest.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/backtest/run_backtest.py)
- [backtest/backtest_evaluation.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/backtest/backtest_evaluation.py)

Behavior now:

- supports `--timesteps`
- if `--timesteps` is omitted, reads `timesteps` from checkpoint `training_config`
- records `timesteps` in `backtest_results.json`

### 5. Fixed visualization timesteps consistency

Updated file:

- [visualize/visualize_rss_maps.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/visualize/visualize_rss_maps.py)

Behavior now:

- no longer hardcodes `timesteps=1000`
- supports `--timesteps`
- defaults to checkpoint `training_config["timesteps"]` when available

### 6. Kept the preprocessing / normalization improvements

Updated files:

- [common/pathloss.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/common/pathloss.py)
- [model_input/model_input.py](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/model_input/model_input.py)

What remains in place:

- preprocessing can operate directly on a single scene directory like `scene_generation/scene0`
- non-scene directories can be skipped when preprocessing a root folder
- stats now include feature means/stds in addition to maxima
- normalized `scene0` input channels are centered near zero with unit-scale variance

## Final Scene0 Training Run

Dataset artifacts:

- [artifacts/scene0_originalrun/data/input/scene0_input.npy](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/data/input/scene0_input.npy)
- [artifacts/scene0_originalrun/data/target/scene0_target.npy](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/data/target/scene0_target.npy)
- [artifacts/scene0_originalrun/data/normalization_stats.json](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/data/normalization_stats.json)

Checkpoint artifacts:

- [artifacts/scene0_originalrun/checkpoints/model_final.pt](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/checkpoints/model_final.pt)

Training hyperparameters:

- `epochs=1500`
- `batch_size=1`
- `lr=2e-4`
- `timesteps=100`
- `save_every=100`
- `num_workers=0`
- device: `cuda`
- model config:
  - `base_ch=32`
  - `channel_mults=(1, 2, 4)`
  - `num_res_blocks=2`
  - `time_emb_dim=128`
  - `out_ch=1`

## Final Scene0 Results

Backtest output:

- [artifacts/scene0_originalrun/backtest/backtest_results.json](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/backtest/backtest_results.json)
- [artifacts/scene0_originalrun/backtest/backtest_metrics.png](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/backtest/backtest_metrics.png)

Backtest metrics for the working run:

- `RMSE = 14.90 dB`
- `MAE = 12.35 dB`
- `Median error = 11.21 dB`
- `Bias = 0.09 dB`
- `Pearson r = 0.878`

## Visualizations Produced

Comparison figure:

- [artifacts/scene0_originalrun/visuals/scene0_pathloss_comparison.png](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/visuals/scene0_pathloss_comparison.png)

Direct inference outputs:

- [artifacts/scene0_originalrun/inference/scene0_prediction_t100.png](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/inference/scene0_prediction_t100.png)
- [artifacts/scene0_originalrun/inference/scene0_prediction_t100.npy](/home/saeed/vault/cempact/nn-propagation-model-for-urban-areas/artifacts/scene0_originalrun/inference/scene0_prediction_t100.npy)

Visualization run settings:

- `sampling_steps=100`
- `timesteps=100`

## Reproduction Commands

### Preprocess scene0

```bash
python -m model_input.model_input \
  --scenes-root scene_generation/scene0 \
  --output-input artifacts/scene0_originalrun/data/input \
  --output-target artifacts/scene0_originalrun/data/target \
  --stats-file artifacts/scene0_originalrun/data/normalization_stats.json
```

### Train

```bash
python models/model.py \
  --input-dir artifacts/scene0_originalrun/data/input \
  --target-dir artifacts/scene0_originalrun/data/target \
  --stats-file artifacts/scene0_originalrun/data/normalization_stats.json \
  --epochs 1500 \
  --batch-size 1 \
  --lr 2e-4 \
  --timesteps 100 \
  --save-every 100 \
  --checkpoint-dir artifacts/scene0_originalrun/checkpoints \
  --num-workers 0
```

### Backtest

```bash
python backtest/run_backtest.py \
  --checkpoint artifacts/scene0_originalrun/checkpoints/model_final.pt \
  --input-dir artifacts/scene0_originalrun/data/input \
  --target-dir artifacts/scene0_originalrun/data/target \
  --stats-file artifacts/scene0_originalrun/data/normalization_stats.json \
  --output-dir artifacts/scene0_originalrun/backtest \
  --batch-size 1 \
  --samples-per-scene 1 \
  --diffusion-steps 100 \
  --timesteps 100
```

### Visualization

```bash
python visualize/visualize_rss_maps.py \
  --checkpoint artifacts/scene0_originalrun/checkpoints/model_final.pt \
  --input-dir artifacts/scene0_originalrun/data/input \
  --target-dir artifacts/scene0_originalrun/data/target \
  --stats-file artifacts/scene0_originalrun/data/normalization_stats.json \
  --output-dir artifacts/scene0_originalrun/visuals \
  --num-scenes 1 \
  --diffusion-steps 100 \
  --timesteps 100
```

### Direct inference

```bash
python models/inference.py \
  --checkpoint artifacts/scene0_originalrun/checkpoints/model_final.pt \
  --input artifacts/scene0_originalrun/data/input/scene0_input.npy \
  --stats-file artifacts/scene0_originalrun/data/normalization_stats.json \
  --output-dir artifacts/scene0_originalrun/inference \
  --output-name scene0_prediction_t100 \
  --sampling-steps 100 \
  --timesteps 100
```
