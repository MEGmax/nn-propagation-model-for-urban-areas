# Path-Loss Pipeline Remediation Changelog

## Purpose
This document summarizes the changes made to stabilize the active training and inference path around a single, explicit data contract:

- Target: `path_loss_db`
- Conditioning channels: `elevation`, `electrical_distance`
- Denoiser input at each diffusion step: `x_t + 2 conditioning channels`
- Normalization: linear only, no `tanh` / `arctanh`

## What Changed

### 1. Unified the active data contract
The active code path no longer mixes RSS and path loss.

- All active preprocessing now uses `path_loss_db` as the target.
- All active conditioning now uses exactly 2 channels:
  - `elevation`
  - `electrical_distance`
- Frequency is no longer a model input.

## 2. Added a shared normalization module
New file:

- [common/pathloss.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/common/pathloss.py)

This module now owns:

- normalization stats schema
- serialization/deserialization of stats JSON
- affine normalization/denormalization for path loss
- linear normalization for elevation
- linear normalization for electrical distance

This replaces the previous situation where preprocessing, inference, backtesting, and visualization each used different assumptions.

## 3. Refactored preprocessing to emit one canonical tensor format
Updated file:

- [model_input/model_input.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/model_input/model_input.py)

Behavior now:

- reads `elevation*.npy`
- reads `pathloss_values*.npy`
- reads `tx_metadata.json`
- computes `electrical_distance` from fixed frequency and radio-map metadata
- writes:
  - `*_input.npy` with shape `(H, W, 2)`
  - `*_target.npy` with shape `(H, W, 1)`
- writes one shared `normalization_stats.json`

Important:

- preprocessing no longer runs automatically on import
- preprocessing now fails fast when required scene artifacts are missing
- preprocessing requires fixed frequency across scenes if frequency is not provided to the model

## 4. Removed the old pooled conditioning path from the U-Net
Updated file:

- [models/diffusion.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/models/diffusion.py)

Previous behavior:

- conditioning image was projected to a global vector using adaptive average pooling
- local spatial conditioning was discarded before most of the network saw it

Current behavior:

- the denoiser concatenates `x_t` with `cond_img` directly
- the first convolution sees 3 total channels:
  - `1` noisy path-loss channel
  - `2` conditioning channels
- conditioning is present on every denoising call

This is the most important architectural correction in the repo.

## 5. Updated training to validate the contract
Updated file:

- [models/model.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/models/model.py)

Training now:

- loads the shared stats JSON
- validates dataset channel counts
- builds the model with the dataset’s conditioning layout
- saves checkpoint metadata that includes:
  - model config
  - normalization stats

## 6. Updated inference, backtesting, and visualization
Updated files:

- [models/inference.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/models/inference.py)
- [backtest/backtest_evaluation.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/backtest/backtest_evaluation.py)
- [backtest/run_backtest.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/backtest/run_backtest.py)
- [visualize/visualize_rss_maps.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/visualize/visualize_rss_maps.py)

These now:

- read the shared stats JSON or checkpoint-embedded stats
- denormalize with the same affine inverse used by preprocessing
- stop guessing normalization from raw value ranges
- use path-loss terminology consistently

## 7. Updated Sionna scene generation metadata
Updated file:

- [scene_generation/load_sionna_scene.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/scene_generation/load_sionna_scene.py)

The Sionna path-loss stage now:

- uses a fixed frequency from config
- writes `tx_metadata.json` with:
  - transmitter position
  - transmitter orientation
  - radio-map center
  - radio-map cell size
  - radio-map size

This metadata is required to compute electrical distance correctly during feature construction.

## 8. Replaced the duplicate tensor builder
Updated file:

- [tmp_build_tensors.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/tmp_build_tensors.py)

It now forwards to the canonical preprocessing entrypoint instead of maintaining a second incompatible feature pipeline.

## 9. Added contract tests
New file:

- [tests/test_pathloss_contract.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/tests/test_pathloss_contract.py)

Current tests cover:

- normalization round-trip
- dataset shape enforcement
- denoiser forward-pass shape contract

## Validation Performed

The following checks were run after the refactor:

- `python -m compileall common model_input models backtest visualize tests scene_generation`
- `python -m unittest discover -s tests`

Both passed.

## Single-Scene Smoke / Overfit Results
Artifacts were generated for `scene1` under:

- [artifacts/smoke_scene1](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/artifacts/smoke_scene1)
- [artifacts/smoke_scene1_overfit](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/artifacts/smoke_scene1_overfit)
- [artifacts/smoke_scene1_overfit_t20](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/artifacts/smoke_scene1_overfit_t20)

Best single-scene diffusion result achieved in this pass:

- `timesteps=100`
- `epochs=1500`
- RMSE about `15.39 dB`

This is better than the initial smoke run, but still not true memorization.

## Known Remaining Issues

### Diffusion overfit quality is still weaker than expected
Even after cleanup, the current epsilon-prediction DDPM setup does not perfectly memorize a single scene. The active code is now internally consistent, but that does not guarantee that the current objective/sampler pair is ideal for tiny-data overfitting.

### Old documentation is still stale
Several files under `docs/auto_generated/`, older notebooks, and some legacy testing artifacts still mention:

- RSS instead of path loss
- 3 conditioning channels
- previous normalization schemes

Those were not fully rewritten in this pass.

### Empty or partial scene folders still need filtering
The active preprocessing assumes each scene directory is valid. Empty directories like a stub `scene0/` can still cause preprocessing failure if included in the root scenes directory.

## Recommended Next Cleanup

1. Update README and selected docs to reflect the new contract.
2. Add explicit train/validation/test split tooling.
3. Make preprocessing skip invalid scene folders with a clear warning instead of failing the full run.
4. If single-scene memorization is still important, test:
   - `x0` prediction instead of epsilon prediction
   - deterministic sampling
   - a direct conditional regression baseline for sanity checking
