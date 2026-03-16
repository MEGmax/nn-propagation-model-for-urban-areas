# Recipe: Extend the Single-Scene Training Run to a Larger Dataset

## Goal
Take the current single-scene workflow and scale it to a multi-scene dataset while preserving the corrected contract:

- target = `path_loss_db`
- conditioning = `elevation`, `electrical_distance`
- one shared normalization stats file
- no frequency channel

## Canonical Data Contract

### Raw scene folder requirements
Each scene folder should contain:

- `sceneX.xml`
- `meshes/*.ply`
- `elevation.npy`
- `pathloss_values*.npy`
- `tx_metadata.json`

### Processed tensor outputs
For each valid scene:

- `sceneX_input.npy` with shape `(H, W, 2)`
- `sceneX_target.npy` with shape `(H, W, 1)`

Channel order for `sceneX_input.npy`:

1. `elevation`
2. `electrical_distance`

Target channel for `sceneX_target.npy`:

1. normalized `path_loss_db`

## Preconditions

### 1. Keep frequency fixed across the full dataset
Because frequency is not a conditioning input anymore, it must not vary across scenes.

The active code expects a single fixed value in:

- [configs/config.toml](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/configs/config.toml)

### 2. Ensure every scene has valid radio-map metadata
The preprocessing step uses:

- radio-map center
- radio-map cell size
- transmitter position

These are written by:

- [scene_generation/load_sionna_scene.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/scene_generation/load_sionna_scene.py)

If scenes are imported from elsewhere, make sure `tx_metadata.json` matches this structure.

## Step-by-Step Workflow

## Step 1. Add or generate more scenes
Place each scene under:

- [scene_generation/automated_scenes](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/scene_generation/automated_scenes)

Recommended folder layout:

- `scene0/`
- `scene1/`
- `scene2/`
- ...

Avoid empty stub folders. The safest pattern is to only create the folder once the XML and mesh files are actually present.

## Step 2. Generate elevation maps
Run:

```bash
python scene_generation/2d_elevation_map.py
```

Expected result per scene:

- `elevation.npy`
- `elevation.png`

## Step 3. Generate path-loss maps with Sionna
Run on the actual machine, outside the sandbox if needed:

```bash
export DRJIT_CACHE_PATH=/tmp
python scene_generation/load_sionna_scene.py
```

Expected result per scene:

- `pathloss_values_*.npy`
- `pathloss_render_*.png`
- `tx_metadata.json`

If LLVM lookup fails, point Dr.Jit at the correct 64-bit LLVM library on the machine.

## Step 4. Split scenes into train / validation / test roots
Do not compute normalization stats from all data together. Compute them from the training split only.

Recommended directory structure:

```text
dataset_scenes/
  train/
    scene0/
    scene1/
    ...
  val/
    sceneA/
    sceneB/
  test/
    sceneM/
    sceneN/
```

You can create these either by:

- copying scene folders into split directories, or
- creating split manifests and teaching preprocessing to read them later

For now, copying is simpler and less error-prone.

## Step 5. Build tensors for the training split
Run:

```bash
python model_input/model_input.py \
  --scenes-root /path/to/dataset_scenes/train \
  --output-input model_input/data/training/input \
  --output-target model_input/data/training/target \
  --stats-file model_input/data/training/normalization_stats.json
```

This step:

- builds the training tensors
- computes `normalization_stats.json`

That stats file becomes the single source of truth for all later stages.

## Step 6. Build tensors for validation and test using the same stats
Current preprocessing code computes stats from the supplied root. For a proper larger-dataset workflow, the next practical extension should be:

- add a `--use-existing-stats` mode to [model_input/model_input.py](/home/saeed/vault/probe/nn-propagation-model-for-urban-areas/model_input/model_input.py)

Until that is implemented, do not treat validation/test metrics as authoritative if they were normalized with independently computed stats.

Recommended next code change:

1. add `--stats-file-in`
2. if provided, load existing training stats instead of recomputing them
3. write tensors for validation/test with those same stats

That should be the first follow-up before any serious multi-scene experiment.

## Step 7. Train on the larger dataset
Run:

```bash
cd models
python model.py \
  --input-dir ../model_input/data/training/input \
  --target-dir ../model_input/data/training/target \
  --stats-file ../model_input/data/training/normalization_stats.json \
  --checkpoint-dir ../artifacts/multiscene/checkpoints \
  --epochs 100 \
  --batch-size 4 \
  --timesteps 1000
```

Suggested starting point:

- `batch-size 4`
- `timesteps 1000`
- `epochs 100`

If the dataset is still small, it is reasonable to start with fewer timesteps for faster iteration, but keep evaluation conditions recorded clearly.

## Step 8. Evaluate and visualize using the same stats
Backtest:

```bash
python backtest/run_backtest.py \
  --checkpoint ../artifacts/multiscene/checkpoints/model_final.pt \
  --input-dir ../model_input/data/test/input \
  --target-dir ../model_input/data/test/target \
  --stats-file ../model_input/data/training/normalization_stats.json
```

Visualize:

```bash
python visualize/visualize_rss_maps.py \
  --checkpoint artifacts/multiscene/checkpoints/model_final.pt \
  --input-dir model_input/data/test/input \
  --target-dir model_input/data/test/target \
  --stats-file model_input/data/training/normalization_stats.json
```

Note:

- the visualization script name is still legacy, but the active implementation now visualizes path loss

## Practical Recommendations for Scaling

## Scene count
Move from `1` scene to at least:

- `20-50` scenes for pipeline debugging
- `100+` scenes for meaningful early learning curves
- more beyond that if you want stable generalization

## Geometry diversity
Increase variation in:

- building count
- footprint density
- height distribution
- street openness / blockage

If all scenes are too similar, the model will overfit the generator’s template instead of learning useful propagation behavior.

## Quality control checks per scene
For each new scene, inspect:

- `elevation.png`
- `pathloss_render_*.png`
- `tx_metadata.json`

Reject scenes with:

- broken geometry export
- path-loss maps with obvious corruption
- missing metadata

## Versioning
Version the dataset and stats together. A simple pattern is:

- `artifacts/dataset_v1/`
- `artifacts/dataset_v2/`

and keep:

- processed tensors
- normalization stats
- checkpoints
- evaluation outputs

under the same version root.

## Immediate Follow-Up Improvements

1. Add `--use-existing-stats` support to preprocessing.
2. Add split creation tooling.
3. Add scene-folder validation and skipping logic.
4. Add a compact manifest file that records:
   - scene ids
   - split membership
   - frequency
   - tensor resolution
   - normalization stats path

## Minimal Safe Scaling Recipe
If you want the shortest path from today’s single-scene run to a real dataset:

1. create `train/`, `val/`, `test/` scene roots
2. generate elevation + path loss for each scene
3. preprocess the training split and save stats
4. extend preprocessing to reuse those training stats for `val` and `test`
5. train only after that
6. keep all outputs versioned

That is the smallest workflow that preserves the normalization consistency we just fixed.
