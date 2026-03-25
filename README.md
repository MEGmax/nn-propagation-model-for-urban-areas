# Neural-Network Powered Tool for Real-Time Radio Coverage Analysis of Urban Areas

## Motivation:
Tools for rendering accurate models of radio wave coverage maps, such as Ray-tracing, are time-consuming and expensive. This project aims to use recent progress in machine learning based radio wave propagation modeling to train a real-time, generalizable, radio wave propagation neural network.

## Description
This project aims to develop a machine learning-assisted system for optimizing the placement of outdoor wireless transmitters. Data will be generated using ray tracing simulations in urban environments to create received signal strength (RSS) and path loss maps for candidate transmitter positions. A neural network will be trained to accurately predict radio propagation with input features relating to the urban environment. An optimization algorithm will then evaluate predicted coverage to select transmitter locations to maximize performance while minimizing deployment costs. 


## Setup Instructions
Run `source ./venv/bin/activate` to activate the virtual environment.

Then install the required packages using:
```
pip install -r requirments.txt
```
#### load_sionna_scene.py Dependencies
- Python 3.12 (due to Sionna dependency)
Use `pyenv` to manage multiple python versions
- llvm compiler (run `brew install llvm` on MacOS) and set environment:
  - `export LDFLAGS="-L/usr/local/opt/llvm/lib"`
  - `export CPPFLAGS="-I/usr/local/opt/llvm/include"`
  - `export PATH="/usr/local/opt/llvm/bin:$PATH"`

### Repository Structure
- `scene_generation/`: Scripts to generate urban scenes and ray-traced radio maps using Sionna
- `model_input/`: Preprocessing and normalized training tensors
- `models/`: Model definition, training, and inference scripts
- `backtest/`: Scripts to evaluate model performance against ray-traced ground truth
- `visualize/`: Scripts to visualize model predictions and errors
- `docs/`: Documentation and quick reference guides


## Data Generation
To generate a complete dataset (scenes, elevation maps, and path loss maps), you can run the automated pipeline script:

```bash
python generate_data.py --num-scenes 5
```

This script sequentially executes the following steps:
1. **Scene Generation**: Runs `scene_generation/studio_setup.py` in Blender to create 3D city models and export them as XML.
2. **Elevation Mapping**: Runs `scene_generation/2d_elevation_map.py` to compute 2D normalized elevation maps from the building meshes.
3. **Radio Map Tracing**: Runs `scene_generation/load_sionna_scene.py` to simulate radio propagation using Sionna ray tracing and output path loss maps.
4. **Data Formatting**: Runs `model_input/model_input.py` to convert raw scene data into training-ready tensor files.

#### Manual Execution
Alternatively, you can run each step individually:

1. **Generate Scenes (Blender):**
   ```bash
   /Applications/Blender.app/Contents/MacOS/Blender --background --python scene_generation/studio_setup.py -- --num-scenes 5
   ```
   *Note: Adjust the Blender path for your system.*

2. **Generate Elevation Maps:**
   ```bash
   python scene_generation/2d_elevation_map.py
   ```

3. **Generate Radio Maps (Sionna):**
   ```bash
   python scene_generation/load_sionna_scene.py
   ```

4. **Format Data for Training:**
   ```bash
   python model_input/model_input.py
   ```

### Output Structure
All generated data is stored in `scene_generation/automated_scenes/` with the following structure per scene:

##### Data Model
- `elevation.npy`: elevation map of the scene
- `pathloss_values.npy`: ray-traced path loss map
- `tx_metadata.json`: transmitter position and orientation
##### Supporting Files
- `meshes/`: contains building meshes in PLY format
- `scene.xml`: Mitsuba scene file used for ray-tracing

##### tx_metadata.json Format
```json
{
  "frequency": [3500000000.0],
  "tx_position": [x, y, z],
  "tx_orientation": [azimuth, elevation]
}
```

# Creating, Evaluating, and Inferencing a Model Instance:

## 1. Prepare your data
Make sure you have generated the `.npy` files for your scenes and that they are processed into:
- `model_input/data/training/input/sceneX_input.npy`
- `model_input/data/training/target/sceneX_target.npy`
- `model_input/data/training/normalization_stats.json`

The current model expects:
- input tensors with shape `(H, W, 2)` containing normalized elevation and electrical distance
- target tensors with shape `(H, W, 1)` containing normalized path loss

## 2. Instantiate new model and train
Using `models/model.py` (saves checkpoints in `models/checkpoints/`, used for evaluation and inference)
Run one of the following commands:

```bash
cd models

# Basic training
python model.py --epochs 50 --batch-size 4

# Or with custom settings
python model.py \
    --input-dir ../model_input/data/training/input \
    --target-dir ../model_input/data/training/target \
    --stats-file ../model_input/data/training/normalization_stats.json \
    --epochs 100 \
    --batch-size 8 \
    --lr 2e-4 \
    --timesteps 1000 \
    --save-every 10
```

### 3. Run Backtesting Evaluation
Using `backtest/run_backtest.py` (saves results in `backtest_results/` by default)

Run one of the following commands:

```bash
# Runs on fewer samples with fewer diffusion steps (fast, good for debugging)
python backtest/run_backtest.py --quick

# Runs on the dataset with an explicit checkpoint
python backtest/run_backtest.py \
    --checkpoint artifacts/training_run/checkpoints/model_final.pt \
    --input-dir model_input/data/training/input \
    --target-dir model_input/data/training/target \
    --stats-file model_input/data/training/normalization_stats.json \
    --samples-per-scene 1 \
    --diffusion-steps 100 \
    --timesteps 100
```

### 4. Inference and Visualize Path Loss Maps
Using `visualize/visualize_rss_maps.py` for comparison figures and `models/inference.py` for direct inference.

Run one of the following commands:
```bash
# Run visualization on 1 scene
python visualize/visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Run visualization on 5 scenes
python visualize/visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50

# Run the inference script directly on one input tensor
python models/inference.py \
    --checkpoint artifacts/training_run/checkpoints/model_final.pt \
    --input model_input/data/training/input/scene0_input.npy \
    --stats-file model_input/data/training/normalization_stats.json \
    --output-dir artifacts/training_run/inference \
    --output-name scene0_prediction
```

### Contributors
- Ahmet Hamamcioglu
- Khushi Patel
- Matthew Henriquet
- Matthew Grech
