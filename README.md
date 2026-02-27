# Neural-Network Powered Tool for Real-Time Radio Coverage Analysis of Urban Areas

## Motivation:
Tools for rendering accurate models of radio wave coverage maps, such as Ray-tracing, are time-consuming and expensive. This project aims to use recent progress in machine learning based radio wave propagation modeling to train a real-time, generalizable, radio wave propagation neural network.

## Description
This project aims to develop a machine learning-assisted system for optimizing the placement of outdoor wireless transmitters. Data will be generated using ray tracing simulations in urban environments to create received signal strength (RSS) and path loss maps for candidate transmitter positions. A neural network will be trained to accurately predict radio propagation with input features relating to the urban environment. An optimization algorithm will then evaluate predicted coverage to select transmitter locations to maximize performance while minimizing deployment costs. 


## Setup Instructions
Run ```source ./venv/bin/activate``` to activate the virtual environment.

Then install the required packages using:
```
pip install -r requirements.txt
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
- `automated_scenes/`: Contains rss_maps and elevation maps, along with misc scene files
- `models/`: Model definition and training scripts
- `backtesting/`: Scripts to evaluate model performance against ray-traced ground truth
- `visualize/`: Scripts to visualize model predictions and errors
- `docs/`: Documentation and quick reference guides


## Data Generation
To generate a complete dataset (scenes, elevation maps, and radio maps), you can run the automated pipeline script:

```bash
python generate_data.py --num-scenes 5
```

This script sequentially executes the following steps:
1. **Scene Generation**: Runs `scene_generation/studio_setup.py` in Blender to create 3D city models and export them as XML.
2. **Elevation Mapping**: Runs `scene_generation/2d_elevation_map.py` to compute 2D normalized elevation maps from the building meshes.
3. **Radio Map Tracing**: Runs `scene_generation/load_sionna_scene.py` to simulate radio propagation using Sionna ray tracing and output the ground truth RSS maps.
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
   python model_input.py
   ```

### Output Structure
All generated data is improved in `scene_generation/automated_scenes/` with the following structure per scene:

##### Data Model
- elevation.npy : elevation map of the scene
- rss_values.npy : ray-traced received signal strength map
- tx_metadata.json : transmitter position and orientation
##### Supporting Files
- meshes folder: contains building meshes in PLY format
- scene.xml : mitsuba scene file used for ray-tracing

##### tx_metadata.json Format
{
 "frequency": np.array([config.FREQUENCY_HZ]),
 "tx_position": [np.array([x, y, z])],
 "tx_orientation": [np.array([azimuth, elevation])]
}

# Creating, Evaluating, and Inferencing a Model Instance:

## 1. Prepare your data
Make sure you have generated the .npy files for your scenes using `load_sionna_scene.py` and that they are located in `automated_scenes/sceneX/` where X is the scene number (0-N). Each scene folder should contain:
- `elevation.npy`
- `rss_values.npy`
- `tx_metadata.json`

## 2. Instantiate new model and train 
Using `models/model.py` (saves checkpoint in `models/checkpoints/` which is used for evaluation and inference) 
Run one of the following commands:

```bash
# Basic training
python model.py --epochs 50 --batch-size 4

# Or with custom settings
python model.py \
    --epochs 100 \
    --batch-size 8 \
    --lr 2e-4 \
    --save-every 10

# Resume interrupted training
python model.py --resume --epochs 100
```

### 3. Run Backtesting Evaluation 
Using `run_backtest.py` (saves results in `backtest_results/`)

Run one of the following commands:

```bash
# Runs on 5 scenes with 20 diffusion steps (fast, good for debugging)
python run_backtest.py --quick 

# Runs on all 20 scenes with 50 diffusion steps (slow, good for final evaluation)
python run_backtest.py \
    --checkpoint ../models/checkpoints/model_final.pt \
    --samples-per-scene 5 \
    --diffusion-steps 50 
```

### 4. Inference and Visualize RSS Maps 
Using inference.py (saves visualizations in `rss_visualizations/`)

Run one of the following commands (last is easiest for minimal setup):
```bash
# Run quick inference on 1 scene with 20 diffusion steps (fast, good for debugging)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Run inference on 5 scenes with 50 diffusion steps (slow, good for final evaluation)
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50

#run the inference script directly
python inference.py --checkpoint ../models/checkpoints/model_final.pt --scene scene0
```

### Contributors
- Ahmet Hamamcioglu
- Khushi Patel
- Matthew Henriquet
- Matthew Grech