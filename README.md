# Neural-Network Powered Tool for Real-Time Radio Coverage Analysis of Urban Areas

### Motivation:
Tools for rendering accurate models of radio wave coverage maps, such as Ray-tracing, are time-consuming and expensive. This project aims to use recent progress in machine learning based radio wave propagation modeling to train a real-time, generalizable, radio wave propagation neural network.

### Description
This project aims to develop a machine learning-assisted system for optimizing the placement of outdoor wireless transmitters. Data will be generated using ray tracing simulations in urban environments to create received signal strength (RSS) and path loss maps for candidate transmitter positions. A neural network will be trained to accurately predict radio propagation with input features relating to the urban environment. An optimization algorithm will then evaluate predicted coverage to select transmitter locations to maximize performance while minimizing deployment costs. 

### Contributors
- Ahmet Hamamcioglu
- Khushi Patel
- Matthew Henriquet
- Matthew Grech


### Setup Instructions
run ```source ./venv/bin/activate``` to activate the virtual environment.

Then install the required packages using:
```pip install -r requirements.txt
```

### Repository Structure
- `scene_generation/`: Scripts to generate urban scenes and ray-traced radio maps using Sionna 
- `automated_scenes/`: Contains rss_maps and elevation maps, along with misc scene files

## To Generate Data
- Run `python scene_generation/load_sionna_scene.py` to generate .npy files containing radio maps for all scenes in `automated_scenes/`

### scene_generation/automated_scenes/scene/ Contents
##### Data Model
- elevation.npy : elevation map of the scene
- rss_values.npy : ray-traced received signal strength map
- elevation.npy : elevation map of the scene
- tx_metadata.json : transmitter position and orientation
#### Supporting Files
- meshes folder: contains building meshes in PLY format
- scene.xml : mitsuba scene file used for ray-tracing

#### tx_metadata.json Format
{
 "frequency": np.array([config.FREQUENCY_HZ]),
 "tx_position": [np.array([x, y, z])],
 "tx_orientation": [np.array([azimuth, elevation])]
}

## load_sionna_scene.py Dependencies
- Python 3.12 (due to Sionna dependency)
Use `pyenv` to manage multiple python versions
- llvm compiler (run `brew install llvm` on MacOS) and set environment:
  - `export LDFLAGS="-L/usr/local/opt/llvm/lib"`
  - `export CPPFLAGS="-I/usr/local/opt/llvm/include"`
  - `export PATH="/usr/local/opt/llvm/bin:$PATH"`



