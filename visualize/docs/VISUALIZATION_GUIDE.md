# RSS Map Visualization Guide

## Quick Start

Generate PNG visualizations of RSS map predictions from your model:

### Basic Usage

```bash
# Visualize all 5 training scenes
python visualize_rss_maps.py

# Quick visualization (just 1 scene, faster)
python visualize_rss_maps.py --num-scenes 1

# Customize output directory
python visualize_rss_maps.py --output-dir my_visualizations
```

## What Gets Generated

For each scene, you'll get **3 PNG files**:

### 1. **Comparison Figure** (`scene_name_rss_comparison.png`)
A 2×3 grid showing:
- **Top row**: 
  - Predicted RSS map
  - Ground truth RSS map
  - Error map (red = over-prediction, blue = under-prediction)
- **Bottom row**:
  - Elevation map (conditioning input)
  - Distance map (conditioning input)
  - Frequency value (conditioning input)

**Also displays**: RMSE, MAE, and Bias statistics

### 2. **Predicted Map** (`scene_name_predicted.png`)
- Single heatmap of model's predicted RSS
- Shows statistics: mean, std, min, max

### 3. **Ground Truth Map** (`scene_name_groundtruth.png`)
- Single heatmap of actual/target RSS
- Shows statistics for comparison

## Command Options

```bash
# Basic options
--num-scenes N                  # Number of scenes to visualize (default: all 5)
--diffusion-steps N             # Number of reverse diffusion steps (default: 50)
--checkpoint PATH               # Custom model checkpoint
--output-dir PATH               # Output directory (default: rss_visualizations)
--gpu N                          # GPU device ID (default: 0)
```

### Example Commands

```bash
# Fast visualization (1 scene, 20 steps) - ~30 seconds
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Medium (2 scenes, 50 steps) - ~2 minutes
python visualize_rss_maps.py --num-scenes 2 --diffusion-steps 50

# Full (all 5 scenes, 50 steps) - ~5 minutes
python visualize_rss_maps.py

# Use specific checkpoint
python visualize_rss_maps.py --checkpoint models/checkpoints/model_epoch30.pt --num-scenes 3
```

## Output Directory Structure

```
rss_visualizations/
├── scene0_rss_comparison.png         # 2×3 grid comparison
├── scene0_predicted.png              # Predicted RSS only
├── scene0_groundtruth.png            # Ground truth RSS only
├── scene1_rss_comparison.png
├── scene1_predicted.png
├── scene1_groundtruth.png
├── ... (and so on for each scene)
```

## Interpreting the Visualizations

### Color Scales

**RSS Maps (Viridis)**:
- Dark purple/black = Low RSS (weak signal, ~-100 dBm)
- Yellow/bright = High RSS (strong signal, ~-50 dBm)

**Error Map (RdBu_r, diverging)**:
- Blue = Under-prediction (model predicts too weak)
- White = No error
- Red = Over-prediction (model predicts too strong)

**Elevation (Terrain)**:
- Dark = Low elevation
- Bright = High elevation

**Distance (Hot)**:
- Dark = Near transmitter
- Bright = Far from transmitter

### What to Look For

1. **Good prediction**:
   - Predicted map looks similar to ground truth
   - Error map is mostly white/neutral
   - RMSE < 10 dB
   - Spatial patterns match (peaks/valleys in same locations)

2. **Poor prediction**:
   - Predicted map is uniform/featureless
   - Large red/blue patches in error map
   - RMSE > 20 dB
   - No correlation with ground truth

3. **Systematic bias**:
   - Entire error map is red (over-predict) or blue (under-predict)
   - Bias value far from 0 dB

## Statistics Explained

For each scene, summary statistics are printed:

```
SCENE                    RMSE (dB)       MAE (dB)        Bias (dB)
scene0                   18.50           14.20           2.10
scene1                   19.30           15.10           -1.50
AVERAGE                  18.90           14.65           0.30
```

- **RMSE**: Root mean squared error (penalizes large errors more)
- **MAE**: Mean absolute error (average error magnitude)
- **Bias**: Mean error (positive = over-prediction, negative = under-prediction)

## Tips for Best Visualizations

1. **Train model first** (if not done):
   ```bash
   cd models
   python model.py --epochs 20 --batch-size 2
   cd ..
   ```

2. **Use 50+ diffusion steps** for good quality (default: 50)
   - 20 steps = faster, lower quality
   - 50 steps = balanced
   - 100+ steps = best quality but slow

3. **View images in order**:
   - First look at comparison figure
   - Then look at individual predicted vs ground truth
   - Check if patterns match

4. **Check multiple scenes** to see consistency:
   - If all scenes show poor prediction, model needs more training
   - If some scenes are good and others bad, dataset diversity issue

## Troubleshooting

### "ERROR: Checkpoint not found"
- Make sure you've trained the model: `python models/model.py`
- Or use custom checkpoint: `--checkpoint models/checkpoints/model_epoch10.pt`

### "Out of memory"
- Reduce diffusion steps: `--diffusion-steps 20`
- Visualize fewer scenes: `--num-scenes 1`
- Use GPU with more memory

### Images look uniform/blank
- Model might not be trained yet
- Try training: `python models/model.py --epochs 10`
- Check that conditioning inputs are being used

### Images look very noisy
- Normal for diffusion models with few training samples
- This is expected uncertainty
- Train longer or add more data

## Expected Results by Training Stage

### Untrained Model
- Very random/noisy predictions
- No spatial structure
- RMSE: 30+ dB

### Early Training (5-10 epochs)
- Some spatial structure emerging
- Partially correlates with ground truth
- RMSE: 20-25 dB

### Well-Trained Model (20+ epochs)
- Clear spatial structure
- Strong correlation with ground truth
- RMSE: 10-15 dB

### Very Well-Trained (100+ epochs, 1000+ scenes)
- Nearly identical to ground truth
- Clear RSS patterns
- RMSE: < 5 dB (like AIRMap)

## Advanced: Custom Visualization

You can modify `visualize_rss_maps.py` to:
- Change color maps (e.g., `cmap='plasma'` or `cmap='coolwarm'`)
- Add custom overlays
- Generate different subplot layouts
- Export in different formats

See the `create_rss_map_figure()` function for customization options.

---

**Next steps**: Run `python visualize_rss_maps.py --num-scenes 1` and check the output PNG files!
