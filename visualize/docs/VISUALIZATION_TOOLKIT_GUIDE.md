# Visualization Toolkit Guide

Complete guide for generating and interpreting RSS map visualizations.

---

## Quick Start

### 1. Generate visualizations (simplest approach)
```bash
# Quick test (1 scene, ~30 seconds)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Standard (3 scenes, ~2 minutes)
python batch_visualize.py --config standard

# All scenes high-quality
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50
```

### 2. View results
```bash
# Look at PNG files generated in rss_visualizations/ directory
# Each scene produces 3 PNGs:
#  - {scene}_rss_comparison.png  (6-panel comparison)
#  - {scene}_predicted.png        (predicted map with stats)
#  - {scene}_groundtruth.png      (ground truth map with stats)
```

---

## Tool Overview

### visualize_rss_maps.py
**Purpose**: Generate RSS map heatmaps with diffusion model predictions

**Command**:
```bash
python visualize_rss_maps.py [OPTIONS]
```

**Options**:
| Option | Default | Description |
|--------|---------|-------------|
| `--checkpoint` | models/checkpoints/model_final.pt | Path to trained model checkpoint |
| `--input-dir` | model_input/data/training/input | Directory with input conditioning data |
| `--target-dir` | model_input/data/training/target | Directory with ground truth RSS maps |
| `--output-dir` | rss_visualizations | Where to save PNG files |
| `--num-scenes` | 5 | How many scenes to visualize (1-5) |
| `--diffusion-steps` | 50 | Number of reverse diffusion steps (higher=better quality, slower) |
| `--gpu` | None | GPU device ID (auto-detect if available) |

**Examples**:
```bash
# Single scene, fast (20 steps)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Three scenes, balanced quality
python visualize_rss_maps.py --num-scenes 3 --diffusion-steps 50

# All scenes, high quality
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 100 --output-dir rss_hires

# With custom checkpoint
python visualize_rss_maps.py --checkpoint models/checkpoints/epoch_50.pt --num-scenes 2
```

**Output files** (per scene):
```
rss_visualizations/
├── scene0_rss_comparison.png     # 2×3 grid with predictions, GT, error, conditioning
├── scene0_predicted.png          # Predicted RSS heatmap
├── scene0_groundtruth.png        # Ground truth RSS heatmap
├── scene1_rss_comparison.png
├── scene1_predicted.png
├── scene1_groundtruth.png
...
```

**Time estimates**:
- 1 scene, 20 steps: ~30 seconds
- 1 scene, 50 steps: ~1 minute
- 3 scenes, 50 steps: ~3 minutes
- 5 scenes, 50 steps: ~5 minutes
- 5 scenes, 100 steps: ~10 minutes

---

### batch_visualize.py
**Purpose**: Run multiple visualization configurations in sequence

**Command**:
```bash
python batch_visualize.py [OPTIONS]
```

**Options**:
| Option | Description |
|--------|-------------|
| `--config {quick,standard,complete,hires,sampling_study}` | Preset configuration (default: standard) |
| `--all` | Run all 5 configurations |
| `--list` | List available configurations |

**Preset Configurations**:

| Preset | Scenes | Steps | Time | Purpose |
|--------|--------|-------|------|---------|
| **quick** | 1 | 20 | 30 sec | Test setup |
| **standard** | 3 | 50 | 2 min | Balanced quality |
| **complete** | 5 | 50 | 5 min | All data |
| **hires** | 2 | 100 | 4 min | High quality |
| **sampling_study** | 1 | 10 | 15 sec | Minimal steps test |

**Examples**:
```bash
# Test setup
python batch_visualize.py --config quick

# Standard quality across 3 scenes
python batch_visualize.py --config standard

# Run all configurations
python batch_visualize.py --all

# List available configurations
python batch_visualize.py --list
```

**Output structure**:
```
rss_visualizations_quick/        (for --config quick)
rss_visualizations_standard/     (for --config standard)
rss_visualizations_hires/        (for --config hires)
...
```

---

## Understanding the Visualizations

### 2×3 Comparison Figure (`{scene}_rss_comparison.png`)
Six-panel layout showing complete prediction analysis:

```
┌──────────────────────────────────────────┐
│  Predicted RSS      │  Ground Truth RSS   │
├──────────────────────────────────────────┤
│  Error Map          │  Elevation         │
├──────────────────────────────────────────┤
│  Distance Heatmap   │  Frequency Log10   │
└──────────────────────────────────────────┘
```

**Top row**: Model performance
- **Left**: Predicted RSS (model output after reverse diffusion)
- **Right**: Ground truth RSS (reference from ray-tracing)

**Middle row**: Error analysis
- **Left**: Absolute error map (|predicted - ground truth|)
- **Right**: Elevation conditioning map (input to model)

**Bottom row**: Input conditioning
- **Left**: Distance heatmap (transmitter distance)
- **Right**: Frequency log10 (frequency conditioning)

**Color scales**:
- **RSS maps**: -100 dBm (dark blue) to -50 dBm (red)
- **Error map**: 0 dB (light) to 10+ dB (dark red)
- **Elevation**: 0 m (blue) to max height (yellow)
- **Distance**: 0 m (blue) to max distance (red)

### Single Heatmap Figures

**Predicted map** (`{scene}_predicted.png`):
- Single large heatmap of model predictions
- Statistics overlay:
  - Mean: Average RSS value
  - Std Dev: Variability
  - Min/Max: Dynamic range
  - Coverage@-70dB: Percentage of pixels above -70 dBm

**Ground truth map** (`{scene}_groundtruth.png`):
- Single large heatmap of reference RSS
- Same statistics overlay for comparison

---

## Interpreting Results

### Quality indicators

**Good predictions** show:
- ✓ Similar color distribution to ground truth
- ✓ Error map mostly blue (< 3 dB)
- ✓ Peak locations aligned
- ✓ Smooth transitions (not noisy)

**Poor predictions** show:
- ✗ Completely different color distribution
- ✗ Large red/orange regions in error map (>5 dB)
- ✗ Peaks in wrong locations
- ✗ Noisy speckled appearance

### Typical performance by dataset size

| Dataset Size | Expected RMSE | Interpretation |
|--------------|---------------|-----------------|
| **5 scenes** | 15-25 dB | Baseline - significant variance expected |
| **50 scenes** | 10-15 dB | Moderate quality |
| **500 scenes** | 7-10 dB | Good quality |
| **2000+ scenes** | <5 dB | Excellent (AIRMap level) |

**Note**: With only 5 training scenes, high error is expected. Visual inspection should focus on whether model is learning the *general* propagation patterns, not precise predictions.

---

## Troubleshooting

### Script fails to run
```bash
# Check Python environment is configured
python -c "import torch; print(f'GPU: {torch.cuda.is_available()}')"

# Verify model checkpoint exists
ls -lh models/checkpoints/model_final.pt

# Check data directories
ls -la model_input/data/training/input/ | head -5
```

### No output files created
- Check terminal output for error messages
- Verify write permissions: `touch rss_visualizations/test.txt`
- Check disk space: `df -h`

### Very slow execution
- Reduce `--diffusion-steps` (default 50, try 20)
- Reduce `--num-scenes` (try 1)
- On GPU: Check VRAM usage with `nvidia-smi` or `metal` command

### Memory error (CUDA out of memory)
```bash
# Run on CPU
python visualize_rss_maps.py --gpu -1

# Or reduce scene count
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
```

### Weird looking visualizations
- Check if model was actually trained (check loss in logs)
- Run with `--diffusion-steps 100` for better quality
- Compare with ground truth - small dataset may have low quality anyway

---

## Advanced Usage

### Generate visualizations with specific checkpoint
```bash
# After running training, use a specific epoch checkpoint
python visualize_rss_maps.py \
    --checkpoint models/checkpoints/epoch_20.pt \
    --num-scenes 3 \
    --output-dir rss_viz_epoch20
```

### Custom output directory
```bash
python visualize_rss_maps.py \
    --num-scenes 2 \
    --output-dir my_custom_viz_dir
```

### Batch process with detailed timing
```bash
#!/bin/bash
# Loop through diffusion step counts
for steps in 20 50 100; do
    python visualize_rss_maps.py \
        --num-scenes 1 \
        --diffusion-steps $steps \
        --output-dir rss_viz_steps${steps}
    echo "Completed $steps steps"
done
```

---

## Performance Metrics Interpretation

When viewing multiple scenes, look for patterns:

**Consistency metrics**:
- Do errors scale with distance from transmitter?
- Are high-loss regions (buildings) captured?
- Is elevation correctly influencing predictions?

**Spatial patterns**:
- Line-of-sight effects (sharp dropoffs)
- Shadowing behind buildings
- Diffraction patterns

---

## Integration with Backtesting

For quantitative metrics on the same visualizations:

```bash
# 1. Generate visualizations
python batch_visualize.py --config standard

# 2. Run quantitative backtesting
python run_backtest.py --diffusion-steps 50 --samples-per-scene 3 --output-dir backtest_results

# 3. Compare results
#    - backtest_results/evaluation_metrics.json has quantitative scores
#    - rss_visualizations_standard/ has qualitative visualizations
```

---

## Citation & References

- **AIRMap Paper**: "AIRMap: Efficient Outdoor Radio Map Prediction using AI"
  - 60,000 training scenes
  - Achieves ~5 dB RMSE
  
- **Your Dataset**: 5 training scenes
  - Expected 15-25 dB RMSE
  - Useful for validating methodology
  - Ready to scale with more data

---

## Questions & Next Steps

**Common questions**:

*Q: Why is the error so high?*  
A: With only 5 training scenes, high error is normal. Add more scenes to improve.

*Q: Are the predictions reasonable?*  
A: Check the comparison figures. Model should capture general propagation patterns even if absolute values are off.

*Q: How do I improve quality?*  
A: (1) Add more training data, (2) Train longer, (3) Tune hyperparameters.

**Next steps**:
1. Run `batch_visualize.py --config quick` to see your first visualizations
2. Compare predicted vs ground truth in the 2×3 grids
3. Run `run_backtest.py` for quantitative metrics
4. Consider data augmentation or additional scenes

