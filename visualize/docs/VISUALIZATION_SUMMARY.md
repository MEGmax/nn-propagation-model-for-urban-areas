# Visualization Tools Summary

## What You Can Do Now

You have a complete toolkit for generating and analyzing RSS map visualizations:

### 1. **Direct Visualization** (`visualize_rss_maps.py`)
- Generate PNG heatmaps of RSS predictions vs ground truth
- Full CLI control over options
- Best for: Custom configurations

### 2. **Batch Processing** (`batch_visualize.py`)
- Run multiple preset configurations in sequence
- 5 pre-optimized profiles (quick, standard, complete, hires, sampling_study)
- Best for: Systematic comparison

### 3. **Quick Start** (`quickstart_viz.sh`)
- Single shell script that runs quick test
- Checks dependencies automatically
- Best for: Fast verification

---

## Quickest Start

```bash
# Absolute fastest way to see results
cd /Users/matthewgrech/ECE2T5F/ECE496/Diffusion
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Or if you prefer bash
bash quickstart_viz.sh
```

**Expected runtime**: ~30 seconds  
**Output**: PNG files in `rss_visualizations/`

---

## Generated Files This Session

### Scripts
- ✅ `visualize_rss_maps.py` - Core visualization (300+ lines)
- ✅ `batch_visualize.py` - Batch runner (200+ lines)
- ✅ `quickstart_viz.sh` - One-command launcher

### Documentation
- ✅ `VISUALIZATION_TOOLKIT_GUIDE.md` - Complete reference (500+ lines)
- ✅ `VISUALIZATION_TOOLS_QUICKSTART.md` - Quick start guide
- ✅ `VISUALIZATION_SUMMARY.md` - This file

---

## Integration with Previous Work

You now have a complete evaluation pipeline:

```
┌─────────────────────────────────────────────┐
│  1. Training                                 │
│     run_training.py                         │
│     → models/checkpoints/model_final.pt    │
└────────────────┬────────────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
┌───v────────────┐   ┌───────v──────────┐
│  Visualization │   │  Quantitative     │
│  PNG Maps      │   │  Metrics          │
│                │   │                   │
│ visualize_*    │   │ run_backtest.py   │
│ batch_visualize│   │ backtest_eval.py  │
│ quickstart_viz │   │ → JSON reports    │
└────────────────┘   └───────────────────┘
```

**To get complete picture**:
```bash
# 1. Generate visualizations (qualitative)
python batch_visualize.py --config standard

# 2. Run backtest (quantitative)
python run_backtest.py --output-dir backtest_results

# 3. Compare results from both
```

---

## Key Capabilities

| Tool | Input | Output | Time |
|------|-------|--------|------|
| `visualize_rss_maps.py` | Checkpoint + data | PNG heatmaps | <5 min for 5 scenes |
| `batch_visualize.py` | Config preset | Multiple PNG sets | <30 min for all presets |
| `run_backtest.py` | Checkpoint + data | JSON metrics + plots | <10 min |

---

## Command Reference

### Most Common Commands

```bash
# See what the model generates (quick)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Standard quality (3 scenes)
python batch_visualize.py --config standard

# All scenes high quality
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50

# With different checkpoint
python visualize_rss_maps.py --checkpoint models/checkpoints/epoch_50.pt --num-scenes 2

# Get numerical metrics
python run_backtest.py

# List batch presets
python batch_visualize.py --list

# Run all batch presets
python batch_visualize.py --all
```

---

## File Locations

```
/Users/matthewgrech/ECE2T5F/ECE496/Diffusion/
├── visualize_rss_maps.py              ← Use this for custom runs
├── batch_visualize.py                 ← Use this for presets
├── quickstart_viz.sh                  ← Use this for quick test
├── run_backtest.py                    ← Use this for metrics
├── VISUALIZATION_TOOLKIT_GUIDE.md     ← Full documentation
├── VISUALIZATION_TOOLS_QUICKSTART.md  ← Getting started
└── rss_visualizations/                ← Output directory (created automatically)
    ├── scene0_rss_comparison.png
    ├── scene0_predicted.png
    ├── scene0_groundtruth.png
    └── ... (more scenes)
```

---

## Expected Outputs

### Example: `scene0_rss_comparison.png`
A 2×3 grid showing:
- Top-left: Your model's predicted RSS map (what it thinks)
- Top-right: Ground truth RSS map (what ray-tracing shows)
- Middle-left: Error map (where predictions are wrong)
- Middle-right: Elevation input (one of the model's inputs)
- Bottom-left: Distance heatmap (another input)
- Bottom-right: Frequency input (another input)

### Color meanings
- **RSS maps**: Blue = weak signal (-100 dBm), Red = strong signal (-50 dBm)
- **Error map**: Blue = small error (< 3 dB), Red = large error (> 10 dB)

### What to look for
- ✓ Similar overall color distribution between predicted and ground truth
- ✓ Error map mostly blue (small errors)
- ✓ Peak locations roughly aligned
- ✓ Smooth transitions (not random noise)

### If something looks wrong
- Make sure model was trained: `ls -lh models/checkpoints/model_final.pt`
- Try more diffusion steps: `--diffusion-steps 100`
- Check training loss was decreasing (review training logs)

---

## Next Steps

1. **Generate visualizations**:
   ```bash
   python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
   ```

2. **View the PNGs**: Open the files in your image viewer or terminal:
   ```bash
   open rss_visualizations/scene0_rss_comparison.png
   ```

3. **Compare with metrics**:
   ```bash
   python run_backtest.py
   ```

4. **For detailed interpretation**: Read [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md)

---

## Support

If you have issues:
1. Check [VISUALIZATION_TOOLS_QUICKSTART.md](VISUALIZATION_TOOLS_QUICKSTART.md) troubleshooting section
2. See [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md) for advanced options
3. Review [ASSESSMENT_REPORT.md](ASSESSMENT_REPORT.md) for context on expected performance

You're all set! 🚀

