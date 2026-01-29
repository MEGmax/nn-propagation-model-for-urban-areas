# 🚀 Visualization Toolkit - Ready to Use

## ⚡ Fastest Way to Get Started (Pick One)

### Option A: Python (Direct)
```bash
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
```

### Option B: Batch Runner (Organized)
```bash
python batch_visualize.py --config standard
```

### Option C: Shell Script (One-liner)
```bash
bash quickstart_viz.sh
```

**Expected time**: 30 seconds to 2 minutes  
**Expected output**: PNG files in `rss_visualizations/`

---

## 📁 Files You Now Have

### Scripts (Ready to Run)
| File | Purpose | Run With |
|------|---------|----------|
| `visualize_rss_maps.py` | Generate PNG heatmaps | `python visualize_rss_maps.py ...` |
| `batch_visualize.py` | Run presets | `python batch_visualize.py --config X` |
| `quickstart_viz.sh` | One-command test | `bash quickstart_viz.sh` |

### Documentation (4 Guides)
| File | Best For | Read Time |
|------|----------|-----------|
| `VISUALIZATION_TOOLS_QUICKSTART.md` | Getting started | 5 min |
| `VISUALIZATION_SUMMARY.md` | Overview | 10 min |
| `VISUALIZATION_TOOLKIT_GUIDE.md` | Complete reference | 20 min |
| `TOOLKIT_INDEX.md` | Finding what you need | 15 min |

### Utility Files
| File | Purpose |
|------|---------|
| `WHAT_WAS_CREATED.md` | This session's deliverables |

---

## 🎯 What Each Script Does

### `visualize_rss_maps.py`
Generates PNG heatmaps of what the model predicts for RSS strength

**Example**:
```bash
# See predictions for 1 scene (fast)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# See predictions for 3 scenes (standard)
python visualize_rss_maps.py --num-scenes 3 --diffusion-steps 50

# Get high quality for all 5 scenes
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 100
```

**Produces**:
- `scene0_rss_comparison.png` - 6-panel grid (predicted, truth, error, inputs)
- `scene0_predicted.png` - Just the prediction heatmap
- `scene0_groundtruth.png` - Just the reference heatmap

### `batch_visualize.py`
Runs multiple visualization configurations automatically

**Example**:
```bash
python batch_visualize.py --config quick       # 30 seconds, 1 scene
python batch_visualize.py --config standard    # 2 minutes, 3 scenes
python batch_visualize.py --config complete    # 5 minutes, all scenes
python batch_visualize.py --all                # Run all 5 configs
```

### `quickstart_viz.sh`
Checks everything and runs a quick test in one command

```bash
bash quickstart_viz.sh
```

---

## 📊 Understanding the Output

### The 2×3 Grid (`scene0_rss_comparison.png`)
```
┌─────────────────┬─────────────────┐
│  PREDICTED RSS  │ GROUND TRUTH    │
├─────────────────┼─────────────────┤
│  ERROR MAP      │  ELEVATION      │
├─────────────────┼─────────────────┤
│  DISTANCE       │  FREQUENCY      │
└─────────────────┴─────────────────┘
```

**What it shows**:
- **Top**: How close predictions are to ground truth
- **Middle**: Error magnitude (blue=small, red=large)
- **Bottom**: Input conditioning (what the model saw)

**Good signs**:
- ✓ Similar colors in top row
- ✓ Error map mostly blue
- ✓ Peaks align
- ✓ Smooth appearance

**Bad signs**:
- ✗ Completely different colors
- ✗ Large red error regions
- ✗ Misaligned peaks
- ✗ Noisy speckled appearance

---

## ⏱️ Timing Guide

| Command | Scenes | Steps | Time | Best For |
|---------|--------|-------|------|----------|
| `--num-scenes 1 --diffusion-steps 20` | 1 | 20 | 30s | Test |
| `--num-scenes 1 --diffusion-steps 50` | 1 | 50 | 1m | Quick check |
| `batch_visualize --config standard` | 3 | 50 | 2m | Balanced |
| `--num-scenes 5 --diffusion-steps 50` | 5 | 50 | 5m | Complete |
| `--num-scenes 5 --diffusion-steps 100` | 5 | 100 | 10m | High quality |

---

## 🎯 Most Common Commands

```bash
# See your first results (fastest)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Standard quality (recommended)
python batch_visualize.py --config standard

# Explore options
python batch_visualize.py --list

# All scenes, good quality
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50

# Get quantitative metrics (complementary)
python run_backtest.py
```

---

## 📚 Which Documentation to Read

**Completely new?**
→ Start with: [VISUALIZATION_TOOLS_QUICKSTART.md](VISUALIZATION_TOOLS_QUICKSTART.md) (5 min)

**Want complete instructions?**
→ Read: [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md) (20 min)

**Want to find specific info?**
→ Use: [TOOLKIT_INDEX.md](TOOLKIT_INDEX.md) (look it up)

**Want high-level overview?**
→ Read: [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) (10 min)

**Want to see what was delivered?**
→ Read: [WHAT_WAS_CREATED.md](WHAT_WAS_CREATED.md) (10 min)

---

## 🔍 Expected Performance

With **5 training scenes**, expect:

| Metric | Value | Meaning |
|--------|-------|---------|
| RMSE | 15-25 dB | Normal for this data size |
| Visual quality | Rough patterns | General propagation captured |
| Exact predictions | ±5-10 dB off | Limited by training data |

**To improve**: Add more training scenes (need 50+ for good results)

---

## ✅ Quick Health Check

Before running, verify:

```bash
# Check model exists
ls models/checkpoints/model_final.pt

# Check training data exists  
ls model_input/data/training/input/scene*.npy | head -5

# Create output directory if needed
mkdir -p rss_visualizations
```

---

## 🎬 Next Steps

1. **Run quick test**:
   ```bash
   python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
   ```

2. **View output**:
   Open `rss_visualizations/scene0_rss_comparison.png` in image viewer

3. **Try more**:
   ```bash
   python batch_visualize.py --config standard
   ```

4. **Get context**:
   Read [VISUALIZATION_TOOLS_QUICKSTART.md](VISUALIZATION_TOOLS_QUICKSTART.md)

5. **Explore options**:
   ```bash
   python visualize_rss_maps.py --help
   ```

---

## 💡 Pro Tips

- **Fast iteration**: Start with `--diffusion-steps 20`, increase to 50 or 100 later
- **Batch runs**: Use `batch_visualize.py` to run multiple configurations
- **Compare quality**: Try 10, 50, and 100 steps on same scene
- **Integration**: Run both `visualize_rss_maps.py` and `run_backtest.py` together
- **Directory control**: Use `--output-dir` to organize multiple runs

---

## 🆘 Troubleshooting

| Issue | Fix |
|-------|-----|
| "File not found" | Run from project root: `cd /Users/matthewgrech/ECE2T5F/ECE496/Diffusion` |
| "No PNGs created" | Check output dir: `mkdir -p rss_visualizations` |
| "Takes too long" | Reduce steps: `--diffusion-steps 20` |
| "Script fails" | See [VISUALIZATION_TOOLKIT_GUIDE.md#troubleshooting](VISUALIZATION_TOOLKIT_GUIDE.md) |

---

## 📖 Documentation Map

```
Quick Start (5 min)
        ↓
VISUALIZATION_TOOLS_QUICKSTART.md
        ↓
Want more details?
        ↓
VISUALIZATION_TOOLKIT_GUIDE.md (20 min)
        ↓
Want full context?
        ↓
TOOLKIT_INDEX.md + ASSESSMENT_REPORT.md
```

---

## ✨ Summary

You now have **3 easy ways** to visualize RSS maps:

| Method | Command | Time |
|--------|---------|------|
| Direct Python | `python visualize_rss_maps.py ...` | Custom |
| Batch presets | `python batch_visualize.py --config X` | Fast |
| One-liner | `bash quickstart_viz.sh` | Auto |

**All scripts are ready to use. Just pick one and run it!**

```bash
# Absolute fastest
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
```

See your predictions in **30 seconds**. 🚀

---

**Questions?** See [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md)  
**Lost?** See [TOOLKIT_INDEX.md](TOOLKIT_INDEX.md)  
**Want overview?** See [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md)

You're all set! 🎉

