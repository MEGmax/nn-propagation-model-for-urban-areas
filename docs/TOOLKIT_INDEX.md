# Complete Toolkit Index

**Last Updated**: Latest session  
**Your Project**: NN-Propagation-Model (Diffusion-based RSS Prediction)

---

## 🎯 By Task

### I want to train the model
→ See [Training](#training-tools)

### I want to test the model
→ See [Inference](#inference-tools)

### I want to visualize predictions (PNG maps)
→ See [Visualization](#visualization-tools) ⭐ **NEW**

### I want quantitative metrics (RMSE, MAE, etc.)
→ See [Evaluation](#evaluation-tools)

### I want documentation
→ See [Documentation](#documentation)

---

## 🏋️ Training Tools

### `run_training.py`
**Purpose**: Train the diffusion model  
**Usage**:
```bash
python run_training.py --epochs 50 --batch-size 8 --lr 0.0001
```
**Output**: Checkpoints in `models/checkpoints/`  
**Time**: ~5 minutes per epoch on GPU

**Key options**:
- `--epochs`: Number of training epochs (default 50)
- `--batch-size`: Batch size (default 8)
- `--lr`: Learning rate (default 0.0001)
- `--save-every`: Save checkpoint every N epochs (default 10)
- `--checkpoint`: Resume from checkpoint
- `--num-workers`: Dataloader workers (default 4)

---

## 🔮 Inference Tools

### `run_backtest.py`
**Purpose**: Generate predictions and compute metrics  
**Usage**:
```bash
python run_backtest.py --diffusion-steps 50 --samples-per-scene 3
```
**Output**: JSON metrics + PNG plots  
**Time**: ~5-10 minutes

**Key options**:
- `--checkpoint`: Model checkpoint (default model_final.pt)
- `--diffusion-steps`: Reverse diffusion steps (default 50)
- `--samples-per-scene`: Predictions per scene (default 3)
- `--batch-size`: Batch size for inference (default 8)
- `--output-dir`: Where to save results
- `--quick`: Fast mode (fewer samples)

---

## 🎨 Visualization Tools (NEW)

### `visualize_rss_maps.py`
**Purpose**: Generate PNG heatmaps of RSS predictions  
**Usage**:
```bash
python visualize_rss_maps.py --num-scenes 3 --diffusion-steps 50
```
**Output**: PNG files per scene (3 files each):
- `{scene}_rss_comparison.png` - 2×3 grid (predicted, GT, error, inputs)
- `{scene}_predicted.png` - Predicted RSS heatmap
- `{scene}_groundtruth.png` - Ground truth RSS heatmap

**Time**: 
- 1 scene, 20 steps: ~30 sec
- 3 scenes, 50 steps: ~2 min
- 5 scenes, 100 steps: ~10 min

**Key options**:
- `--num-scenes`: How many scenes (1-5)
- `--diffusion-steps`: Reverse diffusion steps (default 50)
- `--output-dir`: Output directory (default rss_visualizations)
- `--checkpoint`: Model checkpoint

### `batch_visualize.py`
**Purpose**: Run multiple visualization configurations  
**Usage**:
```bash
python batch_visualize.py --config standard
```
**Presets**:
| Name | Scenes | Steps | Time | Purpose |
|------|--------|-------|------|---------|
| quick | 1 | 20 | 30s | Test |
| standard | 3 | 50 | 2m | Balanced |
| complete | 5 | 50 | 5m | All data |
| hires | 2 | 100 | 4m | High quality |
| sampling_study | 1 | 10 | 15s | Minimal test |

**Usage**:
```bash
python batch_visualize.py --config standard   # Run preset
python batch_visualize.py --list              # List presets
python batch_visualize.py --all               # Run all presets
```

### `quickstart_viz.sh`
**Purpose**: One-command launcher for visualization  
**Usage**:
```bash
bash quickstart_viz.sh
```
Runs quick test, checks dependencies, explains options

---

## 📊 Evaluation Tools

### `backtest_evaluation.py`
**Purpose**: Core evaluation framework (library)  
**Features**:
- 15+ metrics (RMSE, MAE, median error, bias, coverage, correlation)
- Per-scene and aggregate statistics
- JSON export
- Visualization plotting

**Classes**:
- `RadioMapEvaluator`: Main metric computation
- `DiffusionSampler`: Prediction generation wrapper
- `RSSNormalizer`: dBm ↔ normalized conversion

**Usually called by**: `run_backtest.py`

---

## 📚 Documentation

### Quick Start Guides
| File | Purpose | Read Time |
|------|---------|-----------|
| [VISUALIZATION_TOOLS_QUICKSTART.md](VISUALIZATION_TOOLS_QUICKSTART.md) | Get started with PNGs | 5 min |
| [VISUALIZATION_SUMMARY.md](VISUALIZATION_SUMMARY.md) | Overview of all tools | 10 min |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Commands at a glance | 3 min |

### Comprehensive Guides
| File | Purpose | Read Time |
|------|---------|-----------|
| [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md) | Complete visualization docs | 20 min |
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | Metrics and interpretation | 25 min |
| [ASSESSMENT_REPORT.md](ASSESSMENT_REPORT.md) | Executive summary | 15 min |

### Strategic Documents
| File | Purpose | Read Time |
|------|---------|-----------|
| [AIRMAP_COMPARISON.md](AIRMAP_COMPARISON.md) | Compare to AIRMap paper | 20 min |
| [BACKTEST_SUMMARY.md](BACKTEST_SUMMARY.md) | Testing methodology | 15 min |
| [README_ASSESSMENT.md](README_ASSESSMENT.md) | Project assessment | 10 min |

### References
| File | Purpose |
|------|---------|
| [FILES_CREATED.txt](FILES_CREATED.txt) | Manifest of all deliverables |
| [DELIVERABLES.md](DELIVERABLES.md) | What was created this session |

---

## 🔧 Core Model Files

### Models
- `models/model.py` - TimeCondUNet architecture (rewritten for training)
- `models/diffusion.py` - Diffusion process implementation
- `model_input/model_input.py` - Dataset loading

### Data
- `model_input/data/training/input/` - Conditioning tensors (5 scenes)
- `model_input/data/training/target/` - Ground truth RSS (5 scenes)

---

## 📋 Common Workflows

### Workflow 1: Quick Visual Check
```bash
# 1. See what the model generates
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# 2. View the PNG files
open rss_visualizations/scene0_rss_comparison.png
```
**Time**: ~1 minute

### Workflow 2: Complete Evaluation
```bash
# 1. Generate visualizations
python batch_visualize.py --config standard

# 2. Compute quantitative metrics
python run_backtest.py

# 3. Review results
# - PNGs in rss_visualizations_standard/
# - Metrics in backtest_results/evaluation_metrics.json
```
**Time**: ~15 minutes

### Workflow 3: Training + Testing
```bash
# 1. Train model (optional if already trained)
python run_training.py --epochs 50

# 2. Generate visualizations with new checkpoint
python visualize_rss_maps.py --checkpoint models/checkpoints/model_final.pt --num-scenes 3

# 3. Compute metrics
python run_backtest.py --checkpoint models/checkpoints/model_final.pt
```
**Time**: ~30+ minutes

### Workflow 4: Systematic Comparison
```bash
# 1. Run batch visualizations
python batch_visualize.py --all

# 2. Run backtest with different step counts
for steps in 20 50 100; do
    python run_backtest.py --diffusion-steps $steps --output-dir backtest_steps${steps}
done

# 3. Compare outputs across directories
```
**Time**: ~1+ hour

---

## 🚀 Quick Commands

```bash
# See predictions (fastest)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Standard quality
python batch_visualize.py --config standard

# High quality (all scenes)
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50

# Get metrics
python run_backtest.py

# Train (if no checkpoint)
python run_training.py --epochs 20

# See available tools
python batch_visualize.py --list
```

---

## 📖 Reading Guide

**If you're new**: Start with [VISUALIZATION_TOOLS_QUICKSTART.md](VISUALIZATION_TOOLS_QUICKSTART.md)

**If you want details**: Read [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md)

**If you want context**: Read [ASSESSMENT_REPORT.md](ASSESSMENT_REPORT.md)

**If you want specifics**: Read [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)

**If you want to understand**: Read [AIRMAP_COMPARISON.md](AIRMAP_COMPARISON.md)

---

## ✅ What's Available

| Tool | Status | Usage |
|------|--------|-------|
| Training (`run_training.py`) | ✅ Ready | `python run_training.py --epochs 50` |
| Inference (`run_backtest.py`) | ✅ Ready | `python run_backtest.py --diffusion-steps 50` |
| Visualization (`visualize_rss_maps.py`) | ✅ Ready | `python visualize_rss_maps.py --num-scenes 3` |
| Batch processing (`batch_visualize.py`) | ✅ Ready | `python batch_visualize.py --config standard` |
| Quick start (`quickstart_viz.sh`) | ✅ Ready | `bash quickstart_viz.sh` |
| Evaluation framework (`backtest_evaluation.py`) | ✅ Ready | Used by run_backtest.py |

---

## 💾 Directory Structure

```
/Users/matthewgrech/ECE2T5F/ECE496/Diffusion/
├── VISUALIZATION_TOOLS_QUICKSTART.md    ← Start here
├── VISUALIZATION_TOOLKIT_GUIDE.md       ← Full reference
├── VISUALIZATION_SUMMARY.md             ← This file
│
├── visualize_rss_maps.py               ← Main visualization
├── batch_visualize.py                  ← Batch runner
├── quickstart_viz.sh                   ← Quick launcher
│
├── run_backtest.py                     ← Evaluation CLI
├── backtest_evaluation.py              ← Evaluation framework
├── run_training.py                     ← Training CLI
│
├── models/
│   ├── model.py                        ← Architecture + training
│   ├── diffusion.py                    ← Diffusion process
│   └── checkpoints/
│       └── model_final.pt              ← Trained model
│
├── model_input/
│   ├── model_input.py                  ← Data loading
│   └── data/training/
│       ├── input/  (5 conditioning tensors)
│       └── target/ (5 ground truth RSS)
│
├── rss_visualizations/                 ← PNG outputs (created)
│
└── [documentation files above]
```

---

## 🎓 Learning Path

**5 minutes**: Run quickstart
```bash
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
```

**15 minutes**: Read VISUALIZATION_TOOLS_QUICKSTART.md

**30 minutes**: Generate standard visualizations + read interpretation guide

**1 hour**: Read VISUALIZATION_TOOLKIT_GUIDE.md + run batch_visualize.py --all

**2 hours**: Read ASSESSMENT_REPORT.md + AIRMAP_COMPARISON.md for context

---

## Next Steps

1. **Start**: `python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20`
2. **View**: Open PNG files in `rss_visualizations/`
3. **Understand**: Read [VISUALIZATION_TOOLS_QUICKSTART.md](VISUALIZATION_TOOLS_QUICKSTART.md)
4. **Explore**: Try different commands from [Quick Commands](#-quick-commands)
5. **Deepen**: Read [VISUALIZATION_TOOLKIT_GUIDE.md](VISUALIZATION_TOOLKIT_GUIDE.md) for details

---

**You're all set! Everything you need is ready to use.** 🚀

