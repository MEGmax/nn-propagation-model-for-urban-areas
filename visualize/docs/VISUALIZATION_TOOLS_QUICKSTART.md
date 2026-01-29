# Visualization Tools - Quick Reference

You now have **three easy ways** to generate RSS map visualizations.

---

## 🚀 **Fastest Way to Start** (< 1 minute setup)

```bash
# Option A: Direct script (simplest)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20

# Option B: Batch runner (most organized)
python batch_visualize.py --config quick

# Option C: Shell script (one-liner)
bash quickstart_viz.sh
```

**Expected result**: Creates `rss_visualizations/` directory with PNG files

---

## 📊 **Files Created**

| File | Purpose | Use When |
|------|---------|----------|
| **visualize_rss_maps.py** | Core visualization engine | Direct Python execution |
| **batch_visualize.py** | Batch runner with presets | Want multiple configurations |
| **quickstart_viz.sh** | One-command launcher | Prefer shell/bash |
| **VISUALIZATION_TOOLKIT_GUIDE.md** | Complete reference | Need detailed docs |
| **VISUALIZATION_TOOLS_QUICKSTART.md** | This file | Getting started |

---

## 🎯 **Typical Workflow**

### Step 1: Quick test (verify everything works)
```bash
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20
```
**Time**: ~30 seconds  
**Output**: 3 PNG files for scene 0

### Step 2: Standard quality (3 scenes)
```bash
python batch_visualize.py --config standard
```
**Time**: ~2 minutes  
**Output**: 9 PNG files (3 scenes × 3 files each)

### Step 3: Production quality (all scenes)
```bash
python visualize_rss_maps.py --num-scenes 5 --diffusion-steps 50
```
**Time**: ~5 minutes  
**Output**: 15 PNG files (5 scenes × 3 files each)

---

## 📁 **Output Structure**

After running visualizations, your directory looks like:

```
rss_visualizations/
├── scene0_rss_comparison.png      ← 6-panel grid (predictions, GT, error, conditioning)
├── scene0_predicted.png           ← Just the predicted RSS map
├── scene0_groundtruth.png         ← Just the reference RSS map
├── scene1_rss_comparison.png
├── scene1_predicted.png
├── scene1_groundtruth.png
├── scene2_rss_comparison.png
...
```

---

## 🔍 **What Each PNG Shows**

### `scene0_rss_comparison.png` (2×3 grid)
```
Predicted RSS   │   Ground Truth RSS
────────────────┼─────────────────────
Error Map       │   Elevation (input)
────────────────┼─────────────────────
Distance Map    │   Frequency (input)
```

**Interpretation**:
- ✓ **Good**: Similar color distribution, error map mostly blue (< 3 dB)
- ✗ **Poor**: Different colors, large red patches (> 5 dB error)

### `scene0_predicted.png`
Single heatmap of model predictions with statistics overlay:
- Mean RSS value
- Standard deviation
- Min/Max range
- Coverage percentage above -70 dB

### `scene0_groundtruth.png`
Single heatmap of reference RSS for direct comparison

---

## ⚙️ **Common Command Variations**

```bash
# Test with different step counts (quality tradeoff)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 10   # Fast (15 sec)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 50   # Balanced (1 min)
python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 100  # High-quality (2 min)

# Use different checkpoint
python visualize_rss_maps.py --checkpoint models/checkpoints/epoch_30.pt --num-scenes 1

# Save to custom directory
python visualize_rss_maps.py --num-scenes 3 --output-dir my_viz_folder

# Run all batch presets
python batch_visualize.py --all
```

---

## 🐛 **Troubleshooting**

| Problem | Solution |
|---------|----------|
| **"No such file"** | Run from project root: `cd /Users/matthewgrech/ECE2T5F/ECE496/Diffusion` |
| **"GPU out of memory"** | Reduce scenes: `--num-scenes 1` |
| **"Script takes forever"** | Reduce steps: `--diffusion-steps 20` |
| **No PNGs generated** | Check `rss_visualizations/` directory exists: `mkdir -p rss_visualizations` |
| **Weird looking images** | Check model was trained: `ls -lh models/checkpoints/` |

---

## 💡 **Expected Performance**

With **5 training scenes**, expect:

| Metric | Value | Meaning |
|--------|-------|---------|
| **RMSE** | 15-25 dB | Normal for limited data |
| **Error map** | Mostly blue | < 5 dB typical error |
| **Pattern matching** | Rough similarity | General propagation captured |
| **Exact values** | Off by 5-10 dB | Precision limited by dataset size |

**Baseline comparison**:
- AIRMap (60,000 scenes): ~5 dB RMSE
- Your model (5 scenes): ~15-25 dB RMSE expected
- To improve: Add more training data

---

## 📚 **Full Documentation**

For detailed information, see:
- **VISUALIZATION_TOOLKIT_GUIDE.md** - Complete reference (all options, interpretation, advanced usage)
- **BACKTEST_SUMMARY.md** - Quantitative metrics
- **ASSESSMENT_REPORT.md** - Overall performance evaluation

---

## ✅ **Verification Checklist**

Before running visualizations:

- [ ] You're in the right directory: `/Users/matthewgrech/ECE2T5F/ECE496/Diffusion`
- [ ] Model checkpoint exists: `ls models/checkpoints/model_final.pt`
- [ ] Training data exists: `ls model_input/data/training/input/`
- [ ] Output directory is writable: `touch rss_visualizations/test.txt && rm rss_visualizations/test.txt`

If all checks pass, you're ready to generate visualizations!

---

## 🎓 **Next Steps**

1. **Run quick test**: `python visualize_rss_maps.py --num-scenes 1 --diffusion-steps 20`
2. **View the PNGs**: Open `rss_visualizations/scene0_rss_comparison.png` in image viewer
3. **Understand the results**: Compare predicted vs ground truth
4. **Generate more**: Try `--num-scenes 3` or `--num-scenes 5` for complete picture
5. **Quantify**: Run `python run_backtest.py` for numerical metrics

