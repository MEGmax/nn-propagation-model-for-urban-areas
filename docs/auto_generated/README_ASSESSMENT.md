# 📚 Complete Documentation Index

## Overview
This directory now contains a comprehensive model evaluation and backtesting framework for your conditional diffusion model for RSS prediction. Below is a guide to all documentation and scripts.

---

## 🚀 QUICKSTART (5 minutes)

### For the Impatient
```bash
# Train (optional)
cd models && python model.py --epochs 10 && cd ..

# Evaluate
python run_backtest.py

# View results
open backtest_results/backtest_metrics.png
cat backtest_results/backtest_results.json
```

**Read**: [`QUICK_REFERENCE.md`](#quick-reference) for metric interpretation

---

## 📄 Documentation Files

### 🎯 **START HERE** — [ASSESSMENT_REPORT.md](ASSESSMENT_REPORT.md)
**What**: Complete executive summary of the assessment  
**Length**: ~5 pages  
**Best for**: Understanding the big picture & expectations  
**Contains**:
- Executive summary of what was created
- Performance expectations by dataset size
- How to use the framework (3 options)
- Step-by-step walkthrough
- Architectural assessment vs AIRMap
- Actionable next steps

**Read this first if you only have 10 minutes.**

---

### 📖 **QUICK_REFERENCE.md** — Quick Lookup
**What**: Fast reference card for metrics & interpretation  
**Length**: ~3 pages (scannable)  
**Best for**: During backtesting, interpreting results on the fly  
**Contains**:
- 2-command startup
- What you'll get (outputs)
- Key metrics explained (table format)
- Success interpretation (RMSE/coverage thresholds)
- Performance by stage (baseline → state-of-art)
- Troubleshooting quick lookup
- Red flags vs green lights
- Metric formulas

**Keep this open while running backtests.**

---

### 📊 **EVALUATION_GUIDE.md** — Detailed Technical Guide
**What**: In-depth guide to metrics, usage, and troubleshooting  
**Length**: ~8 pages  
**Best for**: Deep understanding of evaluation methodology  
**Contains**:
- Detailed metric definitions
- Installation & setup
- Data structure explanation
- Usage examples (training & backtesting)
- Interpreting results section
- Expected results by scenario
- Diagnostic tips (organized by symptom)
- Model architecture reference with FiLM details
- Advanced usage & ensemble inference
- Troubleshooting table

**Read this if you want to understand the methodology deeply.**

---

### 🔬 **AIRMAP_COMPARISON.md** — Competitive Analysis
**What**: Detailed comparison between your model and AIRMap paper  
**Length**: ~10 pages  
**Best for**: Understanding strengths/weaknesses & optimization roadmap  
**Contains**:
- Architecture comparison (deterministic vs stochastic)
- Input conditioning analysis (your 3 channels vs AIRMap's 1)
- Training data gap analysis (5 vs 60,000 samples)
- Expected performance scaling laws
- Realistic RMSE expectations (15-25 dB at 5 scenes)
- What AIRMap did right
- Key insights for your model
- 4-phase optimization roadmap
- Backtesting action items (week 1, month 1)
- Diagnostic visualization suggestions
- Realistic next steps

**Read this for strategic decisions on model development.**

---

### 📋 **BACKTEST_SUMMARY.md** — Project Overview
**What**: Comprehensive summary of what was created & how to use it  
**Length**: ~6 pages  
**Best for**: Understanding the complete framework & next steps  
**Contains**:
- What was created (4 main components)
- Improved training script features
- Quick-start evaluation script
- Comprehensive guides overview
- Key findings (performance expectations)
- Immediate next steps
- Success criteria (minimum/good/competitive)
- Key documents reference table
- Backtesting workflow (4 steps)
- Important insights & unique strengths
- Common pitfalls & solutions
- Checklist before backtesting

**Read this if you want an overview of everything that's new.**

---

## 🛠️ Executable Scripts

### [run_backtest.py](run_backtest.py) — **PRIMARY EVALUATION SCRIPT**
**Purpose**: One-command backtesting & evaluation  
**Usage**:
```bash
# Full evaluation (~10 minutes)
python run_backtest.py

# Quick test (~2 minutes)  
python run_backtest.py --quick

# Custom checkpoint
python run_backtest.py --checkpoint models/checkpoints/model_epoch30.pt
```

**Options**:
- `--quick`: Reduce samples/steps for fast iteration
- `--checkpoint`: Load specific model checkpoint
- `--samples-per-scene`: Number of diffusion samples (default: 5)
- `--diffusion-steps`: Reverse diffusion steps (default: 50)
- `--gpu`: GPU device ID (default: 0)

**Output**:
- `backtest_results/backtest_results.json` — All metrics
- `backtest_results/backtest_metrics.png` — Diagnostic plots

**Start here for evaluation.**

---

### [backtest_evaluation.py](backtest_evaluation.py) — **EVALUATION FRAMEWORK**
**Purpose**: Core evaluation logic (imported by run_backtest.py)  
**Contains**:
- `RSSNormalizer`: Handle dBm ↔ normalized conversion
- `RadioMapEvaluator`: Compute 15+ metrics per scene
- `DiffusionSampler`: Generate ensemble predictions
- `backtest_on_dataset()`: Full evaluation pipeline
- `plot_evaluation_results()`: Generate diagnostic plots
- `print_evaluation_report()`: Format console output

**Use this if you want custom evaluation scripts.**

---

### [models/model.py](models/model.py) — **IMPROVED TRAINING SCRIPT**
**Purpose**: Train model with checkpoint management  
**Usage**:
```bash
cd models
python model.py \
    --epochs 50 \
    --batch-size 4 \
    --lr 2e-4 \
    --save-every 5 \
    --resume
```

**Options**:
- `--epochs`: Training epochs (default: 50)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 2e-4)
- `--save-every`: Save checkpoint every N epochs
- `--resume`: Resume from latest checkpoint
- `--num-workers`: Data loading workers

**New Features**:
- Checkpoint manager with metadata tracking
- Resume capability
- Detailed logging per epoch
- GPU memory profiling
- Training summary report

---

## 📊 Generated Output Files

After running `python run_backtest.py`, you'll get:

```
backtest_results/
├── backtest_results.json
│   └── Contains:
│       ├── per_scene[]: metrics for each scene
│       │   └── {rmse_db, mae_db, bias_db, coverage@3dB/5dB/10dB, ...}
│       ├── aggregate{}: summary statistics
│       │   └── {rmse_db_mean/std/min/max, mae_db_mean/std/min/max, ...}
│       └── config{}: evaluation parameters
│
└── backtest_metrics.png
    └── Contains 4 subplots:
        ├── RMSE distribution histogram
        ├── MAE distribution histogram
        ├── Error CDF curve (for coverage)
        └── Bias distribution histogram
```

---

## 📚 Data Structure Reference

### Input Tensors
```
Location: model_input/data/training/input/
Files: scene{0-4}_input.npy
Shape: (H=107, W=107, C=3)
Channels: [elevation, distance_map, frequency_log10]
Normalization: Varies (elevation 0-1, distance 0-1, frequency [-4, -1] log)
```

### Target Tensors
```
Location: model_input/data/training/target/
Files: scene{0-4}_target.npy
Shape: (H=107, W=107, C=1)
Content: RSS in dBm
Range: Typically -100 to -50 dBm
Normalization: Applied during loading (per model_input.py formula)
```

---

## 🎓 Learning Path

### For Quick Understanding (20 minutes)
1. Read [`ASSESSMENT_REPORT.md`](#assessment-report) (5 min)
2. Skim [`QUICK_REFERENCE.md`](#quick-reference) (5 min)
3. Run `python run_backtest.py --quick` (2 min)
4. Examine output (3 min)

### For Practical Implementation (1-2 hours)
1. Read [`BACKTEST_SUMMARY.md`](#backtest-summary)
2. Read [`QUICK_REFERENCE.md`](#quick-reference)
3. Run full `python run_backtest.py`
4. Analyze `backtest_results/`
5. Read relevant sections of [`EVALUATION_GUIDE.md`](#evaluation-guide)

### For Strategic Planning (2-3 hours)
1. Read [`ASSESSMENT_REPORT.md`](#assessment-report)
2. Read [`AIRMAP_COMPARISON.md`](#airmap-comparison)
3. Review performance expectations section
4. Plan data generation roadmap
5. Define success criteria for your project

### For Technical Mastery (4+ hours)
1. Read all documentation in order
2. Study `backtest_evaluation.py` source code
3. Study `models/diffusion.py` architecture
4. Run backtests with different configurations
5. Modify evaluation script for custom metrics

---

## 🔍 Finding What You Need

| I want to... | Read this | Then do this |
|---|---|---|
| Understand what was created | [ASSESSMENT_REPORT](#assessment-report) | Read "Framework Includes" |
| Run backtesting quickly | [QUICK_REFERENCE](#quick-reference) | Section "Run Backtesting in 2 Commands" |
| Interpret my results | [QUICK_REFERENCE](#quick-reference) | Section "Success Interpretation" |
| Understand metrics deeply | [EVALUATION_GUIDE](#evaluation-guide) | Read full document |
| Compare to AIRMap | [AIRMAP_COMPARISON](#airmap-comparison) | Read full document |
| Troubleshoot problems | [EVALUATION_GUIDE](#evaluation-guide) | Section "Common Pitfalls & Debugging" |
| See what to do next | [ASSESSMENT_REPORT](#assessment-report) | Section "Actionable Next Steps" |
| Understand data format | [EVALUATION_GUIDE](#evaluation-guide) | Section "Dataset Contract" |
| Know what RMSE value is "good" | [QUICK_REFERENCE](#quick-reference) | Section "Performance by Stage" |
| Learn advanced techniques | [EVALUATION_GUIDE](#evaluation-guide) | Section "Advanced Usage" |
| Plan for 1 month | [AIRMAP_COMPARISON](#airmap-comparison) | Section "Phase 2: Increase Training Data" |

---

## ✅ Checklist Before Backtesting

- [ ] Python environment configured (`.venv/bin/python`)
- [ ] Dataset loaded: `model_input/data/training/` contains 5 scenes
- [ ] Model available: Train via `models/model.py` OR use pretrained
- [ ] Read [`QUICK_REFERENCE.md`](#quick-reference) for metric understanding
- [ ] Understand expectations: 15-25 dB RMSE for 5 scenes is normal
- [ ] Run: `python run_backtest.py --quick` (2 min test)
- [ ] Check output: No errors, metrics printed to console

---

## 🚀 Your First Command

```bash
cd /Users/matthewgrech/ECE2T5F/ECE496/Diffusion
python run_backtest.py --quick
```

**Expected runtime**: 2-5 minutes  
**Expected output**:
1. Loading messages (dataset, model)
2. Progress bar "Evaluating: 100%"
3. Formatted report with RMSE/MAE/coverage statistics
4. Path to saved results

---

## 📞 Support Resources

### If you get an error:
1. Check [`EVALUATION_GUIDE.md`](#evaluation-guide) "Troubleshooting" section
2. Check [`QUICK_REFERENCE.md`](#quick-reference) "Quick Troubleshooting" table
3. Verify dataset exists: `ls model_input/data/training/input/`
4. Verify Python env: `/Users/matthewgrech/ECE2T5F/ECE496/Diffusion/.venv/bin/python --version`

### If results seem wrong:
1. Check [`QUICK_REFERENCE.md`](#quick-reference) "Performance by Stage"
2. Compare against RMSE expectations (15-25 dB for your dataset size)
3. Read [`EVALUATION_GUIDE.md`](#evaluation-guide) "Expected Results for Different Scenarios"
4. Look for patterns in `backtest_results/backtest_results.json`

### If you want to understand more:
1. [`EVALUATION_GUIDE.md`](#evaluation-guide) explains every metric
2. [`AIRMAP_COMPARISON.md`](#airmap-comparison) explains why AIRMap is better
3. Source code: `backtest_evaluation.py` and `models/diffusion.py`

---

## 📋 Document Statistics

| Document | Length | Read Time | Best For |
|---|---|---|---|
| [ASSESSMENT_REPORT](#assessment-report) | ~7 pages | 15 min | Overview |
| [QUICK_REFERENCE](#quick-reference) | ~4 pages | 5 min | During backtesting |
| [EVALUATION_GUIDE](#evaluation-guide) | ~10 pages | 20 min | Deep learning |
| [AIRMAP_COMPARISON](#airmap-comparison) | ~11 pages | 25 min | Strategy |
| [BACKTEST_SUMMARY](#backtest-summary) | ~6 pages | 15 min | Project context |

**Total reading time**: ~45-60 minutes for complete understanding

---

## 🎯 Success Indicators

You know the framework is working if:
- [ ] `python run_backtest.py --quick` runs without errors
- [ ] RMSE is between 10-30 dB (reasonable for 5 scenes)
- [ ] Coverage @ 5dB is between 10-50% (reasonable)
- [ ] You see a PNG plot at `backtest_results/backtest_metrics.png`
- [ ] JSON results in `backtest_results/backtest_results.json` are valid

---

## 🔄 Workflow Summary

```
1. Train Model (optional)
   models/model.py → saves checkpoints

2. Evaluate
   run_backtest.py → runs framework

3. Analyze Results
   backtest_results/ → RMSE/MAE/coverage

4. Interpret
   QUICK_REFERENCE.md → success criteria

5. Plan Next Steps
   AIRMAP_COMPARISON.md → optimization roadmap
```

---

**Status**: ✅ Framework Complete  
**Last Updated**: January 29, 2025  
**Ready to Use**: Yes  

**Next Action**: `python run_backtest.py --quick`
