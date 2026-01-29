# Quick Reference: Backtesting Your Model

## 🚀 Run Backtesting in 2 Commands

```bash
# Full evaluation (~10 minutes)
python run_backtest.py

# Quick test (~2 minutes)
python run_backtest.py --quick
```

## 📊 What You'll Get

| Output | Location | Purpose |
|--------|----------|---------|
| Metrics JSON | `backtest_results/backtest_results.json` | Raw data: RMSE, MAE, bias, coverage |
| Plots | `backtest_results/backtest_metrics.png` | Visual: RMSE hist, error CDF, bias |
| Console Report | Printed to terminal | Summary: mean/std/min/max across scenes |

## 📈 Key Metrics Explained

### Primary (Most Important)

```
RMSE (dB)
├─ What: Root mean squared error in dB
├─ Your target: < 10 dB (good), < 20 dB (okay)
├─ AIRMap achieves: < 5 dB
└─ ✓ Lower is better

MAE (dB)
├─ What: Average absolute error
├─ Similar interpretation to RMSE
└─ Usually slightly lower than RMSE

Coverage (% within threshold)
├─ "within 5 dB": % of pixels with |error| ≤ 5 dB
├─ Your target: > 60% (good)
├─ AIRMap achieves: > 70%
└─ ✓ Higher is better
```

### Secondary (Supporting Info)

```
Bias (dB)
├─ What: Systematic over/under-prediction
├─ Ideal: 0 dB (well-calibrated)
├─ ±2-5 dB: Normal, can be corrected
└─ > ±10 dB: Data/model issue

Median Error (dB)
├─ What: 50th percentile (robust to outliers)
├─ Usually better than RMSE for skewed distributions
└─ More stable metric at small sample size

Pearson Correlation
├─ What: Does prediction track ground truth spatially?
├─ Range: -1 (inverse) to +1 (perfect)
├─ Target: > 0.7 (good spatial structure)
└─ < 0.5: Model not learning conditioning
```

## ✅ Success Interpretation

```
RMSE < 10 dB      ✓ Model is learning well
                  → Spatial patterns captured
                  → Realistic error magnitude

RMSE 10-20 dB     ⚠ Decent for early development
                  → Need more training data
                  → Or improve model architecture

RMSE > 20 dB      ⚠ Needs improvement
                  → Check data normalization
                  → Verify model training
                  → Could be just scale (5 scenes)

Coverage > 70%    ✓ Excellent spatial accuracy
@ 5 dB

Coverage < 40%    ⚠ Model is uncertain/variable
@ 5 dB            → Small dataset limitation
                  → Expected at 5 scenes

Bias = 0 dB       ✓ Perfect calibration
                  (unlikely but great if true)

Bias ± 5 dB       ⚠ Systematic offset
                  → Can be corrected post-hoc
                  → Not a major issue

Bias > ± 10 dB    ✗ Significant problem
                  → Check normalization
                  → Verify loss function
```

## 🎯 Performance by Stage

### Baseline (Your Current: 5 Scenes)
```
Expected RMSE:    15-25 dB
Expected MAE:     12-18 dB
Expected Coverage (5dB): 20-40%
Interpretation:   NORMAL - lots of room to improve
```

### Good Progress (50 Scenes)
```
Expected RMSE:    10-15 dB
Expected MAE:     8-12 dB
Expected Coverage (5dB): 40-60%
Interpretation:   Learning working, needs more data
```

### Competitive (500 Scenes)
```
Expected RMSE:    7-10 dB
Expected MAE:     5-8 dB
Expected Coverage (5dB): 65-80%
Interpretation:   Approaching AIRMap range
```

### State-of-Art (2000+ Scenes)
```
Expected RMSE:    < 5 dB ✓
Expected MAE:     3-5 dB ✓
Expected Coverage (5dB): > 80% ✓
Interpretation:   Competitive with AIRMap
```

## 🔧 Quick Troubleshooting

| Problem | Check | Solution |
|---------|-------|----------|
| RMSE > 25 dB | Is model trained? | Run `python models/model.py` first |
| NaN in metrics | Data normalization | Check input ranges |
| Very high bias | Target centering | Verify RSS normalization in model_input.py |
| Low correlation | Conditioning | Check if model receiving inputs correctly |
| Out of memory | Batch size | Reduce from 2 to 1 |
| Slow evaluation | Num samples | Use `--quick` flag or reduce `--samples-per-scene` |

## 📊 Metric Formulas

```
RMSE = sqrt(mean((predicted - ground_truth)²))

MAE = mean(|predicted - ground_truth|)

Bias = mean(predicted - ground_truth)  [signed, can be negative]

Coverage @ Xdb = (# pixels with |error| ≤ X) / total_pixels

Pearson Correlation = cov(predicted, ground_truth) / (std_pred × std_gt)
```

## 🔑 What Each Plot Means

### RMSE Distribution Histogram
- **Left peak**: Easier scenes (lower error)
- **Right tail**: Harder scenes (higher error)
- **Ideal**: Narrow, centered on target RMSE

### Error CDF (Cumulative Distribution Function)
- **Y-axis**: % of pixels below error threshold
- **Steep line**: Most predictions concentrated in narrow error range (good)
- **Flat line**: Errors spread across wide range (bad)
- **Cross @ 5dB**: Coverage threshold visualization

### Bias Distribution
- **Centered on 0**: Model is well-calibrated ✓
- **Shifted left**: Model over-predicts (bias < 0)
- **Shifted right**: Model under-predicts (bias > 0)

## 🎓 Real Numbers from AIRMap Paper

As a benchmark:
```
AIRMap (60k Boston scenes):
├─ RMSE: 4.8 dB
├─ MAE: 3.2 dB
├─ Median: 2.1 dB
├─ Coverage (5dB): 84%
└─ Inference: 4 ms per map

Your Model Target (1000 scenes):
├─ RMSE: 5-7 dB (reasonable)
├─ MAE: 4-5 dB (reasonable)
├─ Coverage (5dB): 70% (reasonable)
└─ Inference: 0.5-2 sec (acceptable for uncertainty)
```

## 📋 Backtest Workflow

```
1. Train model (optional)
   python models/model.py --epochs 10

2. Run evaluation
   python run_backtest.py --quick

3. Check results
   cat backtest_results/backtest_results.json | head -50
   open backtest_results/backtest_metrics.png

4. Iterate
   - If RMSE > 20: train longer or add data
   - If RMSE 10-20: collect more scenes
   - If RMSE < 10: try model improvements
```

## 🚨 Red Flags

```
✗ RMSE = NaN               → Data problem, stop
✗ RMSE = Inf               → Numerical overflow, stop
✗ Loss increases w/ epochs → Learning rate too high
✗ Coverage = 0% @ 10dB     → Model completely broken
✗ Correlation < 0          → Inverted predictions?
✗ Loss = NaN after epoch 3 → Data corruption
```

## ✅ Green Lights

```
✓ Loss decreasing         → Training working
✓ RMSE improving w/ data  → Scaling correctly
✓ Bias ≈ ±2 dB            → Well-calibrated
✓ Correlation > 0.5       → Spatial learning
✓ Coverage 40%+ @ 5dB     → Reasonable accuracy
✓ Per-scene consistency   → Model is stable
```

---

## 🎯 Your Action Items

### Today (5 min)
- [ ] Read this document
- [ ] Run: `python run_backtest.py --quick`
- [ ] Check console output for errors

### This Week (1-2 hours)
- [ ] Run: `python run_backtest.py` (full evaluation)
- [ ] Read the generated JSON results
- [ ] Analyze backtest_metrics.png
- [ ] Compare RMSE to expectations above

### Next 2 Weeks (ongoing)
- [ ] Generate 10+ more training scenes
- [ ] Re-run backtesting with more data
- [ ] Plot RMSE vs dataset size (should decrease)
- [ ] Identify spatial failure patterns

---

**Last Updated**: 2025-01-29  
**Status**: Ready to Run
