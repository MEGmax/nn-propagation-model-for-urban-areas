# Model Assessment: Your Diffusion Model vs AIRMap Paper

## Executive Summary

Your conditional diffusion model and the AIRMap paper tackle the same **core problem**: predicting RSS maps for outdoor urban geometry in real-time, replacing expensive ray-tracing simulators.

**Key Difference**: 
- **AIRMap**: Deterministic U-Net autoencoder (single prediction per input)
- **Your Model**: Stochastic diffusion model (multiple plausible predictions, epistemic uncertainty)

This document analyzes strengths, gaps, and optimization opportunities.

---

## 1. Architecture Comparison

### AIRMap (Deterministic U-Net)
```
Elevation Map (2D)
     ↓
  U-Net Autoencoder
     ↓
RSS Map (deterministic)
```

**Strengths**:
- Simple, fast (4 ms inference on L40S GPU)
- Deterministic → easy to debug & validate
- Transfer learning friendly (pre-trained backbone)

**Weaknesses**:
- No uncertainty quantification
- Single mode → misses multi-modal solutions
- Requires extensive post-hoc calibration

### Your Model (Conditional Diffusion)
```
  Elevation + Distance + Frequency (3 channels)
        ↓
  Time-Conditional U-Net
        ↓
  Reverse Diffusion Sampling
        ↓
RSS Map (stochastic, with variance)
```

**Strengths**:
- ✓ Multiple predictions per scene → ensemble confidence
- ✓ Uncertainty quantification (pixel-wise variance)
- ✓ Richer condition space (distance + frequency)
- ✓ Generative → explores full RSS distribution

**Weaknesses**:
- ✗ Slower inference (multiple reverse steps)
- ✗ More complex architecture (time embeddings, diffusion scheduler)
- ✗ Requires more training data for convergence

---

## 2. Input Conditioning Analysis

### AIRMap Inputs
| Channel | Shape | Purpose |
|---------|-------|---------|
| Elevation | (H, W) | Building heights, terrain |

**Total**: 1 channel

### Your Model Inputs
| Channel | Shape | Purpose |
|---------|-------|---------|
| Elevation | (H, W) | Building heights, terrain |
| Distance | (H, W) | TX distance map (normalized 0-1) |
| Frequency | (H, W) | Broadcast frequency (log-scaled) |

**Total**: 3 channels

**Analysis**:
- Your model provides **richer scene context**
- Distance + frequency enable frequency-dependent effects (path loss exponent)
- However, missing: **material map** (concrete vs glass, which AIRMap doesn't use either)
- Could improve with: **LOS/NLOS mask**, **building density map**

---

## 3. Training Data Comparison

| Aspect | AIRMap | Your Model | Status |
|--------|--------|-----------|--------|
| Samples | 60,000 | 5 scenes | ⚠ **FAR under-resourced** |
| Coverage | Boston, 500m-3km | 5 small indoor scenes? | ⚠ **Limited diversity** |
| Validation | 10,000 held-out | None? | ⚠ **No train/val split** |
| Field measurements | 10,000+ | 0 | ⚠ **No real-world data** |

**Critical Gap**: You have **12,000x fewer samples** than AIRMap.

**Implications**:
1. Model may memorize training data rather than generalize
2. Cannot assess generalization to unseen scenes
3. Cannot compare to real radio measurements
4. High variance in per-scene performance

---

## 4. Expected Performance Scaling

Based on deep learning scaling laws:

```
Performance (RMSE in dB)
|
|  ✓ AIRMap (60k samples, 5 dB RMSE)
|  |
| 8|
| 6|  
| 4|
|  |      ⚠ Your model (5 samples, ~15-25 dB RMSE?)
|  |_______|__________|__________|__________|
    1      10        100       1000      10000    60000
                    Training Samples (log scale)
```

**Prediction**: Your model likely achieves **15-25 dB RMSE** at 5 samples due to:
- Insufficient data to learn spatial patterns
- High per-scene variance
- Overfitting to scene-specific features

**To match AIRMap's 5 dB RMSE**: Need ~1000-10,000 diverse scenes

---

## 5. Evaluation Metrics: What to Expect

### Realistic Baseline Expectations

Given **5 training scenes** and **diffusion-based architecture**:

| Metric | Your Model | AIRMap | Comment |
|--------|-----------|--------|---------|
| RMSE (dB) | 15-25 | < 5 | You need more data |
| MAE (dB) | 12-20 | 3-5 | Same issue |
| Bias (dB) | ±5 | ≈0 | Expect calibration offset |
| Coverage (5dB) | 20-40% | 70%+ | Spatial accuracy is poor |
| Inference time | 500ms-2s | 4ms | Multiple diffusion steps |
| Training time | 1-2 hours | Days | On your data size |

### What This Means

1. **RMSE 15-25 dB is not a failure** with only 5 scenes
2. **Coverage <40% is expected** at this data scale
3. **Your model's advantage** is uncertainty quantification, not raw accuracy

---

## 6. Key Insights from AIRMap Paper

### They Did Right (You Should Too)
1. ✓ **Large, diverse dataset**: 60k samples from real Boston 3D city model
2. ✓ **Robust evaluation**: 10k held-out test set
3. ✓ **Realistic constraints**: Coverage areas 500m-3km
4. ✓ **Validation against real data**: Field measurements on validation
5. ✓ **End-to-end integration**: Tested in Colosseum emulator

### Surprising AIRMap Findings
- Simple U-Net **outperforms** complex architectures
- Single elevation input is sufficient (no extra channels needed)
- Transfer learning calibration (20% field data) **essential** for real deployment
- Spectral efficiency error < 5% (not just RSS accuracy)

### Implications for Your Model
- Diffusion **complexity** may not be justified by accuracy gain
- Focus on **data quality** over architecture sophistication
- Plan **calibration phase** with field measurements
- Test end-to-end in simulation framework (Sionna/Colosseum)

---

## 7. Optimization Roadmap

### Phase 1: Baseline (Current)
**Goal**: Establish model behavior, identify issues

**Actions**:
1. Run backtest_evaluation.py on 5 training scenes
2. Document baseline RMSE/MAE
3. Check if model is learning (loss decreasing?)
4. Analyze error distribution (is it Gaussian?)

**Expected Output**: RMSE 15-30 dB (normal for 5 samples)

---

### Phase 2: Increase Training Data
**Goal**: Reach 100-500 scenes for real improvement

**Actions**:
1. Generate more scenes via `scene_generation/` pipeline
2. Implement train/val/test split (70/15/15)
3. Monitor overfitting (train loss vs val loss)
4. Re-evaluate after 100, 200, 500 scenes

**Expected Progress**:
- 100 scenes → ~10-15 dB RMSE
- 500 scenes → ~7-10 dB RMSE
- 2000+ scenes → ~5-7 dB RMSE (approaching AIRMap range)

---

### Phase 3: Model Improvements
**Goal**: Squeeze out last 1-2 dB of accuracy

**Actions**:
1. Add **material map** as 4th channel (if available from Sionna)
2. Implement **skip connections** between input/output
3. Test **larger models** (base_ch=64, channel_mults=(1,2,4,8))
4. Try **different diffusion schedulers** (cosine, sqrt)

**Expected Gain**: 1-3 dB improvement

---

### Phase 4: Calibration & Validation
**Goal**: Prepare for real-world deployment

**Actions**:
1. Collect field measurements (if possible)
2. Implement post-hoc calibration layer (linear adjustment)
3. Validate on real ray-tracing data (Sionna ground truth)
4. Benchmark inference speed & memory

**Expected Results**: Reduce real-world error by 50-70%

---

## 8. Backtesting Action Items

### Immediate (Run Today)
```bash
# 1. Train for 10 epochs on 5 scenes
python models/model.py --epochs 10 --batch-size 2

# 2. Evaluate
python backtest_evaluation.py

# 3. Check results
cat backtest_results/backtest_results.json
```

**Questions to Answer**:
- Is model loss decreasing (training working)?
- What is RMSE on training set? (Should be < training target)
- What is per-scene variance? (High = overfitting)
- Any NaN/Inf errors? (Data issue or model bug)

### Week 1
- [ ] Generate 10 additional scenes (double dataset size)
- [ ] Re-train and re-evaluate
- [ ] Plot error vs spatial location (identify failure patterns)
- [ ] Compare RSS statistics: predicted vs ground truth

### Month 1
- [ ] Target 50-100 scenes
- [ ] Implement proper train/val/test split
- [ ] Add **material conditioning** (if available)
- [ ] Profile inference time vs accuracy trade-off

---

## 9. Diagnostic Visualizations to Add

### Per-Scene Analysis
```python
# For each test scene, save:
1. Side-by-side: prediction vs ground truth (heatmaps)
2. Difference map: error = prediction - ground truth
3. Error histogram: distribution of per-pixel errors
4. Scatter plot: predicted vs ground truth (should be diagonal)
```

### Cross-Scene Trends
```python
# Plot RMSE vs:
1. Scene size (does accuracy degrade on larger areas?)
2. TX distance from center (is far-field prediction worse?)
3. Elevation range (complex topography = harder?)
4. Distance from TX (path loss accuracy)
```

### Frequency Analysis
```python
# If you evaluate at multiple frequencies:
1. RMSE vs frequency (path loss exponent learning?)
2. Coverage vs frequency (higher freq = worse prediction?)
```

---

## 10. Realistic Next Steps

### If You Want to Match AIRMap's 5 dB RMSE
1. **Double down on data**: Generate 1000+ diverse scenes
2. **Simplify model**: Consider deterministic U-Net (like AIRMap)
3. **Add real validation**: Use Sionna ray-tracing as ground truth
4. **Optimize infrastructure**: Make generation pipeline 10x faster

**Timeline**: 2-3 months of focused work

### If You Want to Differentiate from AIRMap
1. **Keep diffusion architecture** (uncertainty quantification is your strength)
2. **Focus on small-data regime**: "Works well with 10-50 scenes"
3. **Add epistemic uncertainty**: Report prediction confidence intervals
4. **Develop calibration protocol**: Field measurement-based fine-tuning

**Timeline**: 1-2 months + field validation

---

## 11. Key Takeaways

✓ **Your model is scientifically sound** (diffusion theory is solid)
⚠ **Your data is 12,000x smaller** than AIRMap's benchmark
⚠ **Raw RMSE may not be best metric** for judging success at small scale
✓ **Uncertainty quantification** is a genuine advantage over AIRMap
⚠ **Architecture complexity** may not be justified without more data

### Bottom Line for Backtesting

```
Expected RMSE: 15-25 dB (normal at 5 scenes)
Target RMSE: < 10 dB (good progress)
AIRMap RMSE: < 5 dB (requires 1000+ scenes)

Success = Showing learning progress + stable predictions + reasonable uncertainty
Failure = RMSE > 30 dB OR unstable/NaN predictions
```

---

## Appendix: Quick Reference

### Run Evaluation Now
```bash
cd /Users/matthewgrech/ECE2T5F/ECE496/Diffusion
python backtest_evaluation.py 2>&1 | tee backtest.log
```

### Check Results
```bash
cat backtest_results/backtest_results.json | python -m json.tool | head -100
```

### Expected Log Output
```
Loading dataset...
✓ Loaded 5 samples
Using device: cuda
Model parameters: 12.34M
Starting backtesting on 5 scenes...
Evaluating: 100%|███| 5/5 [3:45<00:00, 45s/batch]
Results saved to backtest_results/backtest_results.json
```

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-29  
**Status**: Ready for backtesting
