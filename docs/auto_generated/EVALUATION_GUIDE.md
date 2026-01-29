# Model Evaluation & Backtesting Framework

## Overview

This framework evaluates the conditional diffusion model's performance on RSS (Received Signal Strength) prediction for outdoor urban geometry scenarios. The evaluation methodology is based on the **AIRMap paper** (Saeizadeh et al., arXiv:2511.05522), which demonstrates state-of-the-art performance for radio map estimation.

## Key References

- **AIRMap Paper**: "AI-Generated Radio Maps for Wireless Digital Twins"
  - Achieves **< 5 dB RMSE** on 60,000 Boston-area samples
  - Targets **7000x speedup** vs GPU ray-tracing
  - Uses single-input U-Net on 2D elevation maps
  
- **Your Model**: Conditional diffusion with FiLM-based scene conditioning
  - Uses elevation + distance + frequency as input channels
  - Diffusion timestep embedding for temporal conditioning
  - Generates pixel-wise RSS predictions

---

## Evaluation Metrics (AIRMap-Based)

### Primary Metrics

| Metric | Description | Target | Why Important |
|--------|-------------|--------|---------------|
| **RMSE (dB)** | Root Mean Squared Error | < 5 dB | Overall accuracy vs ground truth |
| **MAE (dB)** | Mean Absolute Error | < 5 dB | Average prediction error magnitude |
| **Median Error (dB)** | Median absolute error | < 3 dB | Robust to outliers |
| **Bias (dB)** | Mean prediction error | ≈ 0 dB | Systematic under/over-prediction |

### Secondary Metrics

- **Coverage**: % of pixels within error thresholds (3 dB, 5 dB, 10 dB)
- **Pearson Correlation**: Spatial prediction structure vs ground truth
- **Error Percentiles**: 25th, 50th, 75th, 90th percentile errors
- **Per-Scene Variation**: Min/max/std of metrics across scenes

---

## Installation & Setup

### Requirements

```bash
pip install torch numpy scipy scikit-image matplotlib seaborn tqdm
```

### Data Structure

```
model_input/data/training/
├── input/
│   ├── scene0_input.npy      # (H, W, 3) [elevation, distance, frequency]
│   ├── scene1_input.npy
│   └── ...
└── target/
    ├── scene0_target.npy     # (H, W, 1) RSS in dBm
    ├── scene1_target.npy
    └── ...
```

**Normalization Convention** (from `model_input.py`):
- RSS_DB_FLOOR = -100.0 dBm (lowest RSS value)
- RSS_DB_SCALE = 50.0 dB (normalization range)
- Normalized range: [-1, 1]
- Denormalization: `rss_dbm = normalized * 50 + (-100)`

---

## Usage

### 1. Training the Model

```bash
cd models
python model.py \
    --input-dir ../model_input/data/training/input \
    --target-dir ../model_input/data/training/target \
    --epochs 50 \
    --batch-size 4 \
    --lr 2e-4 \
    --save-every 5 \
    --checkpoint-dir ./checkpoints
```

**Options**:
- `--resume`: Resume from latest checkpoint
- `--num-workers`: Parallel data loading workers
- `--timesteps`: Diffusion timesteps (default: 1000)

**Output**:
- `checkpoints/model_epoch*.pt`: Periodic checkpoints
- `checkpoints/training_metadata.json`: Training metadata

---

### 2. Backtesting & Evaluation

```bash
cd /Users/matthewgrech/ECE2T5F/ECE496/Diffusion
python backtest_evaluation.py
```

This runs comprehensive evaluation with:
- ✓ Per-scene metrics (RMSE, MAE, bias, etc.)
- ✓ Aggregate statistics across all scenes
- ✓ Error distribution plots (histograms, CDF)
- ✓ Spatial accuracy visualizations
- ✓ Formatted evaluation report

**Output**:
```
backtest_results/
├── backtest_results.json     # Detailed per-scene & aggregate metrics
└── backtest_metrics.png      # Diagnostic plots
```

---

## Interpreting Results

### RMSE Performance Levels

| RMSE (dB) | Assessment | Comparison to AIRMap |
|-----------|-----------|---------------------|
| < 5 | ✓ Excellent | Matches state-of-the-art |
| 5-10 | ✓ Good | Reasonable for initial model |
| 10-20 | ⚠ Fair | Needs improvement |
| > 20 | ✗ Poor | Fundamental issues |

### Coverage Analysis

Coverage measures what % of predictions fall within error thresholds:

```
Within 3 dB:  High-quality predictions (ideally > 40%)
Within 5 dB:  Good predictions (ideally > 60%)
Within 10 dB: Acceptable predictions (ideally > 80%)
```

Example:
- **"70% within 5 dB"** means 70% of pixels predicted within ±5 dB

### Bias Interpretation

- **Bias ≈ 0 dB**: Well-calibrated, no systematic error
- **Bias > 2 dB**: Model systematically over-predicts RSS
- **Bias < -2 dB**: Model systematically under-predicts RSS

Systematic bias can often be corrected with post-processing.

---

## Expected Results for Different Scenarios

### Baseline (Early Training)
- RMSE: 15-30 dB
- MAE: 10-20 dB
- Coverage (5 dB): 20-40%
- **Indicates**: Model still learning RSS physics

### Intermediate (10-20 epochs)
- RMSE: 8-15 dB
- MAE: 6-12 dB
- Coverage (5 dB): 40-65%
- **Indicates**: Good convergence, room for improvement

### Well-Trained Model
- RMSE: 4-7 dB ✓
- MAE: 3-5 dB ✓
- Coverage (5 dB): 70-85% ✓
- Coverage (10 dB): 90%+ ✓
- **Indicates**: Ready for deployment

---

## Diagnostic Tips

### High RMSE but Low Bias
- Model adds noise/variance to predictions
- Check: Ensemble averaging (averaging multiple diffusion samples)
- Solution: Increase diffusion steps, train longer

### High Bias
- Model systematically mis-calibrated
- Check: Input normalization, target normalization range
- Solution: Recalibrate with adjustment layer, retrain with data augmentation

### Poor Spatial Accuracy (Low correlation)
- Model not capturing spatial structure from elevation
- Check: Condition encoding path in model
- Solution: Improve `cond_proj` network, increase conditioning dimensions

### Slow Training Convergence
- Check: Learning rate (may be too high/low)
- Check: Batch size (too small = noisy gradients)
- Solution: Reduce LR to 1e-4, increase batch size to 8-16

---

## Model Architecture Reference

```
TimeCondUNet:
├── Input: (B, 1, H, W) noisy RSS + (B, 3, H, W) conditions
├── Time Embedding: Sinusoidal PE → MLP → (B, 128)
├── Condition Proj: Conv + Global Avg Pool → MLP → (B, 64)
├── Encoder: 3 downsampling blocks (ResBlocks + FiLM)
├── Middle: 2 ResBlocks (time & condition modulated)
├── Decoder: 3 upsampling blocks (skip connections)
└── Output: (B, 1, H, W) predicted noise (epsilon-prediction)
```

**FiLM Modulation**:
- Time embedding → scale & shift per channel
- Condition vector → additional scale & shift
- Combined: `h = h * (1 + time_scale) + time_shift * (1 + cond_scale) + cond_shift`

---

## Advanced Usage

### Custom Evaluation Script

```python
from backtest_evaluation import RadioMapEvaluator, RSSNormalizer

evaluator = RadioMapEvaluator()

# Your predictions and ground truth
metrics = evaluator.compute_metrics(
    predicted=my_predictions,  # (H, W)
    ground_truth=my_ground_truth,  # (H, W)
    scenario_name="custom_test"
)

print(f"RMSE: {metrics['rmse_db']:.2f} dB")
print(f"MAE: {metrics['mae_db']:.2f} dB")
```

### Ensemble Inference

For more robust predictions, average multiple diffusion samples:

```python
sampler = DiffusionSampler(model, device='cuda')
samples = sampler.sample(
    cond_img=conditioning,
    num_samples=10,  # Generate 10 samples per input
    steps=50
)
ensemble_prediction = np.mean(samples, axis=0)
```

---

## Comparison with AIRMap

| Aspect | AIRMap | Your Model |
|--------|--------|-----------|
| Architecture | U-Net autoencoder | Conditional diffusion |
| Inputs | Elevation only | Elevation + distance + freq |
| Training samples | 60,000 | 5 (currently) |
| RMSE target | < 5 dB | TBD |
| Inference speed | 4 ms (L40S) | TBD |
| Advantages | Fast, simple | Generative, uncertainty-aware |

---

## Troubleshooting

### "RMSE > 20 dB"
- Check data normalization: Are inputs/targets in expected ranges?
- Verify shapes: Model expects (B, 3, H, W) input, (B, 1, H, W) target
- Look for NaN/Inf: `np.isinf(metrics['rmse_db'])`

### "Correlation ≈ 0"
- Model not learning to condition on inputs
- Check condition encoder (`cond_proj`): Is it learning?
- Increase `cond_emb_dim` to 128+

### "Out of memory"
- Reduce `batch_size` (default: 4)
- Reduce image resolution or spatial size
- Use gradient accumulation

### "Loss doesn't decrease"
- Learning rate too high: Try 1e-4 instead of 2e-4
- Learning rate too low: Try 1e-3
- Check for NaN in data: `np.any(np.isnan(data))`

---

## References

1. **AIRMap Paper**: Saeizadeh et al., 2025. "AIRMap -- AI-Generated Radio Maps for Wireless Digital Twins" arXiv:2511.05522
2. **Diffusion Models**: Ho et al., 2020. "Denoising Diffusion Probabilistic Models" (DDPM)
3. **Conditional Generation**: Dhariwal & Nichol, 2021. "Diffusion Models Beat GANs on Image Synthesis"
4. **Radio Propagation**: Rappaport, 2002. "Wireless Communications: Principles and Practice"

---

## Contact & Questions

For evaluation framework questions or to report issues, refer to the project documentation.
