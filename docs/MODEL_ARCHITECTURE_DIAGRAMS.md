# Model Architecture and Data Flow Diagrams
## Visual Reference for Understanding the System

---

## 1. OVERALL SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PHASE                              │
└─────────────────────────────────────────────────────────────────┘

Step 1: DATA GENERATION (Offline)
┌─────────────────┐
│  Blender/       │
│  Mitsuba        │──────> Urban Scene     ┌──────────────┐
│  (3D modeling)  │        Geometry        │   Sionna     │
└─────────────────┘                        │  Ray-Tracing │
                                           │  Simulator   │
┌─────────────────┐                        └──────────────┘
│  Transmitter    │                               │
│  Configuration  │───────────────────────────────┘
│  (pos, freq)    │                               │
└─────────────────┘                               ▼
                                          Ground Truth
                                          RSS Map (.npy)


Step 2: DATA PREPROCESSING
┌──────────────┐      ┌───────────────────────────────┐
│ Elevation    │      │   Input Tensor (H, W, 3)      │
│ Map          │────> │   - Elevation (normalized)     │
└──────────────┘      │   - Distance from TX           │
                      │   - Frequency                  │
┌──────────────┐      └───────────────────────────────┘
│ TX Position  │────>
└──────────────┘

┌──────────────┐      ┌───────────────────────────────┐
│ Ground Truth │      │   Target Tensor (H, W, 1)     │
│ RSS Map      │────> │   - RSS in dBm                │
└──────────────┘      │   - Normalized to [-1, 1]     │
                      └───────────────────────────────┘


Step 3: TRAINING LOOP
┌────────────────────────────────────────────────────┐
│                                                    │
│  Input Tensor ──┐                                 │
│                  │                                 │
│                  ├──> TimeCondUNet ──> Predict    │
│  Target Tensor ─┤                       Noise     │
│  (+ noise)       │                                 │
│                  │                                 │
│  Timestep ───────┘                                 │
│                                                    │
│  Loss = MSE(Predicted Noise, Actual Noise)        │
│  Backpropagate & Update Weights                   │
└────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────┐
│                    INFERENCE PHASE                              │
└─────────────────────────────────────────────────────────────────┘

Step 1: PREPARE INPUT
┌──────────────┐      ┌───────────────────────────────┐
│ New Urban    │      │   Input Tensor (H, W, 3)      │
│ Scene        │────> │   - Elevation                  │
│ + TX Config  │      │   - Distance from TX           │
└──────────────┘      │   - Frequency                  │
                      └───────────────────────────────┘

Step 2: REVERSE DIFFUSION
┌────────────┐      ┌──────────────┐      ┌──────────────┐
│ Pure Noise │      │ Denoise      │      │ Clean RSS    │
│  (Random)  │ ───> │ 50-100 Steps │ ───> │ Prediction   │
└────────────┘      │ (30-120 sec) │      └──────────────┘
                    └──────────────┘
                          ▲
                          │
                   Input Tensor
                   (condition)

Step 3: OUTPUT
┌──────────────┐      ┌───────────────────────────────┐
│ RSS Heatmap  │      │  Applications:                │
│ (dBm values) │ ───> │  - Coverage analysis          │
└──────────────┘      │  - Transmitter placement      │
                      │  - Network optimization       │
                      └───────────────────────────────┘
```

---

## 2. MODEL ARCHITECTURE DETAILED

```
TimeCondUNet Architecture
══════════════════════════════════════════════════════════════════

INPUT STAGE
───────────
                Input Image (Noisy RSS)
                     (B, 1, H, W)
                          ▼
                   ┌──────────────┐
                   │  Conv 3×3    │
                   │  in=1, out=32│
                   └──────────────┘
                          ▼
                    (B, 32, H, W)


TIME & CONDITION EMBEDDING
──────────────────────────
   Timestep t (B,)             Condition (B, 3, H, W)
        ▼                              ▼
   ┌─────────────┐              ┌──────────────┐
   │ Sinusoidal  │              │  Conv 3×3    │
   │ Encoding    │              │  +ReLU       │
   └─────────────┘              │  +AvgPool    │
        ▼                       │  +Linear     │
   (B, 128)                     └──────────────┘
        │                              ▼
        │                         (B, 64)
        │                              │
        └────────┬─────────────────────┘
                 │
                 │ FiLM Conditioning
                 │ (Scale + Shift)
                 ▼


ENCODER (Downsampling Path)
────────────────────────────
Level 1: (B, 32, H, W)
    ├──> ResBlock (32→32)  ──┐
    ├──> ResBlock (32→32)  ──┤ Skip Connection 1
    └──> AvgPool2d         ──┘
              ▼
Level 2: (B, 64, H/2, W/2)
    ├──> ResBlock (32→64)  ──┐
    ├──> ResBlock (64→64)  ──┤ Skip Connection 2
    └──> AvgPool2d         ──┘
              ▼
Level 3: (B, 128, H/4, W/4)
    ├──> ResBlock (64→128) ──┐
    ├──> ResBlock (128→128)──┤ Skip Connection 3
    └──> AvgPool2d         ──┘
              ▼


BOTTLENECK
──────────
(B, 128, H/8, W/8)
    ├──> ResBlock (128→128)
    ├──> ResBlock (128→128)
    └──> (B, 128, H/8, W/8)


DECODER (Upsampling Path)
──────────────────────────
Level 3: (B, 128, H/8, W/8)
    ├──> Upsample2x
    ├──> Concat with Skip 3
    ├──> ResBlock (256→128)  [256 = 128 upsampled + 128 skip]
    └──> ResBlock (128→128)
              ▼
Level 2: (B, 64, H/4, W/4)
    ├──> Upsample2x
    ├──> Concat with Skip 2
    ├──> ResBlock (128→64)   [128 = 64 upsampled + 64 skip]
    └──> ResBlock (64→64)
              ▼
Level 1: (B, 32, H/2, W/2)
    ├──> Upsample2x
    ├──> Concat with Skip 1
    ├──> ResBlock (64→32)    [64 = 32 upsampled + 32 skip]
    └──> ResBlock (32→32)
              ▼


OUTPUT STAGE
────────────
(B, 32, H, W)
    ├──> GroupNorm(8)
    ├──> SiLU activation
    └──> Conv 3×3 (32→1)
              ▼
    Predicted Noise (B, 1, H, W)
```

---

## 3. RESIDUAL BLOCK DETAIL

```
ResBlock with FiLM Conditioning
════════════════════════════════

Input: h (B, C_in, H, W)
Time Emb: t_emb (B, 128)
Cond Emb: c_emb (B, 64)
────────────────────────────

    h (B, C_in, H, W)
         │
         ├────────────────────────────┐ Residual
         │                            │ Connection
         ▼                            │
    ┌─────────┐                       │
    │ Conv 3×3│ (C_in → C_out)        │
    └─────────┘                       │
         │                            │
         ▼                            │
    ┌──────────┐                      │
    │GroupNorm │                      │
    └──────────┘                      │
         │                            │
         ▼                            │
    ┌─────────┐                       │
    │  SiLU   │                       │
    └─────────┘                       │
         │                            │
         ├──FiLM (Time)───────────────┤
         │  t_emb → Linear            │
         │  → (scale, shift)          │
         │  h = h * (1+scale) + shift │
         │                            │
         ├──FiLM (Condition)──────────┤
         │  c_emb → Linear            │
         │  → (scale, shift)          │
         │  h = h * (1+scale) + shift │
         │                            │
         ▼                            │
    ┌─────────┐                       │
    │ Conv 3×3│ (C_out → C_out)       │
    └─────────┘                       │
         │                            │
         ▼                            │
    ┌──────────┐                      │
    │GroupNorm │                      │
    └──────────┘                      │
         │                            │
         ▼                            │
    ┌─────────┐                       │
    │  SiLU   │                       │
    └─────────┘                       │
         │                            │
         ▼                            │
    ┌─────────┐                       │
    │ Dropout │                       │
    └─────────┘                       │
         │                            │
         ├────────────────────────────┘
         │  (Add residual)
         ▼
    Output (B, C_out, H, W)
```

---

## 4. DIFFUSION PROCESS VISUALIZATION

```
FORWARD DIFFUSION (Training - Add Noise)
═════════════════════════════════════════

t=0             t=250           t=500           t=750          t=1000
Clean           Slightly        Half            Very           Pure
RSS Map         Noisy           Noisy           Noisy          Noise
────────        ────────        ────────        ────────       ────────
[Heatmap]       [Heatmap +]     [Noise +]       [Mostly]       [Random]
[Clear]         [Some noise]    [Faint map]     [noise]        [Gaussian]

    │               │               │               │              │
    └───────────────┴───────────────┴───────────────┴──────────────┘
                    Add Gaussian noise gradually
                    x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε


REVERSE DIFFUSION (Inference - Remove Noise)
═════════════════════════════════════════════

t=1000          t=750           t=500           t=250          t=0
Pure            Very            Half            Slightly       Clean
Noise           Noisy           Noisy           Noisy          RSS Map
────────        ────────        ────────        ────────       ────────
[Random]        [Mostly]        [Noise +]       [Heatmap +]    [Heatmap]
[Gaussian]      [noise]         [Faint map]     [Some noise]   [Clear]

    │               │               │               │              │
    └───────────────┴───────────────┴───────────────┴──────────────┘
              Model predicts and removes noise at each step
              x_{t-1} = μ(x_t, t, condition) + σ_t·z
              where μ is computed by model


At each timestep t:
┌─────────────────────────────────────────────────────────┐
│  x_t (noisy image) ──┐                                  │
│                       ├──> TimeCondUNet ──> ε_pred     │
│  condition (scene) ───┤                                 │
│  t (timestep) ────────┘                                 │
│                                                         │
│  x_{t-1} = (x_t - β_t·ε_pred) / √α_t + σ_t·z           │
└─────────────────────────────────────────────────────────┘
```

---

## 5. DATA FLOW THROUGH SYSTEM

```
TRAINING DATA FLOW
══════════════════════════════════════════════════════

Scene Files                    Loaded Tensors
───────────────────────────────────────────────────────

scene_0/
├─ elevation.npy ────────────> Input[0, 0, :, :]
├─ rss_values.npy ───────────> Target[0, 0, :, :]
└─ tx_metadata.json
   ├─ position ──────────────> Input[0, 1, :, :] (distance map)
   └─ frequency ─────────────> Input[0, 2, :, :] (const)

                                    ▼
                             DataLoader (batch=4)
                                    ▼
                    ┌───────────────────────────────┐
                    │  Input Batch (4, 3, H, W)     │
                    │  Target Batch (4, 1, H, W)    │
                    └───────────────────────────────┘
                                    ▼
                             Training Loop
                                    ▼
                    ┌───────────────────────────────┐
                    │ 1. Sample timesteps t         │
                    │ 2. Add noise to targets       │
                    │ 3. Model predicts noise       │
                    │ 4. Compute loss               │
                    │ 5. Backprop                   │
                    └───────────────────────────────┘


INFERENCE DATA FLOW
══════════════════════════════════════════════════════

User provides:                 System generates:
───────────────────────────────────────────────────────

New Scene
├─ Elevation map ──────────┐
└─ TX position/freq ───────┤
                           ├──> Input Tensor (1, 3, H, W)
                           │
                           └──> Diffusion Loop (50 steps)
                                      │
                           ┌──────────┴──────────┐
                           │  t=50: Pure noise   │
                           │  t=49: Denoise      │
                           │  t=48: Denoise      │
                           │  ...                │
                           │  t=1:  Denoise      │
                           │  t=0:  Clean RSS    │
                           └─────────────────────┘
                                      │
                                      ▼
                           RSS Prediction (1, 1, H, W)
                                      │
                                      ▼
                           ┌─────────────────────┐
                           │ Visualization (PNG) │
                           │ Quantitative Metrics│
                           │ JSON Export         │
                           └─────────────────────┘
```

---

## 6. CONDITIONING MECHANISM

```
FiLM (Feature-wise Linear Modulation)
══════════════════════════════════════════════════════

Purpose: Inject scene-specific and time-specific information
         into the denoising process

Time Conditioning:
──────────────────
t (scalar) ──> Sinusoidal ──> Linear ──> (scale_t, shift_t)
                Encoding       Layer        (128 dim)

Scene Conditioning:
───────────────────
Scene     ──> Conv ──> AvgPool ──> Linear ──> (scale_c, shift_c)
(3,H,W)       +ReLU      to 1x1     Layer        (64 dim)


Applied to Features:
────────────────────
Features h (B, C, H, W)

Step 1: Time modulation
    h = h * (1 + scale_t) + shift_t
    where scale_t, shift_t are broadcast to (B, C, 1, 1)

Step 2: Condition modulation
    h = h * (1 + scale_c) + shift_c
    where scale_c, shift_c are broadcast to (B, C, 1, 1)

Result: Features are scaled and shifted based on:
    - What timestep we're at (t)
    - What scene geometry we have (elevation, TX position)


Why This Works:
───────────────
- Scale: Amplifies/dampens features (e.g., more attenuation behind tall buildings)
- Shift: Adds bias (e.g., base signal strength offset)
- Adaptive: Each layer learns different scene-dependent transformations
- Efficient: Only adds 2*(128+64) = 384 parameters per ResBlock
```

---

## 7. TENSOR SHAPES REFERENCE

```
Throughout the Network
══════════════════════════════════════════════════════

TRAINING
────────
Batch size (B) = 4
Height (H) = 200 (typical)
Width (W) = 200 (typical)

Input tensor:      (4, 3, 200, 200)
    - elevation:   (4, 1, 200, 200)
    - distance:    (4, 1, 200, 200)
    - frequency:   (4, 1, 200, 200)

Target tensor:     (4, 1, 200, 200)
    - RSS values

Noisy target:      (4, 1, 200, 200)
    - target + noise at timestep t

Timesteps:         (4,)
    - t values for each sample

Model output:      (4, 1, 200, 200)
    - predicted noise


THROUGH THE U-NET
─────────────────
After input conv:        (4, 32, 200, 200)
After encode level 1:    (4, 32, 100, 100)
After encode level 2:    (4, 64, 50, 50)
After encode level 3:    (4, 128, 25, 25)

Bottleneck:              (4, 128, 25, 25)

After decode level 3:    (4, 64, 50, 50)
After decode level 2:    (4, 32, 100, 100)
After decode level 1:    (4, 32, 200, 200)

After output conv:       (4, 1, 200, 200)


EMBEDDINGS
──────────
Time embedding:          (4, 128)
Condition embedding:     (4, 64)
FiLM scale/shift:        (4, C) → broadcast to (4, C, 1, 1)
```

---

## 8. LOSS COMPUTATION DIAGRAM

```
Loss Calculation
══════════════════════════════════════════════════════

Ground Truth      Random Noise      Timestep
RSS (x_0)         (ε ~ N(0,1))      (t)
    │                  │              │
    └─────┬────────────┘              │
          │                           │
          ▼                           │
    ┌───────────┐                     │
    │ Forward   │                     │
    │ Diffusion │                     │
    └───────────┘                     │
          │                           │
          ▼                           │
    Noisy RSS (x_t)                   │
          │                           │
          ├───────────────────────────┘
          │
          ▼
    ┌──────────────────┐
    │  TimeCondUNet    │
    │  (+ conditions)  │
    └──────────────────┘
          │
          ▼
    Predicted Noise (ε_pred)
          │
          │    Actual Noise (ε)
          │         │
          └────┬────┘
               │
               ▼
          ┌─────────┐
          │   MSE   │
          └─────────┘
               │
               ▼
    Loss = mean((ε_pred - ε)²)
               │
               ▼
    ┌──────────────────┐
    │  Backpropagation │
    │  Update Weights  │
    └──────────────────┘


Mathematical Detail:
────────────────────
x_t = √(ᾱ_t)·x_0 + √(1-ᾱ_t)·ε

Model learns: ε_θ(x_t, t, c) ≈ ε

Loss: L = E[||ε - ε_θ(x_t, t, c)||²]

Where:
- x_0: clean RSS map
- x_t: noisy version at timestep t
- ε: actual noise added
- ε_θ: model's noise prediction
- c: conditioning (scene features)
- ᾱ_t: cumulative noise schedule
```

---

## SUMMARY

This document provides visual representations of:
1. Overall system architecture (data generation → training → inference)
2. Detailed model architecture (TimeCondUNet structure)
3. Residual block internals (with FiLM conditioning)
4. Diffusion process (forward and reverse)
5. Data flow through the system
6. Conditioning mechanism explanation
7. Tensor shape reference
8. Loss computation process

Use these diagrams to:
- Explain the model to stakeholders
- Debug issues (check tensor shapes)
- Understand information flow
- Design improvements

**For more details, see:**
- `PROFESSOR_OVERVIEW.md` - Complete written explanation
- `TRAINING_PROCESS_SUMMARY.md` - Quick reference guide
- `models/diffusion.py` - Actual implementation

**Last Updated**: January 30, 2026
