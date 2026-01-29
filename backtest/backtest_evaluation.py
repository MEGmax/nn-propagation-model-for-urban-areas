# ========================================================================
# Backtesting & Evaluation Framework for RSS Prediction Model
# Based on AIRMap paper methodology (Saeizadeh et al., arXiv:2511.05522)
# ========================================================================
"""
This script evaluates the conditional diffusion model's performance on RSS
prediction for outdoor urban geometry scenarios.

Key metrics based on AIRMap paper:
- Root Mean Squared Error (RMSE) in dB
- Mean Absolute Error (MAE) in dB  
- Median Error
- Error CDF and percentiles
- Coverage probability (spatial accuracy)
- Speedup vs ray-tracing baseline

Normalization reference (from model_input.py and copilot-instructions.md):
- RSS_DB_FLOOR = -100.0 dBm
- RSS_DB_SCALE = 50.0
- Normalized RSS range: [-1, 1]
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import model components
sys.path.insert(0, str(Path(__file__).parent / 'models'))
from diffusion import TimeCondUNet, RadioMapDataset, Diffusion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants (matching model_input.py normalization)
RSS_DB_FLOOR = -100.0
RSS_DB_SCALE = 50.0
RSS_MIN_NORMALIZED = (RSS_DB_FLOOR - RSS_DB_FLOOR) / RSS_DB_SCALE  # -1.0
RSS_MAX_NORMALIZED = (RSS_DB_FLOOR + RSS_DB_SCALE - RSS_DB_FLOOR) / RSS_DB_SCALE  # 1.0


class RSSNormalizer:
    """Handle RSS normalization/denormalization for dBm <-> normalized range."""
    
    @staticmethod
    def denormalize(rss_normalized: np.ndarray) -> np.ndarray:
        """Convert from [-1, 1] normalized to dBm."""
        return rss_normalized * RSS_DB_SCALE + RSS_DB_FLOOR
    
    @staticmethod
    def normalize(rss_dbm: np.ndarray) -> np.ndarray:
        """Convert from dBm to [-1, 1] normalized."""
        return (rss_dbm - RSS_DB_FLOOR) / RSS_DB_SCALE


class RadioMapEvaluator:
    """Comprehensive evaluation metrics for RSS prediction."""
    
    def __init__(self, normalizer: RSSNormalizer = None):
        self.normalizer = normalizer or RSSNormalizer()
        self.results = {}
    
    def compute_metrics(self, 
                       predicted: np.ndarray,
                       ground_truth: np.ndarray,
                       scenario_name: str = "unknown") -> Dict:
        """
        Compute all evaluation metrics for a single scene.
        
        Args:
            predicted: (H, W) or (1, H, W) predicted RSS in dBm or normalized
            ground_truth: (H, W) or (1, H, W) ground truth RSS in dBm
            scenario_name: Name of the scenario for logging
            
        Returns:
            Dictionary with all computed metrics
        """
        # Flatten and ensure same shape
        pred = predicted.flatten() if predicted.ndim > 1 else predicted
        gt = ground_truth.flatten() if ground_truth.ndim > 1 else ground_truth
        
        # Handle normalized predictions (if in [-1, 1] range)
        if pred.min() >= -1.5 and pred.max() <= 1.5:
            logger.info(f"  Scenario '{scenario_name}': Prediction detected as normalized, denormalizing")
            pred = self.normalizer.denormalize(pred)
        
        if gt.min() >= -1.5 and gt.max() <= 1.5:
            logger.info(f"  Scenario '{scenario_name}': Ground truth detected as normalized, denormalizing")
            gt = self.normalizer.denormalize(gt)
        
        # Compute error metrics
        error = pred - gt  # prediction error
        abs_error = np.abs(error)
        
        metrics = {
            'scenario': scenario_name,
            'num_pixels': len(gt),
            'rmse_db': float(np.sqrt(np.mean(error**2))),
            'mae_db': float(np.mean(abs_error)),
            'std_error_db': float(np.std(error)),
            'median_error_db': float(np.median(abs_error)),
            'max_error_db': float(np.max(abs_error)),
            'min_error_db': float(np.min(abs_error)),
            'percentile_90_db': float(np.percentile(abs_error, 90)),
            'percentile_75_db': float(np.percentile(abs_error, 75)),
            'percentile_50_db': float(np.percentile(abs_error, 50)),
            'percentile_25_db': float(np.percentile(abs_error, 25)),
            'bias_db': float(np.mean(error)),  # systematic bias
            'pred_mean_db': float(np.mean(pred)),
            'gt_mean_db': float(np.mean(gt)),
            'pred_std_db': float(np.std(pred)),
            'gt_std_db': float(np.std(gt)),
        }
        
        # Within-error-threshold coverage
        for threshold in [3.0, 5.0, 10.0]:
            within = (abs_error <= threshold).sum() / len(gt)
            metrics[f'within_{threshold}_db'] = float(within)
        
        # Correlation metrics
        if len(pred) > 1:
            correlation = np.corrcoef(pred, gt)[0, 1]
            metrics['pearson_correlation'] = float(correlation)
        
        return metrics
    
    def summarize_batch(self, batch_metrics: List[Dict]) -> Dict:
        """Aggregate metrics across multiple scenarios."""
        if not batch_metrics:
            return {}
        
        summary = {}
        keys_to_aggregate = [
            'rmse_db', 'mae_db', 'median_error_db', 'bias_db',
            'within_3_db', 'within_5_db', 'within_10_db', 'pearson_correlation'
        ]
        
        for key in keys_to_aggregate:
            values = [m[key] for m in batch_metrics if key in m]
            if values:
                summary[f'{key}_mean'] = float(np.mean(values))
                summary[f'{key}_std'] = float(np.std(values))
                summary[f'{key}_min'] = float(np.min(values))
                summary[f'{key}_max'] = float(np.max(values))
        
        summary['num_scenarios'] = len(batch_metrics)
        return summary


class DiffusionSampler:
    """Wrapper for sampling from trained diffusion model."""
    
    def __init__(self, model: TimeCondUNet, device: str = 'cuda'):
        self.model = model
        self.device = device
        self.diffusion = None
    
    def setup_diffusion(self, timesteps: int = 1000):
        """Initialize diffusion process."""
        self.diffusion = Diffusion(self.model, timesteps=timesteps, device=self.device)
    
    @torch.no_grad()
    def sample(self,
               cond_img: torch.Tensor,
               num_samples: int = 1,
               steps: int = 50) -> np.ndarray:
        """
        Generate samples from model.
        
        Args:
            cond_img: (B, C, H, W) conditioning image
            num_samples: Number of samples to generate per input
            steps: Number of reverse diffusion steps
            
        Returns:
            (B*num_samples, 1, H, W) sampled RSS maps
        """
        if self.diffusion is None:
            self.setup_diffusion()
        
        self.model.eval()
        all_samples = []
        
        for _ in range(num_samples):
            samples = self.diffusion.sample(cond_img, steps=steps)
            all_samples.append(samples.detach().cpu().numpy())
        
        return np.concatenate(all_samples, axis=0)


def load_trained_model(checkpoint_path: str,
                       model_config: Dict,
                       device: str = 'cuda') -> TimeCondUNet:
    """Load a trained model checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")
    model = TimeCondUNet(**model_config)
    
    if os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state)
        logger.info(f"✓ Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint not found at {checkpoint_path}")
    
    model.to(device)
    return model


def backtest_on_dataset(
    model: TimeCondUNet,
    dataset: RadioMapDataset,
    device: str = 'cuda',
    num_samples_per_scene: int = 5,
    diffusion_steps: int = 50,
    batch_size: int = 2,
    output_dir: str = './backtest_results'
) -> Tuple[List[Dict], Dict]:
    """
    Run full backtesting on dataset.
    
    Args:
        model: Trained TimeCondUNet model
        dataset: RadioMapDataset with input/target pairs
        device: 'cuda' or 'cpu'
        num_samples_per_scene: Number of diffusion samples per input
        diffusion_steps: Number of reverse diffusion steps
        batch_size: Batch size for inference
        output_dir: Directory to save results
        
    Returns:
        (per_scene_metrics, aggregate_metrics)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    evaluator = RadioMapEvaluator()
    sampler = DiffusionSampler(model, device=device)
    sampler.setup_diffusion()
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_metrics = []
    
    logger.info(f"Starting backtesting on {len(dataset)} scenes...")
    
    for batch_idx, batch in enumerate(tqdm(loader, desc="Evaluating")):
        inputs = batch['input'].to(device)  # (B, C, H, W)
        targets = batch['target'].to(device)  # (B, 1, H, W)
        
        # Generate samples
        with torch.no_grad():
            predictions = sampler.sample(
                inputs,
                num_samples=num_samples_per_scene,
                steps=diffusion_steps
            )  # (B*num_samples, 1, H, W)
        
        # Evaluate each original scene
        for scene_idx in range(inputs.shape[0]):
            scene_target = targets[scene_idx, 0].cpu().numpy()  # (H, W)
            
            # Average across diffusion samples for this scene
            sample_preds = predictions[
                scene_idx*num_samples_per_scene:(scene_idx+1)*num_samples_per_scene,
                0
            ]  # (num_samples, H, W)
            
            # Use ensemble mean
            ensemble_pred = np.mean(sample_preds, axis=0)
            
            scenario_name = f"batch{batch_idx}_scene{scene_idx}"
            metrics = evaluator.compute_metrics(ensemble_pred, scene_target, scenario_name)
            all_metrics.append(metrics)
            
            logger.debug(f"  {scenario_name}: RMSE={metrics['rmse_db']:.2f}dB, "
                        f"MAE={metrics['mae_db']:.2f}dB")
    
    # Aggregate results
    aggregate = evaluator.summarize_batch(all_metrics)
    
    # Save results
    results_file = os.path.join(output_dir, 'backtest_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'per_scene': all_metrics,
            'aggregate': aggregate,
            'config': {
                'num_scenes': len(all_metrics),
                'num_samples_per_scene': num_samples_per_scene,
                'diffusion_steps': diffusion_steps,
            }
        }, f, indent=2)
    
    logger.info(f"Results saved to {results_file}")
    return all_metrics, aggregate


def plot_evaluation_results(metrics: List[Dict],
                            aggregate: Dict,
                            output_dir: str = './backtest_results'):
    """Generate diagnostic plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract key metrics
    rmses = [m['rmse_db'] for m in metrics]
    maes = [m['mae_db'] for m in metrics]
    medians = [m['median_error_db'] for m in metrics]
    biases = [m['bias_db'] for m in metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # RMSE distribution
    axes[0, 0].hist(rmses, bins=20, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(rmses), color='r', linestyle='--', label=f'Mean: {np.mean(rmses):.2f}dB')
    axes[0, 0].set_xlabel('RMSE (dB)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('RMSE Distribution Across Scenes')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # MAE distribution
    axes[0, 1].hist(maes, bins=20, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(maes), color='r', linestyle='--', label=f'Mean: {np.mean(maes):.2f}dB')
    axes[0, 1].set_xlabel('MAE (dB)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('MAE Distribution Across Scenes')
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    # Error CDF
    sorted_maes = np.sort(maes)
    cdf = np.arange(1, len(sorted_maes) + 1) / len(sorted_maes)
    axes[1, 0].plot(sorted_maes, cdf, linewidth=2, marker='o', markersize=4)
    axes[1, 0].axvline(5.0, color='g', linestyle='--', alpha=0.5, label='5 dB threshold')
    axes[1, 0].axvline(10.0, color='r', linestyle='--', alpha=0.5, label='10 dB threshold')
    axes[1, 0].set_xlabel('Absolute Error (dB)')
    axes[1, 0].set_ylabel('CDF')
    axes[1, 0].set_title('Error Cumulative Distribution Function')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Bias distribution
    axes[1, 1].hist(biases, bins=20, alpha=0.7, color='orange', edgecolor='black')
    axes[1, 1].axvline(0, color='k', linestyle='-', alpha=0.5)
    axes[1, 1].axvline(np.mean(biases), color='r', linestyle='--', label=f'Mean: {np.mean(biases):.2f}dB')
    axes[1, 1].set_xlabel('Bias (dB)')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Prediction Bias Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'backtest_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plots to {plot_path}")
    plt.close()


def print_evaluation_report(metrics: List[Dict], aggregate: Dict):
    """Print formatted evaluation report."""
    print("\n" + "="*80)
    print("BACKTESTING REPORT: Conditional Diffusion Model for RSS Prediction")
    print("Based on AIRMap paper evaluation methodology")
    print("="*80)
    
    print(f"\nEvaluated: {aggregate.get('num_scenarios', 0)} scenes")
    
    print("\n--- RMSE (dB) ---")
    print(f"  Mean:   {aggregate.get('rmse_db_mean', np.nan):.3f} dB")
    print(f"  Std:    {aggregate.get('rmse_db_std', np.nan):.3f} dB")
    print(f"  Min:    {aggregate.get('rmse_db_min', np.nan):.3f} dB")
    print(f"  Max:    {aggregate.get('rmse_db_max', np.nan):.3f} dB")
    
    print("\n--- MAE (dB) ---")
    print(f"  Mean:   {aggregate.get('mae_db_mean', np.nan):.3f} dB")
    print(f"  Std:    {aggregate.get('mae_db_std', np.nan):.3f} dB")
    print(f"  Min:    {aggregate.get('mae_db_min', np.nan):.3f} dB")
    print(f"  Max:    {aggregate.get('mae_db_max', np.nan):.3f} dB")
    
    print("\n--- Median Absolute Error (dB) ---")
    print(f"  Mean:   {aggregate.get('median_error_db_mean', np.nan):.3f} dB")
    
    print("\n--- Coverage (% within threshold) ---")
    print(f"  Within 3 dB:  {aggregate.get('within_3_db_mean', 0)*100:.1f}%")
    print(f"  Within 5 dB:  {aggregate.get('within_5_db_mean', 0)*100:.1f}%")
    print(f"  Within 10 dB: {aggregate.get('within_10_db_mean', 0)*100:.1f}%")
    
    print("\n--- Bias ---")
    print(f"  Mean:   {aggregate.get('bias_db_mean', np.nan):.3f} dB")
    print(f"  Std:    {aggregate.get('bias_db_std', np.nan):.3f} dB")
    
    print("\n--- Correlation ---")
    print(f"  Pearson R:  {aggregate.get('pearson_correlation_mean', np.nan):.3f}")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("-" * 80)
    
    rmse = aggregate.get('rmse_db_mean', np.nan)
    if rmse < 5:
        print("✓ EXCELLENT: Model achieves < 5 dB RMSE (on par with AIRMap paper)")
    elif rmse < 10:
        print("✓ GOOD: Model achieves < 10 dB RMSE")
    elif rmse < 20:
        print("⚠ FAIR: Model achieves < 20 dB RMSE (needs improvement)")
    else:
        print("✗ POOR: RMSE > 20 dB (significant improvement needed)")
    
    coverage_5db = aggregate.get('within_5_db_mean', 0) * 100
    if coverage_5db > 70:
        print(f"✓ Coverage within 5 dB: {coverage_5db:.1f}% (strong spatial accuracy)")
    elif coverage_5db > 50:
        print(f"⚠ Coverage within 5 dB: {coverage_5db:.1f}% (moderate spatial accuracy)")
    else:
        print(f"✗ Coverage within 5 dB: {coverage_5db:.1f}% (poor spatial accuracy)")
    
    bias = aggregate.get('bias_db_mean', np.nan)
    if abs(bias) < 1:
        print(f"✓ Bias: {bias:.2f} dB (well-calibrated, no systematic error)")
    else:
        print(f"⚠ Bias: {bias:.2f} dB (systematic error detected)")
    
    print("="*80 + "\n")


if __name__ == '__main__':
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Dataset paths
    INPUT_DIR = 'model_input/data/training/input'
    TARGET_DIR = 'model_input/data/training/target'
    CHECKPOINT = 'models/checkpoints/model_final.pt'
    OUTPUT_DIR = 'backtest_results'
    
    # Model configuration
    model_config = {
        'in_ch': 1,
        'cond_channels': 3,  # elevation, distance, frequency
        'base_ch': 32,
        'channel_mults': (1, 2, 4),
        'num_res_blocks': 2,
        'time_emb_dim': 128,
        'cond_emb_dim': 64
    }
    
    try:
        # Load dataset
        logger.info("Loading dataset...")
        dataset = RadioMapDataset(INPUT_DIR, TARGET_DIR)
        logger.info(f"✓ Loaded {len(dataset)} samples")
        
        # Load model
        model = load_trained_model(CHECKPOINT, model_config, device=device)
        
        # Run backtesting
        metrics, aggregate = backtest_on_dataset(
            model,
            dataset,
            device=device,
            num_samples_per_scene=5,
            diffusion_steps=50,
            batch_size=2,
            output_dir=OUTPUT_DIR
        )
        
        # Generate visualizations
        plot_evaluation_results(metrics, aggregate, OUTPUT_DIR)
        
        # Print report
        print_evaluation_report(metrics, aggregate)
        
    except Exception as e:
        logger.error(f"Error during backtesting: {e}", exc_info=True)
        sys.exit(1)
