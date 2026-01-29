#!/usr/bin/env python
# ========================================================================
# Batch Visualization Utility
# Generate multiple sets of visualizations with different configurations
# ========================================================================
"""
Create multiple visualization sets for comparison and analysis.

Usage:
    python batch_visualize.py [--configs PRESET]

Presets:
    quick       - 1 scene, 20 steps (30 sec)
    standard    - 3 scenes, 50 steps (2 min)
    complete    - all scenes, 50 steps (5 min)
    hires       - 2 scenes, 100 steps (4 min)
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

# Visualization configurations
CONFIGS = {
    'quick': {
        'num_scenes': 1,
        'diffusion_steps': 20,
        'output_dir': 'rss_visualizations_quick',
        'description': 'Quick test (1 scene, 20 steps)'
    },
    'standard': {
        'num_scenes': 3,
        'diffusion_steps': 50,
        'output_dir': 'rss_visualizations_standard',
        'description': 'Standard (3 scenes, 50 steps)'
    },
    'complete': {
        'num_scenes': 5,
        'diffusion_steps': 50,
        'output_dir': 'rss_visualizations_complete',
        'description': 'Complete (all scenes, 50 steps)'
    },
    'hires': {
        'num_scenes': 2,
        'diffusion_steps': 100,
        'output_dir': 'rss_visualizations_hires',
        'description': 'High-resolution (2 scenes, 100 steps)'
    },
    'sampling_study': {
        'num_scenes': 1,
        'diffusion_steps': 10,
        'output_dir': 'rss_visualizations_steps10',
        'description': 'Sampling study - 10 steps'
    },
}


def run_visualization(config):
    """Run visualization with given config."""
    print(f"\n{'='*80}")
    print(f"Configuration: {config['description']}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,
        'visualize_rss_maps.py',
        '--num-scenes', str(config['num_scenes']),
        '--diffusion-steps', str(config['diffusion_steps']),
        '--output-dir', config['output_dir']
    ]
    
    result = subprocess.run(cmd, capture_output=False)
    
    if result.returncode == 0:
        print(f"✓ Completed: {config['output_dir']}")
        return True
    else:
        print(f"✗ Failed: {config['output_dir']}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Batch visualization with multiple configurations'
    )
    parser.add_argument(
        '--config',
        choices=list(CONFIGS.keys()),
        default='standard',
        help=f'Preset configuration (default: standard)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all configurations'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available configurations'
    )
    
    args = parser.parse_args()
    
    # List configurations
    if args.list:
        print("\nAvailable configurations:")
        print("-" * 80)
        for key, config in CONFIGS.items():
            print(f"{key:<20} - {config['description']:<40}")
        print("-" * 80)
        return
    
    # Run specific or all configurations
    if args.all:
        configs_to_run = list(CONFIGS.values())
        print(f"\nRunning {len(configs_to_run)} visualization configurations...\n")
    else:
        if args.config not in CONFIGS:
            print(f"ERROR: Unknown configuration '{args.config}'")
            print(f"Available: {', '.join(CONFIGS.keys())}")
            sys.exit(1)
        configs_to_run = [CONFIGS[args.config]]
    
    # Run visualizations
    results = []
    start_time = datetime.now()
    
    for config in configs_to_run:
        success = run_visualization(config)
        results.append({
            'config': config['description'],
            'output_dir': config['output_dir'],
            'success': success
        })
    
    # Print summary
    elapsed = datetime.now() - start_time
    
    print(f"\n{'='*80}")
    print("BATCH VISUALIZATION SUMMARY")
    print(f"{'='*80}\n")
    
    print(f"{'Configuration':<40} {'Status':<15} {'Output Directory':<30}")
    print("-" * 80)
    for result in results:
        status = "✓ Success" if result['success'] else "✗ Failed"
        print(f"{result['config']:<40} {status:<15} {result['output_dir']:<30}")
    
    print(f"\nTotal time: {elapsed.total_seconds():.1f} seconds")
    print(f"\n✓ Visualizations saved to individual directories")
    print(f"✓ View PNGs to compare model predictions with ground truth")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
