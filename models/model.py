
# =========================================================================
# Training Entry Point for Conditional Diffusion Model
# Includes checkpoint management, logging, and backtesting integration
# =========================================================================
import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from diffusion import TimeCondUNet, RadioMapDataset, train, Diffusion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manage model checkpoints with metadata."""
    
    def __init__(self, checkpoint_dir: str = './checkpoints'):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.metadata_file = self.checkpoint_dir / 'training_metadata.json'
    
    def save_checkpoint(self,
                       model: nn.Module,
                       optimizer,
                       epoch: int,
                       loss: float,
                       config: dict):
        """Save model checkpoint with metadata."""
        timestamp = datetime.now().isoformat()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'timestamp': timestamp,
            'config': config
        }
        
        filename = f'model_epoch{epoch:03d}.pt'
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['latest_checkpoint'] = filename
        metadata['latest_epoch'] = epoch
        metadata['latest_loss'] = loss
        metadata['last_updated'] = timestamp
        self._save_metadata(metadata)
        
        logger.info(f"✓ Checkpoint saved: {filename} (loss: {loss:.6f})")
        return path
    
    def load_checkpoint(self, filename: str = None):
        """Load checkpoint. If filename is None, load latest."""
        if filename is None:
            metadata = self._load_metadata()
            filename = metadata.get('latest_checkpoint')
            if not filename:
                logger.warning("No checkpoint found")
                return None
        
        path = self.checkpoint_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        checkpoint = torch.load(path, map_location='cpu')
        logger.info(f"✓ Loaded checkpoint: {filename}")
        return checkpoint
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self, metadata):
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Train conditional diffusion model for RSS prediction'
    )
    parser.add_argument('--input-dir', default='../model_input/data/training/input',
                       help='Path to input tensors directory')
    parser.add_argument('--target-dir', default='../model_input/data/training/target',
                       help='Path to target tensors directory')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--timesteps', type=int, default=1000,
                       help='Number of diffusion timesteps')
    parser.add_argument('--save-every', type=int, default=5,
                       help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint-dir', default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='Number of data loading workers')
    
    args = parser.parse_args()
    
    try:
        # Validate directories
        if not os.path.isdir(args.target_dir):
            raise FileNotFoundError(f"Target directory not found: {args.target_dir}")
        if not os.path.isdir(args.input_dir):
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        
        # Device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {device}")
        if device == 'cuda':
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        
        # Load dataset
        logger.info("Loading dataset...")
        dataset = RadioMapDataset(args.input_dir, args.target_dir)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty. Check data directories.")
        logger.info(f"✓ Loaded {len(dataset)} samples")
        
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
        
        logger.info("Initializing model...")
        model = TimeCondUNet(**model_config)
        model.to(device)
        
        num_params = sum(p.numel() for p in model.parameters()) / 1e6
        logger.info(f"Model parameters: {num_params:.2f}M")
        logger.info(f"Model config: {model_config}")
        
        # Initialize checkpoint manager
        checkpoint_mgr = CheckpointManager(args.checkpoint_dir)
        
        # Resume or start fresh
        start_epoch = 0
        if args.resume:
            try:
                checkpoint = checkpoint_mgr.load_checkpoint()
                if checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = checkpoint['epoch'] + 1
                    logger.info(f"Resuming from epoch {start_epoch}")
            except Exception as e:
                logger.warning(f"Could not resume: {e}. Starting from scratch.")
        
        # Training configuration
        logger.info(f"Training config:")
        logger.info(f"  Epochs: {args.epochs}")
        logger.info(f"  Batch size: {args.batch_size}")
        logger.info(f"  Learning rate: {args.lr}")
        logger.info(f"  Timesteps: {args.timesteps}")
        
        # Train
        logger.info("Starting training...")
        train(
            model,
            dataset,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            timesteps=args.timesteps,
            save_every=args.save_every,
            out_dir=args.checkpoint_dir
        )
        
        logger.info("✓ Training completed successfully!")
        
        # Print summary
        print("\n" + "="*80)
        print("TRAINING SUMMARY")
        print("="*80)
        print(f"Model: TimeCondUNet")
        print(f"Samples: {len(dataset)}")
        print(f"Parameters: {num_params:.2f}M")
        print(f"Device: {device}")
        print(f"Checkpoints saved to: {args.checkpoint_dir}")
        print("="*80 + "\n")
        
    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Value error: {e}")
        sys.exit(1)
    except RuntimeError as e:
        logger.error(f"Runtime error (possibly out of memory): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if device == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
    
