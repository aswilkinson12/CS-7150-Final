"""
TrajectoryGPT - Training Loop
Main training script
"""

import os
import sys
import time
import math
import torch
from tqdm import tqdm
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.gpt_decoder import TrajectoryGPT, TrajectoryGPTConfig
from data_processing.dataset import create_dataloaders
from training.config import TrainingConfig, get_lr, print_config


class Trainer:
    """Trainer class for TrajectoryGPT"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Create model
        model_config = TrajectoryGPTConfig(
            vocab_size=config.vocab_size,
            n_embd=config.n_embd,
            n_layer=config.n_layer,
            n_head=config.n_head,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout
        )

        print("Creating model...")
        self.model = TrajectoryGPT(model_config).to(self.device)

        # Compile if requested (PyTorch 2.0+)
        if config.compile_model:
            print("Compiling model...")
            self.model = torch.compile(self.model)

        # Create optimizer
        self.optimizer = self.model.configure_optimizers(
            weight_decay=config.weight_decay,
            learning_rate=config.learning_rate,
            betas=config.betas,
            device_type=config.device
        )

        # Create dataloaders
        print("Creating dataloaders...")
        self.train_loader, self.val_loader = create_dataloaders(
            train_path=config.data_path,
            val_path=config.val_path,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            val_split=config.val_split
        )

        # Setup logging
        self.setup_logging()

        # Training state
        self.iteration = 0
        self.best_val_loss = float('inf')

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)

    def setup_logging(self):
        """Setup Weights & Biases logging if requested"""
        if self.config.use_wandb:
            try:
                import wandb
                wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=vars(self.config)
                )
                self.wandb = wandb
                print("✓ Weights & Biases logging enabled")
            except ImportError:
                print("Warning: wandb not installed, skipping W&B logging")
                self.config.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            # Update learning rate
            lr = get_lr(self.iteration, self.config)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            # Forward pass
            logits, loss = self.model(inputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip
                )

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            self.iteration += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{lr:.2e}'
            })

            # Logging
            if self.iteration % self.config.log_interval == 0:
                if self.wandb:
                    self.wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': lr,
                        'train/iteration': self.iteration
                    })

            # Validation
            if self.iteration % self.config.eval_interval == 0:
                val_loss = self.validate()
                self.model.train()  # Back to training mode

                if self.wandb:
                    self.wandb.log({
                        'val/loss': val_loss,
                        'val/iteration': self.iteration
                    })

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint('best_model.pt')
                    print(f"✓ New best validation loss: {val_loss:.4f}")

            # Checkpoint
            if self.iteration % self.config.checkpoint_interval == 0:
                self.save_checkpoint(f'checkpoint_iter_{self.iteration}.pt')

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    @torch.no_grad()
    def validate(self):
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0

        for batch in self.val_loader:
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)

            logits, loss = self.model(inputs, targets)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)

        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'iteration': self.iteration,
            'best_val_loss': self.best_val_loss
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Saved checkpoint: {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"✓ Loaded checkpoint from iteration {self.iteration}")

    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Epoch summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Time: {epoch_time:.1f}s")

            if self.wandb:
                self.wandb.log({
                    'epoch/train_loss': train_loss,
                    'epoch/val_loss': val_loss,
                    'epoch/time': epoch_time,
                    'epoch/number': epoch
                })

        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Final checkpoint: {self.config.checkpoint_dir}/best_model.pt")

        if self.wandb:
            self.wandb.finish()


def main():
    """Main training function"""
    import argparse

    parser = argparse.ArgumentParser(description='Train TrajectoryGPT')
    parser.add_argument('--data', type=str, default='./data/trajectories_mini.pkl',
                       help='Path to training data')
    parser.add_argument('--config', type=str, default='mini',
                       choices=['mini', 'full', 'small', 'large'],
                       help='Preset configuration')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (overrides config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--wandb', action='store_true',
                       help='Enable Weights & Biases logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--vocab-size', type=int, default=None,
                        help='Vocabulary size')


    args = parser.parse_args()

    # Load config
    from training.config import CONFIGS
    config = CONFIGS.get(args.config, TrainingConfig())

    # Override with args
    config.data_path = args.data
    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.wandb:
        config.use_wandb = True
    if args.vocab_size is not None:
        config.vocab_size = args.vocab_size

    # Print configuration
    print_config(config)

    # Create trainer
    trainer = Trainer(config)

    # Resume if requested
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
