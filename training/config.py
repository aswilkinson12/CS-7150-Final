"""
TrajectoryGPT - Training Configuration
Hyperparameters and settings for training
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    
    # Model architecture
    vocab_size: int = 280
    n_embd: int = 384
    n_layer: int = 6
    n_head: int = 6
    max_seq_len: int = 64
    dropout: float = 0.1
    
    # Training
    batch_size: int = 128
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.999)
    grad_clip: float = 1.0
    
    # Learning rate schedule
    warmup_iters: int = 100
    lr_decay_iters: int = 5000
    min_lr: float = 1e-5
    
    # Data
    data_path: str = './data/trajectories_mini.pkl'
    val_path: Optional[str] = None
    val_split: float = 0.1
    num_workers: int = 4
    
    # Logging & checkpointing
    log_interval: int = 10
    eval_interval: int = 100
    checkpoint_interval: int = 500
    checkpoint_dir: str = './checkpoints'
    
    # Weights & Biases
    use_wandb: bool = False
    wandb_project: str = 'TrajectoryGPT'
    wandb_run_name: Optional[str] = None
    
    # Device
    device: str = 'cuda' if None else 'cpu'  # Will be set automatically
    compile_model: bool = False  # PyTorch 2.0 compile (if available)
    
    # Generation (for eval)
    num_samples: int = 10  # Number of diverse predictions per input
    temperature: float = 1.0
    top_p: float = 0.9
    
    def __post_init__(self):
        """Set device automatically"""
        import torch
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            self.device = 'mps'
        else:
            self.device = 'cpu'


def get_lr(iteration: int, config: TrainingConfig) -> float:
    """
    Learning rate schedule with warmup and cosine decay
    
    Args:
        iteration: Current training iteration
        config: Training configuration
        
    Returns:
        Learning rate for this iteration
    """
    # 1) Linear warmup
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    
    # 2) Constant after decay
    if iteration > config.lr_decay_iters:
        return config.min_lr
    
    # 3) Cosine decay
    decay_ratio = (iteration - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


# Preset configurations
CONFIGS = {
    'mini': TrainingConfig(
        # For quick testing on mini dataset
        batch_size=64,
        num_epochs=20,
        data_path='./data/trajectories_mini.pkl'
    ),
    
    'full': TrainingConfig(
        # For full dataset training
        batch_size=128,
        num_epochs=100,
        data_path='./data/trajectories_full.pkl',
        lr_decay_iters=50000
    ),
    
    'small': TrainingConfig(
        # Smaller model for debugging
        n_embd=256,
        n_layer=4,
        n_head=4,
        batch_size=64
    ),
    
    'large': TrainingConfig(
        # Larger model for better performance
        n_embd=512,
        n_layer=8,
        n_head=8,
        batch_size=64,  # Smaller batch for larger model
        num_epochs=150
    )
}


import math


def print_config(config: TrainingConfig):
    """Pretty print configuration"""
    print("="*60)
    print("TrajectoryGPT Training Configuration")
    print("="*60)
    
    print("\nModel Architecture:")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Embedding dimension: {config.n_embd}")
    print(f"  Number of layers: {config.n_layer}")
    print(f"  Number of heads: {config.n_head}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Dropout: {config.dropout}")
    
    print("\nTraining:")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Number of epochs: {config.num_epochs}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Weight decay: {config.weight_decay}")
    print(f"  Gradient clipping: {config.grad_clip}")
    print(f"  Warmup iterations: {config.warmup_iters}")
    
    print("\nData:")
    print(f"  Data path: {config.data_path}")
    print(f"  Validation split: {config.val_split}")
    print(f"  Num workers: {config.num_workers}")
    
    print("\nDevice:")
    print(f"  Device: {config.device}")
    print(f"  Compile model: {config.compile_model}")
    
    print("\nLogging:")
    print(f"  Use wandb: {config.use_wandb}")
    print(f"  Log interval: {config.log_interval}")
    print(f"  Eval interval: {config.eval_interval}")
    print(f"  Checkpoint interval: {config.checkpoint_interval}")
    
    print("="*60)


if __name__ == "__main__":
    # Test configurations
    config = TrainingConfig()
    print_config(config)
    
    print("\nAvailable preset configs:")
    for name in CONFIGS.keys():
        print(f"  - {name}")
