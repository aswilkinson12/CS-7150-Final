"""
LSTM Baseline for Trajectory Prediction
Simple encoder-decoder LSTM for comparison with TrajectoryGPT

Architecture:
- Encoder LSTM: processes history (4 timesteps)
- Decoder LSTM: generates future (6 timesteps)
- Single deterministic prediction (no diversity)

This shows the advantage of GPT's multi-modal predictions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional

from tqdm import tqdm


class LSTMBaseline(nn.Module):
    """
    Simple LSTM encoder-decoder for trajectory prediction
    Deterministic baseline - produces single prediction
    """

    def __init__(
        self,
        input_dim: int = 2,      # (x, y) coordinates
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: processes history
        self.encoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Decoder: generates future
        self.decoder = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Output layer: predicts next position
        self.output_layer = nn.Linear(hidden_dim, input_dim)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(
        self,
        history: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        teacher_forcing_ratio: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            history: (batch, history_len, 2) past trajectory
            future: (batch, future_len, 2) future trajectory (for training)
            teacher_forcing_ratio: probability of using ground truth

        Returns:
            predictions: (batch, future_len, 2) predicted trajectory
            loss: scalar loss (if future provided)
        """
        batch_size = history.size(0)
        future_len = 6 if future is None else future.size(1)

        # Encode history
        _, (hidden, cell) = self.encoder(history)

        # Decoder initialization
        decoder_input = history[:, -1:, :]  # Last position as start
        predictions = []

        # Decode future
        for t in range(future_len):
            # Decoder step
            decoder_output, (hidden, cell) = self.decoder(
                decoder_input, (hidden, cell)
            )

            # Predict next position
            pred = self.output_layer(decoder_output)
            predictions.append(pred)

            # Teacher forcing
            if future is not None and torch.rand(1).item() < teacher_forcing_ratio:
                decoder_input = future[:, t:t+1, :]
            else:
                decoder_input = pred

        # Stack predictions
        predictions = torch.cat(predictions, dim=1)  # (batch, future_len, 2)

        # Compute loss if ground truth provided
        loss = None
        if future is not None:
            loss = nn.MSELoss()(predictions, future)

        return predictions, loss

    def predict(self, history: torch.Tensor) -> torch.Tensor:
        """
        Generate prediction (inference mode)

        Args:
            history: (batch, history_len, 2)

        Returns:
            prediction: (batch, future_len, 2)
        """
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(history, future=None, teacher_forcing_ratio=0.0)
        return predictions

    def get_num_params(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMTrainer:
    """Training helper for LSTM baseline"""

    def __init__(
        self,
        model: LSTMBaseline,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.model = model.to(device)
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

    def train_epoch(self, train_loader, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in train_loader:
            history = batch['history'].to(self.device)  # (batch, 4, 2)
            future = batch['future'].to(self.device)    # (batch, 6, 2)

            # Forward pass
            predictions, loss = self.model(history, future, teacher_forcing_ratio=0.5)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                history = batch['history'].to(self.device)
                future = batch['future'].to(self.device)

                predictions, loss = self.model(history, future, teacher_forcing_ratio=0.0)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Save model checkpoint"""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss
        }, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['epoch'], checkpoint['loss']


def create_lstm_dataloader(data_path: str, batch_size: int = 128, train_split: float = 0.9):
    """
    Create dataloaders for LSTM baseline
    Converts token data to continuous trajectories
    """
    import pickle
    from torch.utils.data import Dataset, DataLoader

    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # FIX 1: Use saved config from data file
    config = data['config']
    from trajectory_processor import TrajectoryProcessor
    processor = TrajectoryProcessor(config)

    # Convert to continuous trajectories
    dataset_list = []
    for sample in data['data']:
        # FIX 2: Use ego_history/ego_future directly (not tokens)
        history = np.array(sample['ego_history'])
        future = np.array(sample['ego_future'])

        if len(history) == 4 and len(future) == 6:
            dataset_list.append({
                'history': torch.FloatTensor(history),
                'future': torch.FloatTensor(future),
                'is_lane_change': sample['is_lane_change']
            })

    # Split
    n_train = int(len(dataset_list) * train_split)
    train_data = dataset_list[:n_train]
    val_data = dataset_list[n_train:]

    # Custom dataset
    class TrajectoryDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    # Dataloaders
    train_loader = DataLoader(
        TrajectoryDataset(train_data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        TrajectoryDataset(val_data),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    return train_loader, val_loader


def train_lstm_baseline(
    data_path: str = './data/trajectories_full.pkl',
    epochs: int = 30,
    batch_size: int = 128,
    output_dir: str = './checkpoints'
):
    """
    Train LSTM baseline model

    Usage:
        python lstm_baseline.py
    """
    import os

    print("="*60)
    print("Training LSTM Baseline")
    print("="*60)

    # Create model
    model = LSTMBaseline(
        input_dim=2,
        hidden_dim=128,
        num_layers=2,
        dropout=0.1
    )

    print(f"\nModel: LSTM Encoder-Decoder")
    print(f"Parameters: {model.get_num_params():,}")

    # Create dataloaders
    print("\nLoading data...")
    train_loader, val_loader = create_lstm_dataloader(
        data_path,
        batch_size=batch_size
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create trainer
    trainer = LSTMTrainer(model, learning_rate=1e-3)

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("="*60)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = trainer.train_epoch(train_loader, epoch)

        # Validate
        val_loss = trainer.validate(val_loader)

        # Update learning rate
        old_lr = trainer.optimizer.param_groups[0]['lr']
        trainer.scheduler.step(val_loss)
        new_lr = trainer.optimizer.param_groups[0]['lr']

        # Print progress
        print(f"Epoch {epoch}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"LR: {new_lr:.6f}")

        # Log LR change
        if new_lr != old_lr:
            print(f"  → Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(output_dir, exist_ok=True)
            trainer.save_checkpoint(
                f'{output_dir}/lstm_baseline_best.pt',
                epoch,
                val_loss
            )
            print(f"  ✓ Saved best model (val_loss: {val_loss:.4f})")

        # Regular checkpoint
        if epoch % 10 == 0:
            trainer.save_checkpoint(
                f'{output_dir}/lstm_baseline_epoch_{epoch}.pt',
                epoch,
                val_loss
            )

    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {output_dir}/lstm_baseline_best.pt")
    print("="*60)


def test_lstm_model():
    """Test LSTM model"""
    print("Testing LSTM Baseline...")

    # Create model
    model = LSTMBaseline(hidden_dim=128, num_layers=2)

    # Test forward pass
    batch_size = 4
    history = torch.randn(batch_size, 4, 2)  # 4 timesteps
    future = torch.randn(batch_size, 6, 2)   # 6 timesteps

    # Forward pass
    predictions, loss = model(history, future)

    print(f"✓ Input history: {history.shape}")
    print(f"✓ Ground truth future: {future.shape}")
    print(f"✓ Predictions: {predictions.shape}")
    print(f"✓ Loss: {loss.item():.4f}")
    print(f"✓ Parameters: {model.get_num_params():,}")

    # Test inference
    pred_only = model.predict(history)
    print(f"✓ Inference predictions: {pred_only.shape}")

    print("\n✓ LSTM baseline test passed!")

def evaluate_lstm_baseline(
        checkpoint_path: str = './checkpoints/lstm_baseline_best.pt',
        data_path: str = './data/trajectories_full.pkl',
        batch_size: int = 128
):
    """Evaluate LSTM baseline"""
    import pickle

    print("Evaluating LSTM Baseline...")

    # Load model
    model = LSTMBaseline(hidden_dim=128, num_layers=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Load test data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    test_samples = data['data'][-1000:]  # Last 1000 samples

    ade_list = []
    fde_list = []
    lane_change_detected = 0
    total_lane_changes = 0

    with torch.no_grad():
        for sample in tqdm(test_samples):
            history = torch.FloatTensor(sample['ego_history']).unsqueeze(0).to(device)
            future_gt = np.array(sample['ego_future'])

            # Predict
            pred = model.predict(history).cpu().numpy()[0]

            # Metrics
            errors = np.linalg.norm(pred - future_gt, axis=1)
            ade_list.append(errors.mean())
            fde_list.append(errors[-1])

            # Lane change
            if sample['is_lane_change']:
                total_lane_changes += 1
                lateral = abs(pred[-1, 1] - pred[0, 1])
                if lateral > 2.0:
                    lane_change_detected += 1

    # Results
    print("\n" + "=" * 60)
    print("LSTM Baseline Results")
    print("=" * 60)
    print(f"ADE: {np.mean(ade_list):.3f} ± {np.std(ade_list):.3f} m")
    print(f"FDE: {np.mean(fde_list):.3f} ± {np.std(fde_list):.3f} m")
    print(f"Lane Change: {lane_change_detected / total_lane_changes * 100:.1f}%")
    print(f"Diversity: N/A (deterministic)")
    print("=" * 60)


class BestOfNTrajectoryLoss(nn.Module):
    def __init__(self, processor, K=5):
        super().__init__()
        self.processor = processor
        self.K = K  # Number of samples

    def forward(self, model, input_tokens, gt_future):
        """
        Args:
            model: TrajectoryGPT
            input_tokens: (batch, seq_len)
            gt_future: (batch, 6, 2) ground truth
        """
        batch_size = input_tokens.size(0)

        # Generate K samples per input
        all_losses = []

        for k in range(self.K):
            # Generate trajectory
            generated = model.generate(
                input_tokens,
                max_new_tokens=6,
                temperature=1.0,  # Keep diversity
                return_logits=True  # Need logits for backprop
            )

            # Reconstruct trajectory
            pred_tokens = generated[:, input_tokens.size(1):]
            pred_trajs = []

            for b in range(batch_size):
                tokens = pred_tokens[b].tolist()
                tokens = [t for t in tokens if t < 1024]
                traj = self.processor.discretizer.reconstruct_trajectory(tokens)

                if len(traj) > 0:
                    # Pad/truncate to 6 waypoints
                    if len(traj) < 6:
                        traj = np.pad(traj, ((0, 6 - len(traj)), (0, 0)))
                    else:
                        traj = traj[:6]
                    pred_trajs.append(traj)

            pred_trajs = torch.FloatTensor(pred_trajs).to(gt_future.device)

            # Compute loss for this sample
            loss = F.mse_loss(pred_trajs, gt_future)
            all_losses.append(loss)

        # Only use best (minimum loss)
        best_loss = torch.stack(all_losses).min()

        return best_loss

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_lstm_model()
    else:
        # Train model
        train_lstm_baseline(
            data_path='./data/trajectories_full.pkl',
            epochs=30,
            batch_size=128
        )