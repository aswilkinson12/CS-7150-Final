"""
TrajectoryGPT - PyTorch Dataset
Dataset and DataLoader for trajectory training
"""

import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trajectory_processor import TrajectoryConfig


class TrajectoryDataset(Dataset):
    """
    PyTorch Dataset for trajectory sequences

    Loads processed trajectory data and provides sequences for training
    """

    def __init__(
        self,
        data_path: str,
        config: TrajectoryConfig = None,
        filter_lane_changes: bool = False
    ):
        """
        Args:
            data_path: Path to processed .pkl file
            config: Trajectory configuration
            filter_lane_changes: If True, only use lane change samples
        """
        self.config = config or TrajectoryConfig()

        # Load data
        print(f"Loading dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            data_dict = pickle.load(f)

        self.samples = data_dict['data']
        self.stats = data_dict.get('stats', {})

        # Filter if requested
        if filter_lane_changes:
            self.samples = [s for s in self.samples if s['is_lane_change']]
            print(f"Filtered to {len(self.samples)} lane change samples")

        print(f"Loaded {len(self.samples)} trajectory samples")
        if self.stats:
            print(f"Dataset stats: {self.stats}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training sample

        Returns:
            Dict with:
                - input: Input token sequence (includes START)
                - target: Target token sequence (includes END)
                - is_lane_change: Boolean lane change label
        """
        sample = self.samples[idx]

        return {
            'input': torch.tensor(sample['input_tokens'], dtype=torch.long),
            'target': torch.tensor(sample['target_tokens'], dtype=torch.long),
            'is_lane_change': torch.tensor(sample['is_lane_change'], dtype=torch.bool)
        }

    def get_statistics(self) -> Dict:
        """Get dataset statistics"""
        lane_changes = sum(1 for s in self.samples if s['is_lane_change'])

        seq_lengths = [len(s['input_tokens']) for s in self.samples]

        return {
            'total_samples': len(self.samples),
            'lane_changes': lane_changes,
            'lane_change_rate': lane_changes / len(self.samples),
            'avg_seq_length': sum(seq_lengths) / len(seq_lengths),
            'max_seq_length': max(seq_lengths),
            'min_seq_length': min(seq_lengths)
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function to pad sequences to same length

    Args:
        batch: List of samples from dataset

    Returns:
        Batched and padded tensors
    """
    # Get max sequence length in batch (must be same for input and target)
    max_input_len = max(len(item['input']) for item in batch)
    max_target_len = max(len(item['target']) for item in batch)
    max_len = max(max_input_len, max_target_len)

    # Pad sequences
    inputs = []
    targets = []
    lane_changes = []

    for item in batch:
        # Pad input (pad token = 256)
        input_len = len(item['input'])
        padded_input = torch.cat([
            item['input'],
            torch.full((max_len - input_len,), 256, dtype=torch.long)
        ])
        inputs.append(padded_input)

        # Pad target (use -1 for padding so it's ignored in loss)
        target_len = len(item['target'])
        padded_target = torch.cat([
            item['target'],
            torch.full((max_len - target_len,), -1, dtype=torch.long)
        ])
        targets.append(padded_target)

        lane_changes.append(item['is_lane_change'])

    return {
        'input': torch.stack(inputs),
        'target': torch.stack(targets),
        'is_lane_change': torch.stack(lane_changes)
    }


def create_dataloaders(
    train_path: str,
    val_path: str = None,
    batch_size: int = 128,
    num_workers: int = 4,
    val_split: float = 0.1
) -> tuple:
    """
    Create training and validation dataloaders

    Args:
        train_path: Path to training data
        val_path: Path to validation data (if separate file)
        batch_size: Batch size
        num_workers: Number of worker processes
        val_split: Validation split fraction if val_path not provided

    Returns:
        (train_loader, val_loader)
    """
    # Load full dataset
    dataset = TrajectoryDataset(train_path)

    # Split train/val if needed
    if val_path is None:
        # Split the dataset
        total_size = len(dataset)
        val_size = int(total_size * val_split)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        print(f"Split dataset: {train_size} train, {val_size} val")
    else:
        train_dataset = dataset
        val_dataset = TrajectoryDataset(val_path)
        print(f"Using separate validation set: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )

    return train_loader, val_loader


def test_dataset():
    """Test dataset loading"""
    print("Testing TrajectoryDataset...")

    # This will fail if data doesn't exist, but shows the interface
    try:
        dataset = TrajectoryDataset('./data/trajectories_mini.pkl')

        # Get statistics
        stats = dataset.get_statistics()
        print("\nDataset Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Get a sample
        sample = dataset[0]
        print(f"\nSample 0:")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Lane change: {sample['is_lane_change']}")

        # Test dataloader
        print("\nTesting DataLoader...")
        train_loader, val_loader = create_dataloaders(
            './data/trajectories_mini.pkl',
            batch_size=8,
            num_workers=0  # 0 for testing
        )

        batch = next(iter(train_loader))
        print(f"Batch shapes:")
        print(f"  Input: {batch['input'].shape}")
        print(f"  Target: {batch['target'].shape}")
        print(f"  Lane changes: {batch['is_lane_change'].shape}")

        print("\n✓ Dataset test passed!")

    except FileNotFoundError:
        print("Data file not found. Run nuscenes_extractor.py first.")
        print("Testing with dummy data...")

        # Create dummy data structure
        dummy_data = {
            'data': [
                {
                    'input_tokens': [257, 136, 137, 138, 139],
                    'target_tokens': [136, 137, 138, 139, 140, 258],
                    'is_lane_change': False
                }
                for _ in range(10)
            ],
            'stats': {'total_samples': 10}
        }

        # Save and load
        with open('/tmp/dummy_trajectories.pkl', 'wb') as f:
            pickle.dump(dummy_data, f)

        dataset = TrajectoryDataset('/tmp/dummy_trajectories.pkl')
        sample = dataset[0]
        print(f"✓ Dummy dataset works: {len(dataset)} samples")


if __name__ == "__main__":
    test_dataset()