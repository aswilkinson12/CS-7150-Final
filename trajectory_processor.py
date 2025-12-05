"""
TrajectoryGPT - Trajectory Processing Module
Handles extraction, ego-centric transformation, and discretization
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory processing"""
    # Grid parameters
    grid_size: int = 16  # 16×16 grid
    cell_size: float = 2.0  # 2m per cell
    coverage: float = 32.0  # 32m total coverage

    # Temporal parameters
    history_length: int = 4  # 2 seconds @ 2Hz
    future_length: int = 6   # 3 seconds @ 2Hz
    sample_rate: float = 2.0  # Hz

    # Vocabulary
    num_spatial_tokens: int = 256  # 16×16 grid
    pad_token: int = 256
    start_token: int = 257
    end_token: int = 258
    vocab_size: int = 259

    def __post_init__(self):
        assert self.grid_size ** 2 == self.num_spatial_tokens
        assert self.grid_size * self.cell_size == self.coverage


class CoordinateTransform:
    """Handle ego-centric coordinate transformations"""

    @staticmethod
    def to_ego_centric(
        trajectory: np.ndarray,
        agent_pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """
        Transform trajectory to ego-centric coordinates

        Args:
            trajectory: (N, 2) array of [x, y] positions in global coords
            agent_pose: (x, y, heading) of agent at t=0

        Returns:
            (N, 2) array in ego-centric frame (agent at origin, heading = +x)
        """
        x0, y0, heading = agent_pose

        # Translate to origin
        trajectory_centered = trajectory - np.array([x0, y0])

        # Rotate to align heading with +x axis
        cos_h = np.cos(-heading)
        sin_h = np.sin(-heading)
        rotation_matrix = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])

        trajectory_ego = trajectory_centered @ rotation_matrix.T
        return trajectory_ego

    @staticmethod
    def from_ego_centric(
        trajectory: np.ndarray,
        agent_pose: Tuple[float, float, float]
    ) -> np.ndarray:
        """Transform from ego-centric back to global coordinates"""
        x0, y0, heading = agent_pose

        # Rotate back
        cos_h = np.cos(heading)
        sin_h = np.sin(heading)
        rotation_matrix = np.array([
            [cos_h, -sin_h],
            [sin_h, cos_h]
        ])

        trajectory_global = trajectory @ rotation_matrix.T
        # Translate back
        trajectory_global += np.array([x0, y0])
        return trajectory_global


class GridDiscretizer:
    """Discretize continuous trajectories into grid tokens"""

    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.grid_min = -config.coverage / 2
        self.grid_max = config.coverage / 2

    def xy_to_token(self, x: float, y: float) -> Optional[int]:
        """
        Convert (x, y) coordinate to grid token

        Returns:
            Token ID (0-255) or None if out of bounds
        """
        # Check bounds
        if not (self.grid_min <= x < self.grid_max and
                self.grid_min <= y < self.grid_max):
            return None

        # Convert to grid indices
        col = int((x - self.grid_min) / self.config.cell_size)
        row = int((y - self.grid_min) / self.config.cell_size)

        # Ensure within bounds
        col = np.clip(col, 0, self.config.grid_size - 1)
        row = np.clip(row, 0, self.config.grid_size - 1)

        # Token ID: row-major order
        token = row * self.config.grid_size + col
        return int(token)

    def token_to_xy(self, token: int) -> Tuple[float, float]:
        """
        Convert grid token back to (x, y) coordinate (cell center)

        Args:
            token: Token ID (0-255)

        Returns:
            (x, y) at center of grid cell
        """
        assert 0 <= token < self.config.num_spatial_tokens

        row = token // self.config.grid_size
        col = token % self.config.grid_size

        # Get cell center
        x = self.grid_min + (col + 0.5) * self.config.cell_size
        y = self.grid_min + (row + 0.5) * self.config.cell_size

        return x, y

    def discretize_trajectory(self, trajectory: np.ndarray) -> List[int]:
        """
        Discretize full trajectory to token sequence

        Args:
            trajectory: (N, 2) array of [x, y] positions

        Returns:
            List of token IDs (filters out-of-bounds points)
        """
        tokens = []
        for x, y in trajectory:
            token = self.xy_to_token(x, y)
            if token is not None:
                tokens.append(token)
        return tokens

    def reconstruct_trajectory(self, tokens: List[int]) -> np.ndarray:
        """
        Reconstruct trajectory from tokens

        Args:
            tokens: List of token IDs

        Returns:
            (N, 2) array of reconstructed [x, y] positions
        """
        trajectory = []
        for token in tokens:
            if token < self.config.num_spatial_tokens:
                x, y = self.token_to_xy(token)
                trajectory.append([x, y])
        return np.array(trajectory)


class TrajectoryProcessor:
    """Main class for processing trajectories"""

    def __init__(self, config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()
        self.transform = CoordinateTransform()
        self.discretizer = GridDiscretizer(self.config)

    def process_trajectory(
        self,
        trajectory: np.ndarray,
        agent_pose: Tuple[float, float, float],
        return_continuous: bool = False
    ) -> Dict:
        """
        Full processing pipeline: global → ego-centric → tokens

        Args:
            trajectory: (N, 2) global coordinates
            agent_pose: (x, y, heading) at t=0
            return_continuous: Whether to include continuous ego trajectory

        Returns:
            Dict with tokens and optional continuous trajectory
        """
        # Transform to ego-centric
        trajectory_ego = self.transform.to_ego_centric(trajectory, agent_pose)

        # Discretize
        tokens = self.discretizer.discretize_trajectory(trajectory_ego)

        result = {'tokens': tokens}
        if return_continuous:
            result['continuous_ego'] = trajectory_ego

        return result

    def create_training_sequence(
        self,
        history_tokens: List[int],
        future_tokens: List[int]
    ) -> Dict[str, List[int]]:
        """
        Create input/target sequences for training

        Format: [START] + history + future + [END]
        Input:  [START] + history + future[:-1]
        Target: history + future + [END]

        Args:
            history_tokens: Past trajectory tokens
            future_tokens: Future trajectory tokens

        Returns:
            Dict with 'input' and 'target' sequences
        """
        # Full sequence with special tokens
        input_seq = [self.config.start_token] + history_tokens + future_tokens[:-1]
        target_seq = history_tokens + future_tokens + [self.config.end_token]

        return {
            'input': input_seq,
            'target': target_seq
        }


def test_pipeline():
    """Test the trajectory processing pipeline"""
    print("Testing TrajectoryGPT Pipeline...")

    config = TrajectoryConfig()
    processor = TrajectoryProcessor(config)

    # Create dummy trajectory
    t = np.linspace(0, 4, 10)
    trajectory = np.column_stack([
        t * 3,  # 3 m/s forward
        np.sin(t) * 2  # slight curve
    ])

    agent_pose = (0.0, 0.0, 0.0)  # at origin, heading +x

    # Process
    result = processor.process_trajectory(
        trajectory,
        agent_pose,
        return_continuous=True
    )

    print(f"✓ Original trajectory shape: {trajectory.shape}")
    print(f"✓ Number of tokens: {len(result['tokens'])}")
    print(f"✓ Token sequence: {result['tokens'][:5]}...")

    # Test reconstruction
    reconstructed = processor.discretizer.reconstruct_trajectory(result['tokens'])
    print(f"✓ Reconstructed shape: {reconstructed.shape}")

    # Test training sequence creation
    history = result['tokens'][:4]
    future = result['tokens'][4:]
    train_seq = processor.create_training_sequence(history, future)

    print(f"✓ Input sequence length: {len(train_seq['input'])}")
    print(f"✓ Target sequence length: {len(train_seq['target'])}")
    print(f"✓ Special tokens - START: {config.start_token}, END: {config.end_token}")

    print("\n✓ Pipeline test passed!")


if __name__ == "__main__":
    test_pipeline()