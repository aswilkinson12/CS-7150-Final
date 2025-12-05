"""
TrajectoryGPT - nuScenes Data Extraction
Extract trajectories from nuScenes for training
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from trajectory_processor import TrajectoryProcessor, TrajectoryConfig

# Import nuScenes (will fail gracefully if not installed yet)
try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.prediction import PredictHelper
    NUSCENES_AVAILABLE = True
except ImportError:
    NUSCENES_AVAILABLE = False
    print("Warning: nuScenes not installed.")


class NuScenesTrajectoryExtractor:
    """Extract and process trajectories from nuScenes dataset"""

    def __init__(
        self,
        dataroot: str = './data/nuscenes',
        version: str = 'v1.0-mini',
        config: TrajectoryConfig = None
    ):
        """
        Args:
            dataroot: Path to nuScenes data
            version: Dataset version ('v1.0-mini' or 'v1.0-trainval')
            config: Trajectory processing config
        """
        if not NUSCENES_AVAILABLE:
            raise ImportError("Install nuscenes-devkit first")

        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        self.helper = PredictHelper(self.nusc)
        self.config = config or TrajectoryConfig()
        self.processor = TrajectoryProcessor(self.config)

        print(f"Loaded nuScenes {version}")
        print(f"Scenes: {len(self.nusc.scene)}")

    def get_agent_trajectory(
        self,
        instance_token: str,
        sample_token: str
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Get history and future trajectory for an agent

        Args:
            instance_token: Agent instance identifier
            sample_token: Current sample (t=0)

        Returns:
            history: (H, 2) past positions
            future: (F, 2) future positions
            metadata: Additional info (agent type, etc.)
        """
        # Get historical trajectory (2 seconds at 2Hz = 4 points)
        history = self.helper.get_past_for_agent(
            instance_token,
            sample_token,
            seconds=2.0,
            in_agent_frame=False  # We'll transform ourselves
        )

        # Get future trajectory (3 seconds at 2Hz = 6 points)
        future = self.helper.get_future_for_agent(
            instance_token,
            sample_token,
            seconds=3.0,
            in_agent_frame=False
        )

        # Get current position and heading for ego-centric transform
        current_sample = self.nusc.get('sample', sample_token)

        # Get annotation for this instance at current time
        anns = [self.nusc.get('sample_annotation', tok)
                for tok in current_sample['anns']]
        agent_ann = [ann for ann in anns
                     if ann['instance_token'] == instance_token]

        if not agent_ann:
            return None, None, None

        agent_ann = agent_ann[0]

        # Extract pose (x, y, heading)
        x, y = agent_ann['translation'][:2]

        # Get heading from quaternion
        rotation = agent_ann['rotation']
        heading = self._quaternion_to_heading(rotation)

        metadata = {
            'category': agent_ann['category_name'],
            'instance_token': instance_token,
            'sample_token': sample_token,
            'pose': (x, y, heading)
        }

        return history, future, metadata

    @staticmethod
    def _quaternion_to_heading(q: List[float]) -> float:
        """
        Convert quaternion to heading angle (yaw)

        Args:
            q: [w, x, y, z] quaternion

        Returns:
            Heading in radians
        """
        w, x, y, z = q
        # Yaw (z-axis rotation)
        heading = np.arctan2(2.0 * (w * z + x * y),
                            1.0 - 2.0 * (y * y + z * z))
        return heading

    def extract_dataset(
        self,
        output_path: str = './data/trajectories.pkl',
        min_history_len: int = 4,
        min_future_len: int = 6,
        max_samples: int = None
    ) -> List[Dict]:
        """
        Extract all valid trajectories from dataset

        Args:
            output_path: Where to save processed data
            min_history_len: Minimum history points required
            min_future_len: Minimum future points required
            max_samples: Limit number of samples (for debugging)

        Returns:
            List of processed trajectory samples
        """
        dataset = []
        sample_count = 0

        print("Extracting trajectories from nuScenes...")

        # Iterate through all samples
        for scene in tqdm(self.nusc.scene, desc="Processing scenes"):
            sample_token = scene['first_sample_token']

            while sample_token:
                if max_samples and sample_count >= max_samples:
                    break

                sample = self.nusc.get('sample', sample_token)

                # Get all agents in this sample
                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)
                    instance_token = ann['instance_token']

                    # Skip if not a vehicle
                    if 'vehicle' not in ann['category_name']:
                        continue

                    # Extract trajectory
                    history, future, metadata = self.get_agent_trajectory(
                        instance_token, sample_token
                    )

                    if history is None or future is None:
                        continue

                    # Filter by length requirements
                    if len(history) < min_history_len or len(future) < min_future_len:
                        continue

                    # Ensure we have exactly the right lengths
                    history = history[:min_history_len]
                    future = future[:min_future_len]

                    # Process to ego-centric and discretize
                    agent_pose = metadata['pose']

                    # Combine for processing
                    full_trajectory = np.vstack([history, future])

                    result = self.processor.process_trajectory(
                        full_trajectory,
                        agent_pose,
                        return_continuous=True
                    )

                    # Only keep if we have enough valid tokens
                    if len(result['tokens']) < min_history_len + min_future_len:
                        continue

                    # Split into history and future tokens
                    history_tokens = result['tokens'][:min_history_len]
                    future_tokens = result['tokens'][min_history_len:]

                    # Create training sequences
                    train_seq = self.processor.create_training_sequence(
                        history_tokens, future_tokens
                    )

                    # Check for lane change (simplified heuristic)
                    lateral_displacement = np.abs(
                        result['continuous_ego'][-1, 1] - result['continuous_ego'][0, 1]
                    )
                    is_lane_change = lateral_displacement > 2.0  # >2m lateral movement

                    # Store processed sample
                    dataset.append({
                        'input_tokens': train_seq['input'],
                        'target_tokens': train_seq['target'],
                        'history_tokens': history_tokens,
                        'future_tokens': future_tokens,
                        'continuous_ego': result['continuous_ego'],
                        'agent_type': metadata['category'],
                        'is_lane_change': is_lane_change,
                        'instance_token': instance_token,
                        'sample_token': sample_token
                    })

                    sample_count += 1

                # Next sample in scene
                sample_token = sample['next']

                if max_samples and sample_count >= max_samples:
                    break

        print(f"\nExtracted {len(dataset)} valid trajectories")

        # Calculate statistics
        lane_changes = sum(1 for d in dataset if d['is_lane_change'])
        print(f"Lane changes: {lane_changes} ({100*lane_changes/len(dataset):.1f}%)")

        # Save dataset
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump({
                'data': dataset,
                'config': self.config,
                'stats': {
                    'total_samples': len(dataset),
                    'lane_changes': lane_changes,
                    'version': self.nusc.version
                }
            }, f)

        print(f"Saved dataset to {output_path}")
        return dataset


def main():
    """Main extraction pipeline"""
    # Check if nuScenes data exists
    dataroot = './data/nuscenes'
    if not os.path.exists(dataroot):
        print(f"Error: nuScenes data not found at {dataroot}")
        print("Please download and extract nuScenes data first:")
        print("  wget https://www.nuscenes.org/data/v1.0-mini.tgz")
        print(f"  tar -xf v1.0-mini.tgz -C {dataroot}")
        return

    # Create extractor
    extractor = NuScenesTrajectoryExtractor(
        dataroot=dataroot,
        version='v1.0-mini'  # Start with mini
    )

    # Extract dataset (limit to 1000 for testing)
    dataset = extractor.extract_dataset(
        output_path='../data/trajectories_mini.pkl',
        max_samples=1000
    )

    print("\n✓ Data extraction complete!")
    print(f"✓ Ready for model training with {len(dataset)} samples")


if __name__ == "__main__":
    main()