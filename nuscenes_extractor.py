"""
nuScenes Extractor - Scene-Aware Trajectories
Extracts: ego trajectory + scene context + nearby agents
Uses ONLY metadata - no sensor data needed (438MB, not 350GB)

FIXED VERSION:
1. Velocity filter: Skip stopped/parked vehicles
2. Agent coordinates: Transform to ego-centric before discretization
"""

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple, Optional

from trajectory_processor import TrajectoryProcessor, TrajectoryConfig


class NuScenesExtractor:
    """
    Extract scene-aware trajectories from nuScenes metadata
    Includes: scene type, lanes, intersections, nearby agents
    """

    def __init__(self, dataroot: str, version: str = 'v1.0-trainval'):
        print(f"Loading nuScenes {version}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)

        # Use processor
        # Create processor with 32x32 grid (1m cells instead of 16x16 with 2m cells)
        config = TrajectoryConfig(
            coverage=32.0,
            grid_size=32,      # Was 16
            cell_size=1.0,     # Was 2.0 (must be coverage/grid_size = 32/32 = 1m)
            history_length=4,
            future_length=6,
            enhanced_mode=True
        )
        self.processor = TrajectoryProcessor(config)

        print(f"Loaded {len(self.nusc.scene)} scenes")

    def get_nearby_agents(
        self,
        sample_token: str,
        ego_pose: np.ndarray,
        ego_rotation: Quaternion,  # ★ ADDED: Need rotation for transformation
        radius: float = 30.0,
        max_agents: int = 5
    ) -> List[Dict]:
        """
        Get nearby agents from metadata
        ★ FIXED: Now transforms agent positions to ego-centric coordinates

        Args:
            sample_token: Current sample
            ego_pose: Ego vehicle pose (x, y) in global coordinates
            ego_rotation: Ego vehicle rotation (quaternion)
            radius: Search radius in meters
            max_agents: Maximum number of agents

        Returns:
            List of agent data with ego-centric positions
        """
        sample = self.nusc.get('sample', sample_token)
        nearby_agents = []

        # Get all annotations in current sample
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)

            # Filter by category (vehicles only)
            category = ann['category_name']
            if not any(veh in category for veh in ['vehicle', 'car', 'truck', 'bus']):
                continue

            # Get agent position (GLOBAL coordinates)
            agent_pos_global = np.array(ann['translation'][:2])

            # Compute distance to ego
            distance = np.linalg.norm(agent_pos_global - ego_pose)

            if distance < radius and distance > 0.5:  # Exclude ego itself
                # ★ TRANSFORM to ego-centric coordinates
                agent_pos_ego = self._global_to_ego_point(
                    agent_pos_global,
                    ego_pose,
                    ego_rotation
                )

                # Get agent trajectory (in global coordinates)
                agent_trajectory_global = self._get_agent_trajectory(ann_token)

                # ★ TRANSFORM trajectory to ego-centric
                if len(agent_trajectory_global) > 0:
                    agent_trajectory_ego = self._global_to_ego_trajectory(
                        agent_trajectory_global,
                        ego_pose,
                        ego_rotation
                    )
                else:
                    agent_trajectory_ego = np.array([])

                nearby_agents.append({
                    'type': category,
                    'position': agent_pos_ego,         # ★ Now ego-centric
                    'distance': distance,
                    'trajectory': agent_trajectory_ego  # ★ Now ego-centric
                })

        # Sort by distance and return closest
        nearby_agents.sort(key=lambda x: x['distance'])
        return nearby_agents[:max_agents]

    def _global_to_ego_point(
        self,
        point: np.ndarray,
        ego_pose: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """
        Transform single point from global to ego-centric coordinates

        Args:
            point: (x, y) in global coordinates
            ego_pose: Ego position (x, y) in global coordinates
            ego_rotation: Ego rotation (quaternion)

        Returns:
            (x, y) in ego-centric coordinates
        """
        # Translate
        point_centered = point - ego_pose

        # Rotate to ego frame
        yaw = quaternion_yaw(ego_rotation)
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)

        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw]
        ])

        point_ego = point_centered @ rotation_matrix.T
        return point_ego

    def _global_to_ego_trajectory(
        self,
        trajectory: np.ndarray,
        ego_pose: np.ndarray,
        ego_rotation: Quaternion
    ) -> np.ndarray:
        """Transform trajectory from global to ego-centric coordinates"""
        if len(trajectory) == 0:
            return trajectory

        ego_trajectory = np.array([
            self._global_to_ego_point(point, ego_pose, ego_rotation)
            for point in trajectory
        ])
        return ego_trajectory

    def _get_agent_trajectory(self, ann_token: str, history_length: int = 4) -> np.ndarray:
        """Get agent's historical trajectory (in global coordinates)"""
        trajectory = []
        current_token = ann_token

        for _ in range(history_length):
            if current_token == '':
                break

            ann = self.nusc.get('sample_annotation', current_token)
            trajectory.append(ann['translation'][:2])

            # Move to previous annotation
            current_token = ann['prev']

        if len(trajectory) > 0:
            trajectory = np.array(trajectory[::-1])  # Reverse to chronological
            return trajectory
        else:
            return np.array([])

    def get_scene_context(self, sample: Dict, scene: Dict) -> List[int]:
        """
        Extract scene context from metadata

        Args:
            sample: nuScenes sample
            scene: nuScenes scene

        Returns:
            List of scene context tokens
        """
        # Use processor scene context extraction
        scene_data = {
            'description': scene['description'],
            'location': self.nusc.get('log', scene['log_token'])['location']
        }

        return self.processor.extract_scene_context(scene_data)

    def get_lane_context(self, trajectory: np.ndarray) -> List[int]:
        """Determine lane context from trajectory"""
        return self.processor.extract_lane_context(trajectory)

    def _get_ego_trajectory(
        self,
        sample_token: str,
        instance_token: str,
        history_length: int = 4,
        future_length: int = 6
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[Quaternion]]:
        """
        Get ego-centric trajectory

        ★ CRITICAL FIX: Now returns pose and rotation for unified coordinate frame

        Args:
            sample_token: Current sample token
            instance_token: Vehicle instance token
            history_length: Number of past samples
            future_length: Number of future samples

        Returns:
            (history, future, current_pose, current_rotation) or (None, None, None, None) if invalid
            ★ pose and rotation are from tracked vehicle for unified frame
        """
        # Get the annotation for this vehicle in this sample
        sample = self.nusc.get('sample', sample_token)

        # Find the annotation for this instance
        ann_token = None
        for token in sample['anns']:
            ann = self.nusc.get('sample_annotation', token)
            if ann['instance_token'] == instance_token:
                ann_token = token
                break

        if ann_token is None:
            return None, None, None, None

        # Get current annotation
        current_ann = self.nusc.get('sample_annotation', ann_token)
        current_pose = np.array(current_ann['translation'][:2])
        current_rotation = Quaternion(current_ann['rotation'])

        # Collect history
        history_global = []
        temp_token = ann_token
        for _ in range(history_length):
            if temp_token == '':
                break
            ann = self.nusc.get('sample_annotation', temp_token)
            history_global.append(ann['translation'][:2])
            temp_token = ann['prev']

        if len(history_global) < history_length:
            return None, None, None, None

        history_global = np.array(history_global[::-1])  # Reverse to chronological

        # Collect future
        future_global = []
        temp_token = current_ann['next']
        for _ in range(future_length):
            if temp_token == '':
                break
            ann = self.nusc.get('sample_annotation', temp_token)
            future_global.append(ann['translation'][:2])
            temp_token = ann['next']

        if len(future_global) < future_length:
            return None, None, None, None

        future_global = np.array(future_global)

        # Convert to ego-centric coordinates
        def global_to_ego(points, ego_pose, ego_rotation):
            """Transform global coordinates to ego-centric"""
            # Translate
            points_centered = points - ego_pose

            # Rotate to ego frame
            yaw = quaternion_yaw(ego_rotation)
            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)

            rotation_matrix = np.array([
                [cos_yaw, -sin_yaw],
                [sin_yaw, cos_yaw]
            ])

            points_ego = points_centered @ rotation_matrix.T
            return points_ego

        history_ego = global_to_ego(history_global, current_pose, current_rotation)
        future_ego = global_to_ego(future_global, current_pose, current_rotation)

        # ★ CRITICAL FIX: Return pose and rotation for unified frame
        return history_ego, future_ego, current_pose, current_rotation

    def extract_enhanced_sample(
        self,
        sample_token: str,
        instance_token: str,
        scene: Dict
    ) -> Optional[Dict]:
        """
        Extract single enhanced trajectory sample

        Returns:
            Dict with:
                - ego_history/future
                - scene_context
                - lane_context
                - nearby_agents
                - input_tokens
                - target_tokens
        """
        sample = self.nusc.get('sample', sample_token)

        # Get ego trajectory WITH pose and rotation for unified frame
        # ★ CRITICAL FIX: Now returns pose/rotation from tracked vehicle
        history, future, ego_pose, ego_rotation = self._get_ego_trajectory(
            sample_token, instance_token
        )

        if history is None or future is None:
            return None

        # ★ NOW FIXED: ego_pose and ego_rotation are from the SAME frame as history/future
        # All agents will be transformed using the tracked vehicle's frame (not LIDAR!)

        # Extract scene context
        scene_context = self.get_scene_context(sample, scene)

        # Extract lane context
        lane_context = self.get_lane_context(history)

        # Get nearby agents using UNIFIED frame
        # ★ CRITICAL: Uses same pose/rotation as ego trajectory
        nearby_agents_data = self.get_nearby_agents(
            sample_token,
            ego_pose,
            ego_rotation,  # ★ Now from tracked vehicle, not LIDAR!
            radius=30.0,
            max_agents=3
        )

        # Encode agents
        agent_tokens = self.processor.encode_nearby_agents(nearby_agents_data)

        # Create enhanced sequence using processor method
        sequence = self.processor.process_enhanced_trajectory(
            history,
            future,
            scene_context,
            lane_context,
            agent_tokens
        )

        # Check if lane change
        is_lane_change = abs(future[-1, 1] - future[0, 1]) > 2.0

        return {
            'ego_history': history,
            'ego_future': future,
            'scene_context': scene_context,
            'lane_context': lane_context,
            'agent_tokens': agent_tokens,
            'num_nearby_agents': len(nearby_agents_data),
            'nearby_agents_data': nearby_agents_data,
            'input_tokens': sequence['input_tokens'],
            'target_tokens': sequence['target_tokens'],
            'input': sequence['input_tokens'],
            'target': sequence['target_tokens'],
            'is_lane_change': is_lane_change,
            'scene_description': scene['description']
        }

    def extract_enhanced_dataset(
        self,
        output_path: str = './data/trajectories_full.pkl',
        max_samples: Optional[int] = None
    ) -> List[Dict]:
        """
        Extract complete enhanced dataset

        Args:
            output_path: Where to save
            max_samples: Maximum samples (None = all)

        Returns:
            List of enhanced trajectory samples
        """
        print("\nExtracting trajectories with TrajectoryProcessor...")
        print(f"Scene context: ✓")
        print(f"Lane context: ✓")
        print(f"Nearby agents: ✓ (ego-centric coordinates)")
        print(f"Filter: Light (removes only parked vehicles, <1m movement)")
        print(f"Vocabulary: {self.processor.config.vocab_size}")

        dataset = []
        sample_count = 0

        for scene in tqdm(self.nusc.scene, desc="Processing scenes"):
            scene_token = scene['token']
            sample_token = scene['first_sample_token']

            while sample_token != '':
                sample = self.nusc.get('sample', sample_token)

                # Process each vehicle annotation
                for ann_token in sample['anns']:
                    ann = self.nusc.get('sample_annotation', ann_token)

                    # Filter vehicles
                    if 'vehicle' not in ann['category_name']:
                        continue

                    enhanced_sample = self.extract_enhanced_sample(
                        sample_token,
                        ann['instance_token'],
                        scene
                    )

                    if enhanced_sample is not None:
                        # Light filter: Remove ONLY truly stationary vehicles
                        # Keep slow traffic, stop lights, deceleration, etc.

                        hist = enhanced_sample['ego_history']
                        fut = enhanced_sample['ego_future']

                        # Total displacement from start to end
                        hist_displacement = np.linalg.norm(hist[-1] - hist[0])
                        fut_displacement = np.linalg.norm(fut[-1] - fut[0])

                        # Skip ONLY if stationary in BOTH history AND future
                        # Threshold: <1m over 2s AND <1m over 3s = truly parked
                        if hist_displacement < 1.0 and fut_displacement < 1.0:
                            continue  # Parked/stationary vehicle

                        dataset.append(enhanced_sample)
                        sample_count += 1

                        if max_samples and sample_count >= max_samples:
                            break

                if max_samples and sample_count >= max_samples:
                    break

                sample_token = sample['next']

            if max_samples and sample_count >= max_samples:
                break

        # Compute statistics
        lane_changes = sum(1 for d in dataset if d['is_lane_change'])
        avg_agents = np.mean([d['num_nearby_agents'] for d in dataset]) if dataset else 0

        stats = {
            'total_samples': len(dataset),
            'lane_changes': lane_changes,
            'lane_change_rate': lane_changes / len(dataset) if dataset else 0,
            'avg_nearby_agents': avg_agents,
            'vocab_size': self.processor.config.vocab_size
        }

        # Save
        output_data = {
            'data': dataset,
            'stats': stats,
            'config': self.processor.config
        }

        with open(output_path, 'wb') as f:
            pickle.dump(output_data, f)

        print(f"\n✓ Extracted {len(dataset)} samples")
        print(f"  Lane changes: {lane_changes} ({100*lane_changes/len(dataset):.1f}%)")
        print(f"  Avg nearby agents: {avg_agents:.1f}")
        print(f"  Vocabulary size: {stats['vocab_size']}")
        print(f"  Saved to: {output_path}")

        return dataset


def main():
    """Extract enhanced dataset"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', type=str,
                       default='./data/nuscenes',
                       help='Path to nuScenes data')
    parser.add_argument('--version', type=str,
                       default='v1.0-trainval',
                       help='nuScenes version')
    parser.add_argument('--output', type=str,
                       default='./data/trajectories_full.pkl',
                       help='Output file')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Max samples to extract')

    args = parser.parse_args()

    # Create extractor
    extractor = NuScenesExtractor(
        dataroot=args.dataroot,
        version=args.version
    )

    # Extract dataset
    extractor.extract_enhanced_dataset(
        output_path=args.output,
        max_samples=args.max_samples
    )

    print("\n✓ Extraction complete!")


if __name__ == "__main__":
    main()