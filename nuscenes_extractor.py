"""
Extract Full nuScenes Trainval Dataset
Handles both mini and full datasets
"""

import sys
import os
import argparse

sys.path.append('.')
from utils.nuscenes_helper import NuScenesTrajectoryExtractor


def main():
    parser = argparse.ArgumentParser(description='Extract trajectories from nuScenes')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='nuScenes version to extract')
    parser.add_argument('--dataroot', type=str, default='./data/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--output', type=str, default=None,
                        help='Output pickle file name')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to extract (None = all)')

    args = parser.parse_args()

    # Auto-generate output name
    if args.output is None:
        if 'mini' in args.version:
            args.output = './data/trajectories_mini.pkl'
        elif 'trainval' in args.version:
            args.output = './data/trajectories_full.pkl'
        else:
            args.output = './data/trajectories.pkl'

    print("=" * 70)
    print("nuScenes Trajectory Extraction")
    print("=" * 70)
    print(f"Version: {args.version}")
    print(f"Dataroot: {args.dataroot}")
    print(f"Output: {args.output}")
    print(f"Max samples: {args.max_samples if args.max_samples else 'All'}")
    print("=" * 70)

    # Check if data exists
    if not os.path.exists(args.dataroot):
        print(f"\n Error: nuScenes data not found at {args.dataroot}")
        print("\nPlease download nuScenes data first:")
        if 'mini' in args.version:
            print("  wget https://www.nuscenes.org/data/v1.0-mini.tgz")
            print(f"  tar -xf v1.0-mini.tgz -C {args.dataroot}")
        else:
            print("  See SCALING_TO_FULL_DATASET.md for download instructions")
        return

    # Create extractor
    print("\nInitializing extractor...")
    try:
        extractor = NuScenesTrajectoryExtractor(
            dataroot=args.dataroot,
            version=args.version
        )
    except Exception as e:
        print(f"\n Error loading nuScenes: {e}")
        print("\nMake sure you have the correct version installed:")
        print(f"  Expected: {args.version} in {args.dataroot}")
        return

    # Extract trajectories
    print("\nExtracting trajectories...")
    print("This may take a while for large datasets...")

    try:
        dataset = extractor.extract_dataset(
            output_path=args.output,
            max_samples=args.max_samples
        )

        print("\n" + "=" * 70)
        print(" Extraction Complete!")
        print("=" * 70)
        print(f"Extracted: {len(dataset)} trajectories")
        print(f"Saved to: {args.output}")

        # Print statistics
        lane_changes = sum(1 for d in dataset if d['is_lane_change'])
        print(f"\nStatistics:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Lane changes: {lane_changes} ({100 * lane_changes / len(dataset):.1f}%)")

        print("\n Ready for training!")
        print(f"Next command:")
        print(f"  python training/train.py --data {args.output} --epochs 50")

    except Exception as e:
        print(f"\n Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        return



if __name__ == "__main__":
    main()