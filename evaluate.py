"""
FIXED Evaluation Script
Uses processor config from data file (critical!)
"""

import torch
import numpy as np
import pickle
from tqdm import tqdm
import sys
import os
import json

sys.path.append('.')
from models.gpt_decoder import TrajectoryGPT, TrajectoryGPTConfig
from trajectory_processor import TrajectoryProcessor

def compute_metrics(model, data_path, num_samples=10, num_test_samples=1000):
    """Compute comprehensive metrics"""

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    all_samples = data['data']
    test_samples = all_samples[-num_test_samples:]

    print(f"Evaluating on {len(test_samples)} test samples...")

    # ✅ FIX: Use config from data file!
    saved_config = data.get('config')
    if saved_config:
        print(f"✓ Using saved config from data file")
        print(f"  Coverage: {saved_config.coverage}m")
        print(f"  Grid size: {saved_config.grid_size}")
        print(f"  Cell size: {saved_config.cell_size}m")
        processor = TrajectoryProcessor(saved_config)
    else:
        print("⚠ No saved config, using default (might be wrong!)")
        from trajectory_processor import TrajectoryConfig
        processor = TrajectoryProcessor(TrajectoryConfig())

    # Metrics storage
    ade_list = []
    fde_list = []
    lane_change_detected = 0
    total_lane_changes = 0
    total_predictions = 0
    valid_predictions = 0
    diversity_scores = []

    model.eval()
    device = next(model.parameters()).device

    for sample in tqdm(test_samples, desc="Evaluating"):
        # Get tokens from sample
        input_tokens = sample.get('input_tokens', sample.get('input', []))
        future_tokens_gt = sample.get('future_tokens', sample.get('target', []))

        # Filter only spatial tokens for GT
        future_tokens_gt = [t for t in future_tokens_gt if t < 256]

        is_lane_change = sample.get('is_lane_change', False)

        if is_lane_change:
            total_lane_changes += 1

        # Generate predictions
        # Use full input sequence (with scene/lane/agent context)
        input_tensor = torch.tensor([input_tokens]).to(device)
        predictions = []

        for i in range(num_samples):
            try:
                with torch.no_grad():
                    generated = model.generate(
                        input_tensor,
                        max_new_tokens=15,  # Enough for 6 future points
                        temperature=1.0,
                        top_p=0.9,
                        end_token=258
                    )

                # Extract generated tokens (after input)
                future_pred = generated[0, len(input_tokens):].tolist()

                # Remove END token if present
                if 258 in future_pred:
                    future_pred = future_pred[:future_pred.index(258)]

                # Filter out special and semantic tokens (keep only spatial)
                future_pred = [t for t in future_pred if t < 256]

                if len(future_pred) > 0:
                    predictions.append(future_pred)
                    total_predictions += 1

            except Exception as e:
                continue

        if len(predictions) == 0:
            continue

        # Reconstruct ground truth
        gt_trajectory = processor.discretizer.reconstruct_trajectory(future_tokens_gt)

        if len(gt_trajectory) == 0:
            continue

        # Compute diversity (average pairwise distance between predictions)
        if len(predictions) > 1:
            pred_trajectories = [processor.discretizer.reconstruct_trajectory(p) for p in predictions]
            pred_trajectories = [p for p in pred_trajectories if len(p) > 0]

            if len(pred_trajectories) > 1:
                pairwise_dists = []
                for i in range(len(pred_trajectories)):
                    for j in range(i+1, len(pred_trajectories)):
                        p1, p2 = pred_trajectories[i], pred_trajectories[j]
                        min_len = min(len(p1), len(p2))
                        if min_len > 0:
                            dist = np.mean(np.linalg.norm(p1[:min_len] - p2[:min_len], axis=1))
                            pairwise_dists.append(dist)

                if pairwise_dists:
                    diversity_scores.append(np.mean(pairwise_dists))

        # Compute best ADE/FDE across all predictions
        best_ade = float('inf')
        best_fde = float('inf')
        detected_lc = False

        for pred_tokens in predictions:
            if len(pred_tokens) == 0:
                continue

            pred_trajectory = processor.discretizer.reconstruct_trajectory(pred_tokens)

            if len(pred_trajectory) == 0:
                continue

            valid_predictions += 1

            # Compute errors
            min_len = min(len(gt_trajectory), len(pred_trajectory))
            if min_len > 0:
                displacement_errors = np.linalg.norm(
                    gt_trajectory[:min_len] - pred_trajectory[:min_len],
                    axis=1
                )
                ade = displacement_errors.mean()
                fde = displacement_errors[-1]

                best_ade = min(best_ade, ade)
                best_fde = min(best_fde, fde)

            # Check lane change detection
            if len(pred_trajectory) > 1:
                lateral_movement = abs(pred_trajectory[-1, 1] - pred_trajectory[0, 1])
                if lateral_movement > 2.0:
                    detected_lc = True

        if best_ade != float('inf'):
            ade_list.append(best_ade)
            fde_list.append(best_fde)

        if is_lane_change and detected_lc:
            lane_change_detected += 1

    # Compute final metrics
    results = {
        'ADE_mean': float(np.mean(ade_list)) if ade_list else float('inf'),
        'ADE_std': float(np.std(ade_list)) if ade_list else 0.0,
        'ADE_median': float(np.median(ade_list)) if ade_list else float('inf'),
        'FDE_mean': float(np.mean(fde_list)) if fde_list else float('inf'),
        'FDE_std': float(np.std(fde_list)) if fde_list else 0.0,
        'FDE_median': float(np.median(fde_list)) if fde_list else float('inf'),
        'Lane_Change_Rate': float(lane_change_detected / total_lane_changes) if total_lane_changes > 0 else 0.0,
        'Total_Lane_Changes': int(total_lane_changes),
        'Detected_Lane_Changes': int(lane_change_detected),
        'Diversity_mean': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        'Diversity_std': float(np.std(diversity_scores)) if diversity_scores else 0.0,
        'Diversity_median': float(np.median(diversity_scores)) if diversity_scores else 0.0,
        'Total_Predictions_Attempted': int(total_predictions),
        'Valid_Predictions': int(valid_predictions),
        'Prediction_Success_Rate': float(valid_predictions / total_predictions) if total_predictions > 0 else 0.0,
        'Num_Test_Samples': int(len(test_samples)),
        'Vocab_Size': int(processor.config.vocab_size)
    }

    return results


def print_results(results, checkpoint_info):
    """Pretty print results"""

    print("\n" + "="*70)
    print("TrajectoryGPT - Evaluation Results (FIXED)")
    print("="*70)

    print("\n📊 Model Info:")
    print(f"  Checkpoint: {checkpoint_info.get('checkpoint_path', 'N/A')}")
    print(f"  Training iterations: {checkpoint_info.get('iteration', 'N/A')}")
    print(f"  Best val loss: {checkpoint_info.get('best_val_loss', 'N/A'):.4f}")
    print(f"  Vocabulary: {results['Vocab_Size']} tokens")

    print("\n📏 Displacement Errors:")
    print(f"  ADE (Average): {results['ADE_mean']:.3f} ± {results['ADE_std']:.3f} m")
    print(f"  ADE (Median):  {results['ADE_median']:.3f} m")
    print(f"  FDE (Final):   {results['FDE_mean']:.3f} ± {results['FDE_std']:.3f} m")
    print(f"  FDE (Median):  {results['FDE_median']:.3f} m")

    print(f"\n🚗 Lane Change Detection:")
    print(f"  Detection Rate: {results['Lane_Change_Rate']*100:.1f}%")
    print(f"  Detected: {results['Detected_Lane_Changes']}/{results['Total_Lane_Changes']} lane changes")

    print(f"\n🌈 Diversity:")
    print(f"  Mean:   {results['Diversity_mean']:.3f} ± {results['Diversity_std']:.3f} m")
    print(f"  Median: {results['Diversity_median']:.3f} m")
    print(f"  (Higher = more diverse predictions)")

    print(f"\n✅ Generation Quality:")
    print(f"  Prediction success rate: {results['Prediction_Success_Rate']*100:.1f}%")
    print(f"  Valid predictions: {results['Valid_Predictions']}/{results['Total_Predictions_Attempted']}")

    print(f"\n📋 Evaluation:")
    print(f"  Test samples: {results['Num_Test_Samples']}")

    # Assessment
    print("\n" + "="*70)
    print("Assessment:")
    print("="*70)

    ade_good = results['ADE_mean'] < 2.0
    lc_good = results['Lane_Change_Rate'] > 0.6
    div_good = results['Diversity_mean'] > 0.5

    if ade_good and lc_good and div_good:
        print("✅ EXCELLENT: Model performing well!")
    elif ade_good or lc_good:
        print("⚠️  GOOD: Model working but needs improvement")
    else:
        print("❌ POOR: Model needs debugging")

    print("="*70)


def main():
    """Main evaluation"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt')
    parser.add_argument('--data', type=str, default='./data/trajectories_full.pkl')
    parser.add_argument('--num-samples', type=int, default=10)
    parser.add_argument('--num-test-samples', type=int, default=1000)
    parser.add_argument('--vocab-size', type=int, default=280,
                        help='Model vocabulary size (280 for 16x16, 1051 for 32x32)')
    parser.add_argument('--output-dir', type=str, default='./outputs')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    config = TrajectoryGPTConfig(vocab_size=args.vocab_size)
    model = TrajectoryGPT(config)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint_info = {
        'checkpoint_path': args.checkpoint,
        'iteration': checkpoint.get('iteration', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
    }

    print(f"✓ Model loaded on {device}")

    # Compute metrics
    print("\nComputing metrics...")
    results = compute_metrics(
        model,
        args.data,
        num_samples=args.num_samples,
        num_test_samples=args.num_test_samples
    )

    # Print results
    print_results(results, checkpoint_info)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, 'evaluation_results_fixed.json')
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint_info': checkpoint_info,
            'metrics': results
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")


if __name__ == "__main__":
    main()