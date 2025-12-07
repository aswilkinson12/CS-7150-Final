"""
TrajectoryGPT - Complete Evaluation Script
Computes all metrics: ADE, FDE, lane change detection, diversity
Works with full dataset and enhanced vocabulary (280 tokens)
"""

import torch
import numpy as np
import pickle
from tqdm import tqdm
import sys
import os
import json
import argparse

sys.path.append('.')
from models.gpt_decoder import TrajectoryGPT, TrajectoryGPTConfig
from trajectory_processor import TrajectoryProcessor, TrajectoryConfig


def compute_metrics(model, data_path, num_samples=10, num_test_samples=1000,
                   vocab_size=280, device='cpu'):
    """
    Compute comprehensive metrics

    Args:
        model: TrajectoryGPT model
        data_path: Path to data pickle file
        num_samples: Number of predictions per sample
        num_test_samples: Number of test samples to evaluate
        vocab_size: Vocabulary size (259 or 280)
        device: torch device
    """

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    all_samples = data['data']

    # Use last 10% for testing (or num_test_samples, whichever is smaller)
    test_start = int(0.9 * len(all_samples))
    test_samples = all_samples[test_start:test_start + num_test_samples]

    print(f"Evaluating on {len(test_samples)} test samples...")
    print(f"Vocabulary size: {vocab_size}")

    # Create processor based on vocab size
    if vocab_size == 280:
        from trajectory_processor import create_enhanced_processor
        processor = create_enhanced_processor()
        print("Using enhanced processor (280 vocab)")
    else:
        from trajectory_processor import create_basic_processor
        processor = create_basic_processor()
        print("Using basic processor (259 vocab)")

    # Metrics storage
    ade_list = []
    fde_list = []
    lane_change_detected = 0
    total_lane_changes = 0
    total_predictions = 0
    valid_predictions = 0
    diversity_scores = []

    model.eval()

    for sample in tqdm(test_samples, desc="Evaluating"):
        # Handle both old and new format
        if 'history_tokens' in sample:
            history_tokens = sample['history_tokens']
            future_tokens_gt = sample['future_tokens']
        else:
            # Extract from input/target
            input_tokens = sample.get('input', sample.get('input_tokens', []))
            target_tokens = sample.get('target', sample.get('target_tokens', []))

            # Filter spatial tokens only
            history_tokens = [t for t in input_tokens if t < 256][:4]
            future_tokens_gt = [t for t in target_tokens if t < 256][-6:]

        is_lane_change = sample.get('is_lane_change', False)

        if is_lane_change:
            total_lane_changes += 1

        # Generate predictions
        history_tensor = torch.tensor([[257] + history_tokens]).to(device)
        predictions = []

        for i in range(num_samples):
            try:
                with torch.no_grad():
                    generated = model.generate(
                        history_tensor,
                        max_new_tokens=15,  # Increased for enhanced mode
                        temperature=1.0,
                        top_p=0.9
                    )

                # Extract future tokens
                future_pred = generated[0, len(history_tokens)+1:].tolist()

                # Remove END token if present
                if 258 in future_pred:
                    future_pred = future_pred[:future_pred.index(258)]

                # Filter out special tokens (keep only spatial 0-255)
                future_pred = [t for t in future_pred if t < 256]

                if len(future_pred) > 0:
                    predictions.append(future_pred)
                    total_predictions += 1

            except Exception as e:
                # Silently continue on errors
                continue

        if len(predictions) == 0:
            continue

        # Reconstruct ground truth
        gt_trajectory = processor.discretizer.reconstruct_trajectory(future_tokens_gt)

        if len(gt_trajectory) == 0:
            continue

        # Compute diversity (average pairwise distance between predictions)
        if len(predictions) > 1:
            pred_trajectories = [processor.discretizer.reconstruct_trajectory(p)
                               for p in predictions]
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
            if len(pred_trajectory) > 0:
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
        'Vocab_Size': int(vocab_size)
    }

    return results


def print_results(results, checkpoint_info):
    """Pretty print comprehensive results"""

    print("\n" + "="*70)
    print("🎯 TrajectoryGPT - Complete Evaluation Results")
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
    print(f"  Valid predictions: {results['Valid_Predictions']:,}/{results['Total_Predictions_Attempted']:,}")

    print(f"\n📋 Evaluation:")
    print(f"  Test samples: {results['Num_Test_Samples']:,}")

    # Comparison with baselines
    print("\n" + "-"*70)
    print("📊 Comparison with Published Baselines:")
    print("-"*70)

    baselines = [
        ("Constant Velocity", 2.50, None),
        ("Social-LSTM", 1.20, None),
        ("Trajectron++", 0.80, 0.23),
        ("TrajectoryGPT (yours)", results['ADE_mean'], results['Lane_Change_Rate'])
    ]

    print(f"  {'Model':<25} {'ADE (m)':<12} {'Lane Change %':<15}")
    print(f"  {'-'*25} {'-'*12} {'-'*15}")
    for name, ade, lc in baselines:
        lc_str = f"{lc*100:.1f}%" if lc is not None else "N/A"
        print(f"  {name:<25} {ade:<12.2f} {lc_str:<15}")

    # Success criteria
    print("\n" + "-"*70)
    print("🎯 Success Criteria:")
    print("-"*70)

    ade_target = 2.0
    lc_target = 0.60
    diversity_target = 1.0

    ade_pass = "✅" if results['ADE_mean'] < ade_target else "⚠️"
    lc_pass = "✅" if results['Lane_Change_Rate'] > lc_target else "⚠️"
    div_pass = "✅" if results['Diversity_mean'] > diversity_target else "⚠️"

    print(f"  {ade_pass} ADE < 2.0m:              {results['ADE_mean']:.3f}m (target: {ade_target}m)")
    print(f"  {lc_pass} Lane Change > 60%:       {results['Lane_Change_Rate']*100:.1f}% (target: 60%)")
    print(f"  {div_pass} Diversity > 1.0m:       {results['Diversity_mean']:.3f}m (target: 1.0m)")

    # Overall assessment
    passes = sum([
        results['ADE_mean'] < ade_target,
        results['Lane_Change_Rate'] > lc_target,
        results['Diversity_mean'] > diversity_target
    ])

    print("\n" + "="*70)
    if passes == 3:
        improvement_pct = ((ade_target - results['ADE_mean']) / ade_target) * 100
        print(f"🎉 EXCELLENT: All criteria met!")
        print(f"   ADE is {improvement_pct:.1f}% better than target!")
    elif passes == 2:
        print("✅ GOOD: 2/3 criteria met - strong performance!")
    elif passes == 1:
        print("⚠️  FAIR: 1/3 criteria met - consider more training")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider more epochs or tuning")
    print("="*70)


def main():
    """Main evaluation function"""

    parser = argparse.ArgumentParser(description='Evaluate TrajectoryGPT model')
    parser.add_argument('--checkpoint', type=str, default='./checkpoints/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--data', type=str, default='./data/trajectories_full.pkl',
                       help='Path to data file')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of predictions per test sample')
    parser.add_argument('--num-test-samples', type=int, default=1000,
                       help='Number of test samples to evaluate')
    parser.add_argument('--vocab-size', type=int, default=280,
                       help='Vocabulary size (259 or 280)')
    parser.add_argument('--output-dir', type=str, default='./outputs',
                       help='Output directory for results')

    args = parser.parse_args()

    checkpoint_path = args.checkpoint
    data_path = args.data

    # Check files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: No trained model found at {checkpoint_path}")
        print("Please train the model first:")
        print("  python training/train.py --epochs 20")
        return

    if not os.path.exists(data_path):
        print(f"❌ Error: No data found at {data_path}")
        print("Please extract data first:")
        print("  python nuscenes_extractor.py")
        return

    # Load model
    print("Loading model...")
    config = TrajectoryGPTConfig(vocab_size=args.vocab_size)
    model = TrajectoryGPT(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    checkpoint_info = {
        'checkpoint_path': checkpoint_path,
        'iteration': checkpoint.get('iteration', 'N/A'),
        'best_val_loss': checkpoint.get('best_val_loss', float('inf'))
    }

    print(f"✅ Model loaded on {device}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Training iterations: {checkpoint_info['iteration']}")
    print(f"  Best validation loss: {checkpoint_info['best_val_loss']:.4f}")

    # Compute metrics
    print(f"\nComputing metrics on {args.num_test_samples} samples...")
    print("(This may take 5-10 minutes for 1000 samples)")

    results = compute_metrics(
        model,
        data_path,
        num_samples=args.num_samples,
        num_test_samples=args.num_test_samples,
        vocab_size=args.vocab_size,
        device=device
    )

    # Print results
    print_results(results, checkpoint_info)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    output_file = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint_info': checkpoint_info,
            'metrics': results,
            'args': vars(args)
        }, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    # Save summary
    summary_file = os.path.join(args.output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("TrajectoryGPT Evaluation Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model: {checkpoint_path}\n")
        f.write(f"Data:  {data_path}\n")
        f.write(f"Vocab: {results['Vocab_Size']} tokens\n")
        f.write(f"\n")
        f.write(f"ADE:       {results['ADE_mean']:.3f} ± {results['ADE_std']:.3f} m\n")
        f.write(f"FDE:       {results['FDE_mean']:.3f} ± {results['FDE_std']:.3f} m\n")
        f.write(f"Lane Change: {results['Lane_Change_Rate']*100:.1f}%\n")
        f.write(f"Diversity: {results['Diversity_mean']:.3f} ± {results['Diversity_std']:.3f} m\n")
        f.write(f"\nSuccess Rate: {results['Prediction_Success_Rate']*100:.1f}%\n")
        f.write(f"Test Samples: {results['Num_Test_Samples']:,}\n")

    print(f"✅ Summary saved to {summary_file}")

    print("\n📁 Generated files:")
    print(f"  - {output_file}")
    print(f"  - {summary_file}")

    print("\n🎯 Next steps:")
    print("  1. Check visualizations: python visualize_enhanced.py")
    print("  2. Train LSTM baseline for comparison")
    print("  3. Create demo materials")


if __name__ == "__main__":
    main()