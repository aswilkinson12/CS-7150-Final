"""
TrajectoryGPT - Complete Evaluation Script
Computes all metrics: ADE, FDE, lane change detection, diversity
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
from trajectory_processor import TrajectoryProcessor, TrajectoryConfig

def compute_metrics(model, data_path, num_samples=10, num_test_samples=100):
    """Compute comprehensive metrics"""

    # Load data
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    all_samples = data['data']
    test_samples = all_samples[-num_test_samples:]

    print(f"Evaluating on {len(test_samples)} test samples...")

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
        history_tokens = sample['history_tokens']
        future_tokens_gt = sample['future_tokens']
        is_lane_change = sample['is_lane_change']

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
                        max_new_tokens=10,  # Increased from 6
                        temperature=1.0,
                        top_p=0.9,
                        end_token=258
                    )

                # Extract future tokens
                future_pred = generated[0, len(history_tokens)+1:].tolist()

                # Remove END token if present
                if 258 in future_pred:
                    future_pred = future_pred[:future_pred.index(258)]

                # Filter out special tokens
                future_pred = [t for t in future_pred if t < 256]

                if len(future_pred) > 0:
                    predictions.append(future_pred)
                    total_predictions += 1

            except Exception as e:
                print(f"Warning: Generation failed for sample: {e}")
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
        'FDE_mean': float(np.mean(fde_list)) if fde_list else float('inf'),
        'FDE_std': float(np.std(fde_list)) if fde_list else 0.0,
        'Lane_Change_Rate': float(lane_change_detected / total_lane_changes) if total_lane_changes > 0 else 0.0,
        'Total_Lane_Changes': int(total_lane_changes),
        'Detected_Lane_Changes': int(lane_change_detected),
        'Diversity_mean': float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        'Diversity_std': float(np.std(diversity_scores)) if diversity_scores else 0.0,
        'Total_Predictions_Attempted': int(total_predictions),
        'Valid_Predictions': int(valid_predictions),
        'Prediction_Success_Rate': float(valid_predictions / total_predictions) if total_predictions > 0 else 0.0,
        'Num_Test_Samples': int(len(test_samples))
    }

    return results

def print_results(results, checkpoint_info):
    """Pretty print comprehensive results"""

    print("\n" + "="*70)
    print("TrajectoryGPT - Complete Evaluation Results")
    print("="*70)

    print("\n📊 Model Info:")
    print(f"  Checkpoint: {checkpoint_info.get('checkpoint_path', 'N/A')}")
    print(f"  Training iterations: {checkpoint_info.get('iteration', 'N/A')}")
    print(f"  Best val loss: {checkpoint_info.get('best_val_loss', 'N/A'):.4f}")

    print("\n📏 Displacement Errors:")
    print(f"  ADE (Average): {results['ADE_mean']:.3f} ± {results['ADE_std']:.3f} meters")
    print(f"  FDE (Final):   {results['FDE_mean']:.3f} ± {results['FDE_std']:.3f} meters")

    print(f"\n🚗 Lane Change Detection:")
    print(f"  Detection Rate: {results['Lane_Change_Rate']*100:.1f}%")
    print(f"  Detected: {results['Detected_Lane_Changes']}/{results['Total_Lane_Changes']} lane changes")

    print(f"\n🌈 Diversity:")
    print(f"  Avg pairwise distance: {results['Diversity_mean']:.3f} ± {results['Diversity_std']:.3f} meters")
    print(f"  (Higher = more diverse predictions)")

    print(f"\n Generation Quality:")
    print(f"  Prediction success rate: {results['Prediction_Success_Rate']*100:.1f}%")
    print(f"  Valid predictions: {results['Valid_Predictions']}/{results['Total_Predictions_Attempted']}")

    print(f"\n📋 Evaluation:")
    print(f"  Test samples: {results['Num_Test_Samples']}")

    # Comparison with baseline
    print("\n" + "-"*70)
    print("📊 Comparison with Baseline:")
    print("-"*70)

    baseline_lc_rate = 0.23  # Trajectron++
    baseline_ade = 2.5

    print(f"  Metric                    | Baseline  | TrajectoryGPT | Result")
    print(f"  --------------------------|-----------|---------------|--------")
    print(f"  Lane Change Detection     | 23.0%     | {results['Lane_Change_Rate']*100:5.1f}%      | ", end="")

    if results['Lane_Change_Rate'] > baseline_lc_rate:
        improvement = (results['Lane_Change_Rate'] - baseline_lc_rate) / baseline_lc_rate * 100
        print(f" +{improvement:.1f}%")
    else:
        print(f"⚠️  Below baseline")

    print(f"  ADE (meters)              | 2.50m     | {results['ADE_mean']:5.2f}m      | ", end="")

    if results['ADE_mean'] < baseline_ade:
        print(f" Better")
    else:
        print(f"⚠️  Higher")

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
        print("🎉 EXCELLENT: All criteria met!")
    elif passes == 2:
        print("✅ GOOD: 2/3 criteria met")
    elif passes == 1:
        print("⚠️  FAIR: 1/3 criteria met - consider training more epochs")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Consider training more epochs or tuning hyperparameters")
    print("="*70)

def main():
    """Main evaluation function"""

    checkpoint_path = 'checkpoints/best_model.pt'
    data_path = 'data/trajectories_mini.pkl'

    # Check files exist
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: No trained model found at {checkpoint_path}")
        print("Please train the model first: python training/train.py --epochs 20")
        return

    if not os.path.exists(data_path):
        print(f"❌ Error: No data found at {data_path}")
        print("Please extract data first: python nuscenes_extractor.py")
        return

    # Load model
    print("Loading model...")
    config = TrajectoryGPTConfig()
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
    print(f"  Training iterations: {checkpoint_info['iteration']}")
    print(f"  Best validation loss: {checkpoint_info['best_val_loss']:.4f}")

    # Compute metrics
    print("\nComputing metrics (this may take a few minutes)...")
    results = compute_metrics(
        model,
        data_path,
        num_samples=10,
        num_test_samples=100
    )

    # Print results
    print_results(results, checkpoint_info)

    # Save results
    os.makedirs('outputs', exist_ok=True)

    output_file = 'outputs/evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump({
            'checkpoint_info': checkpoint_info,
            'metrics': results
        }, f, indent=2)

    print(f"\n✅ Results saved to {output_file}")

    # Save summary
    summary_file = 'outputs/evaluation_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("TrajectoryGPT Evaluation Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"ADE: {results['ADE_mean']:.3f} ± {results['ADE_std']:.3f} meters\n")
        f.write(f"FDE: {results['FDE_mean']:.3f} ± {results['FDE_std']:.3f} meters\n")
        f.write(f"Lane Change Detection: {results['Lane_Change_Rate']*100:.1f}%\n")
        f.write(f"Diversity: {results['Diversity_mean']:.3f} ± {results['Diversity_std']:.3f} meters\n")
        f.write(f"\nPrediction Success Rate: {results['Prediction_Success_Rate']*100:.1f}%\n")

    print(f"✅ Summary saved to {summary_file}")

    print("\n📁 Generated files:")
    print("  - outputs/evaluation_results.json (detailed metrics)")
    print("  - outputs/evaluation_summary.txt (quick summary)")

if __name__ == "__main__":
    main()