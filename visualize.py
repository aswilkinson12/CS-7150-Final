"""
Visualize TrajectoryGPT Predictions
Run after training to see model predictions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append('.')
from models.gpt_decoder import TrajectoryGPT, TrajectoryGPTConfig
from trajectory_processor import TrajectoryProcessor, TrajectoryConfig

def load_model(checkpoint_path='checkpoints/best_model.pt'):
    """Load trained model"""
    print(f"Loading model from {checkpoint_path}...")

    config = TrajectoryGPTConfig()
    model = TrajectoryGPT(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"✓ Model loaded (iteration {checkpoint['iteration']})")
    return model

def generate_predictions(model, history_tokens, num_samples=10, temperature=1.0, top_p=0.9):
    """Generate diverse predictions"""
    predictions = []

    history_tensor = torch.tensor([history_tokens])

    print(f"  Generating {num_samples} predictions...")

    for i in range(num_samples):
        with torch.no_grad():
            generated = model.generate(
                history_tensor,
                max_new_tokens=10,  # Increased from 6
                temperature=temperature,
                top_p=top_p,
                end_token=258
            )

        # Extract future tokens (skip START + history)
        future_tokens = generated[0, len(history_tokens):].tolist()

        # Remove END token if present
        if 258 in future_tokens:
            future_tokens = future_tokens[:future_tokens.index(258)]

        # Filter out special tokens (only keep spatial tokens 0-255)
        future_tokens = [t for t in future_tokens if t < 256]

        if len(future_tokens) > 0:
            predictions.append(future_tokens)
            print(f"    Prediction {i+1}: {len(future_tokens)} tokens")
        else:
            print(f"    Prediction {i+1}: No valid tokens generated")

    print(f"  Total valid predictions: {len(predictions)}/{num_samples}")
    return predictions

def visualize_predictions(
    history_continuous,
    future_continuous,
    predictions_tokens,
    processor,
    save_path='prediction_visualization.png'
):
    """Create beautiful visualization"""

    fig, ax = plt.subplots(figsize=(12, 10))

    # Draw grid
    config = processor.config
    grid_min = -config.coverage / 2
    grid_max = config.coverage / 2

    for i in range(config.grid_size + 1):
        pos = grid_min + i * config.cell_size
        ax.axhline(y=pos, color='gray', linewidth=0.5, alpha=0.2)
        ax.axvline(x=pos, color='gray', linewidth=0.5, alpha=0.2)

    # Draw axes
    ax.axhline(y=0, color='black', linewidth=1.5, alpha=0.3)
    ax.axvline(x=0, color='black', linewidth=1.5, alpha=0.3)

    # Agent at origin
    ax.plot(0, 0, 'r*', markersize=20, label='Ego Vehicle', zorder=10)
    ax.arrow(0, 0, 3, 0, head_width=0.8, head_length=0.8,
             fc='red', ec='red', alpha=0.6, linewidth=2, zorder=10)

    # Plot history (observed trajectory)
    ax.plot(history_continuous[:, 0], history_continuous[:, 1],
           'o-', color='blue', linewidth=3, markersize=8,
           label='History (Observed)', alpha=0.8, zorder=5)

    # Plot ground truth future
    ax.plot(future_continuous[:, 0], future_continuous[:, 1],
           'o-', color='green', linewidth=3, markersize=8,
           label='Ground Truth Future', alpha=0.8, zorder=5)

    # Plot predictions
    colors = plt.cm.rainbow(np.linspace(0, 1, max(len(predictions_tokens), 1)))

    predictions_plotted = 0
    for i, (tokens, color) in enumerate(zip(predictions_tokens, colors)):
        if len(tokens) > 0:
            pred_trajectory = processor.discretizer.reconstruct_trajectory(tokens)

            if len(pred_trajectory) > 0:
                ax.plot(pred_trajectory[:, 0], pred_trajectory[:, 1],
                       '--', linewidth=2, alpha=0.7, color=color,
                       label=f'Prediction {i+1}' if i < 3 else None,
                       zorder=3)
                predictions_plotted += 1

    if predictions_plotted == 0:
        # Add text warning if no predictions
        ax.text(0.5, 0.95, 'WARNING: No predictions generated!',
               transform=ax.transAxes, ha='center', va='top',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
               fontsize=12, fontweight='bold')

    ax.set_xlim(grid_min, grid_max)
    ax.set_ylim(grid_min, grid_max)
    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)', fontsize=14)
    ax.set_ylabel('Y (meters)', fontsize=14)
    ax.set_title('TrajectoryGPT: Multi-Modal Predictions', fontsize=16, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"✓ Saved visualization to {save_path}")

    return fig

def main():
    """Main visualization function"""

    # Check if checkpoint exists
    if not os.path.exists('checkpoints/best_model.pt'):
        print("Error: No trained model found at checkpoints/best_model.pt")
        print("Please train the model first: python training/train.py --epochs 20")
        return

    # Load trained model
    model = load_model('checkpoints/best_model.pt')

    # Load test data
    print("\nLoading test data...")
    if not os.path.exists('data/trajectories_mini.pkl'):
        print("Error: No data found at data/trajectories_mini.pkl")
        print("Please extract data first: python nuscenes_extractor.py")
        return

    with open('data/trajectories_mini.pkl', 'rb') as f:
        data = pickle.load(f)

    # Get test samples
    test_samples = data['data'][-10:]  # Last 10 samples

    processor = TrajectoryProcessor(TrajectoryConfig())

    # Create output directory
    os.makedirs('outputs', exist_ok=True)

    # Visualize several examples
    for idx, sample in enumerate(test_samples[:5]):  # First 5
        print(f"\nProcessing sample {idx+1}/5...")

        # Get history tokens
        history_tokens = sample['history_tokens']
        future_tokens_gt = sample['future_tokens']

        print(f"  History: {len(history_tokens)} tokens")
        print(f"  Ground truth future: {len(future_tokens_gt)} tokens")

        # Generate predictions
        predictions = generate_predictions(
            model,
            [257] + history_tokens,  # Add START token
            num_samples=10,
            temperature=1.0,
            top_p=0.9
        )

        if len(predictions) == 0:
            print("  ⚠️  Warning: No predictions generated for this sample!")
            print("  Skipping visualization...")
            continue

        # Reconstruct trajectories
        history_continuous = processor.discretizer.reconstruct_trajectory(history_tokens)
        future_continuous = processor.discretizer.reconstruct_trajectory(future_tokens_gt)

        print(f"  History trajectory: {history_continuous.shape}")
        print(f"  Future trajectory: {future_continuous.shape}")

        # Visualize
        save_path = f'outputs/prediction_sample_{idx+1}.png'
        visualize_predictions(
            history_continuous,
            future_continuous,
            predictions,
            processor,
            save_path=save_path
        )

        print(f"   Saved to {save_path}")

        # Print diversity stats
        unique_predictions = len(set(tuple(p) for p in predictions if len(p) > 0))
        print(f"  Diversity: {unique_predictions}/{len(predictions)} unique predictions")

    print("\n" + "="*60)
    print("✓ Visualization complete!")
    print("="*60)
    print("\nGenerated files in outputs/:")
    for i in range(1, 6):
        print(f"  - prediction_sample_{i}.png")
    print("\nThese show:")
    print("  • Blue line: History (what model sees)")
    print("  • Green line: Ground truth future")
    print("  • Rainbow dashed lines: 10 diverse predictions")
    print("  • Red star: Ego vehicle")
    print("="*60)

if __name__ == "__main__":
    main()