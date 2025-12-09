#!/usr/bin/env python3
"""
TrajectoryGPT Demo - Showcase Best Predictions

Finds scenarios where the model performs well and shows multi-modal predictions.
Automatically searches for diverse scenario types and filters by performance.

Usage:
    python demo_showcase.py \
        --checkpoint ./checkpoints/best_model.pt \
        --data ./data/trajectories_full.pkl \
        --output ./demo_showcase
"""

import os
import argparse
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.gpt_decoder import TrajectoryGPT, TrajectoryGPTConfig
from trajectory_processor import TrajectoryProcessor


# =====================================================================
# Scenario Detection
# =====================================================================

@dataclass
class ScenarioType:
    """Definition of a scenario type to search for"""
    name: str
    description: str
    detector_fn: callable
    color: str


def detect_straight_driving(sample: Dict, thresh: float = 1.5) -> bool:
    """Vehicle continues mostly straight"""
    fut = np.asarray(sample['ego_future'])
    if len(fut) < 3:
        return False

    lateral_change = abs(fut[-1, 1] - fut[0, 1])
    forward_progress = fut[-1, 0] - fut[0, 0]

    return lateral_change < thresh and forward_progress > 2.0


def detect_gentle_turn(sample: Dict, lat_min: float = 1.5, lat_max: float = 4.0) -> bool:
    """Vehicle makes a gentle turn (not aggressive lane change)"""
    fut = np.asarray(sample['ego_future'])
    if len(fut) < 4:
        return False

    lateral_change = abs(fut[-1, 1] - fut[0, 1])
    forward_progress = fut[-1, 0] - fut[0, 0]

    # Check for smooth curve (not sharp angle)
    mid_lateral = abs(fut[len(fut) // 2, 1] - fut[0, 1])
    is_smooth = mid_lateral > 0.3 * lateral_change  # Progressive curve

    return (lat_min < lateral_change < lat_max and
            forward_progress > 2.0 and
            is_smooth)


def detect_lane_change(sample: Dict, thresh: float = 2.5) -> bool:
    """Clear lane change maneuver"""
    fut = np.asarray(sample['ego_future'])
    if len(fut) < 4:
        return False

    lateral_change = abs(fut[-1, 1] - fut[0, 1])
    forward_progress = fut[-1, 0] - fut[0, 0]

    return lateral_change > thresh and forward_progress > 3.0


def detect_deceleration(sample: Dict, min_speed: float = 0.5, drop_factor: float = 0.6) -> bool:
    """Vehicle decelerates significantly"""
    hist = np.asarray(sample['ego_history'])
    fut = np.asarray(sample['ego_future'])

    if len(hist) < 2 or len(fut) < 2:
        return False

    dt = 0.5  # 2 Hz
    v_hist = np.diff(hist[:, 0]) / dt
    v_fut = np.diff(fut[:, 0]) / dt

    if len(v_hist) == 0 or len(v_fut) == 0:
        return False

    v_start = abs(v_hist[-1])
    v_end = abs(v_fut[-1])

    return v_start > min_speed and v_end < drop_factor * v_start


def detect_with_agents(sample: Dict, min_agents: int = 2) -> bool:
    """Scenario has multiple nearby agents"""
    agent_tokens = sample.get('agent_tokens', [])
    return len(agent_tokens) >= min_agents * 3  # Rough estimate (type + marker + positions)


# Define scenario types to search for
SCENARIO_TYPES = [
    ScenarioType(
        name="straight",
        description="Straight driving (simple baseline)",
        detector_fn=detect_straight_driving,
        color="blue"
    ),
    ScenarioType(
        name="gentle_turn",
        description="Gentle turn/curve",
        detector_fn=detect_gentle_turn,
        color="green"
    ),
    ScenarioType(
        name="lane_change",
        description="Lane change maneuver",
        detector_fn=detect_lane_change,
        color="orange"
    ),
    ScenarioType(
        name="deceleration",
        description="Braking/deceleration",
        detector_fn=detect_deceleration,
        color="red"
    ),
    ScenarioType(
        name="multi_agent",
        description="Multiple nearby agents",
        detector_fn=detect_with_agents,
        color="purple"
    ),
]


# =====================================================================
# Model Generation
# =====================================================================

def build_input_tokens(processor: TrajectoryProcessor, sample: Dict) -> List[int]:
    """Build input sequence without future tokens"""
    cfg = processor.config

    scene_context = sample.get('scene_context', [])
    lane_context = sample.get('lane_context', [])
    agent_tokens = sample.get('agent_tokens', [])

    history = np.asarray(sample['ego_history'])
    history_tokens = processor.discretizer.discretize_trajectory(history)

    tokens = (
            [cfg.START_TOKEN] +
            list(scene_context) +
            list(lane_context) +
            list(agent_tokens) +
            history_tokens
    )
    return tokens


def generate_predictions(
        model: TrajectoryGPT,
        processor: TrajectoryProcessor,
        sample: Dict,
        device: torch.device,
        num_modes: int = 10,
        temperature: float = 1.0,
        top_p: float = 0.9,
) -> Tuple[List[np.ndarray], Dict[str, float]]:
    """Generate multiple prediction modes and compute metrics"""

    cfg = processor.config
    num_spatial = cfg.grid_size ** 2

    input_tokens = build_input_tokens(processor, sample)
    gt_future = np.asarray(sample['ego_future'])

    modes = []
    ades = []
    fdes = []

    for _ in range(num_modes):
        idx = torch.tensor(input_tokens, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            generated = model.generate(
                idx,
                max_new_tokens=15,  # Generous limit
                temperature=temperature,
                top_p=top_p,
                end_token=cfg.END_TOKEN,
            )

        new_tokens = generated[0, len(input_tokens):].tolist()

        # Stop at END token
        if cfg.END_TOKEN in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(cfg.END_TOKEN)]

        # Keep only spatial tokens
        spatial_tokens = [t for t in new_tokens if 0 <= t < num_spatial]

        if len(spatial_tokens) == 0:
            continue

        # Reconstruct trajectory
        pred_traj = processor.discretizer.reconstruct_trajectory(spatial_tokens)

        if len(pred_traj) == 0:
            continue

        # Compute errors
        min_len = min(len(pred_traj), len(gt_future))
        if min_len > 0:
            errors = np.linalg.norm(pred_traj[:min_len] - gt_future[:min_len], axis=1)
            ade = float(np.mean(errors))
            fde = float(errors[-1])

            modes.append(pred_traj)
            ades.append(ade)
            fdes.append(fde)

    if not modes:
        return [], {
            'best_ade': float('inf'),
            'best_fde': float('inf'),
            'mean_ade': float('inf'),
            'diversity': 0.0,
            'num_modes': 0,
        }

    # Compute diversity (final point spread)
    finals = np.array([m[-1] for m in modes])
    diversity = float(np.std(np.linalg.norm(finals - finals.mean(axis=0), axis=1)))

    metrics = {
        'best_ade': float(min(ades)),
        'best_fde': float(min(fdes)),
        'mean_ade': float(np.mean(ades)),
        'diversity': diversity,
        'num_modes': len(modes),
    }

    return modes, metrics


# =====================================================================
# Visualization
# =====================================================================

def decode_agents(processor: TrajectoryProcessor, sample: Dict) -> List[Dict]:
    """Decode agent information from tokens"""
    agent_tokens = sample.get('agent_tokens', [])
    agents = []

    i = 0
    while i < len(agent_tokens):
        token = agent_tokens[i]
        cfg = processor.config

        # Check agent type
        agent_type = None
        if token == cfg.AGENT_CAR:
            agent_type = 'CAR'
        elif token == cfg.AGENT_TRUCK:
            agent_type = 'TRUCK'
        elif token == cfg.AGENT_BUS:
            agent_type = 'BUS'

        if agent_type:
            i += 1
            agent_traj = []

            # Skip AGENT_START marker
            if i < len(agent_tokens) and agent_tokens[i] == cfg.NEARBY_AGENT_START:
                i += 1

            # Collect spatial tokens
            num_spatial = cfg.grid_size ** 2
            while i < len(agent_tokens) and agent_tokens[i] < num_spatial:
                x, y = processor.discretizer.reconstruct_point(agent_tokens[i])
                agent_traj.append([x, y])
                i += 1
                if len(agent_traj) >= 2:
                    break

            if agent_traj:
                agents.append({
                    'type': agent_type,
                    'position': agent_traj[-1]
                })
        else:
            i += 1

    return agents


def visualize_scenario(
        sample: Dict,
        modes: List[np.ndarray],
        metrics: Dict,
        processor: TrajectoryProcessor,
        scenario_type: ScenarioType,
        output_path: str,
):
    """Create visualization for a scenario"""

    fig, ax = plt.subplots(figsize=(10, 10))

    history = np.asarray(sample['ego_history'])
    gt_future = np.asarray(sample['ego_future'])
    agents = decode_agents(processor, sample)

    # Plot predictions (semi-transparent, colorful)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(modes)))
    for i, (mode, color) in enumerate(zip(modes, colors)):
        ax.plot(mode[:, 0], mode[:, 1],
                color=color, alpha=0.6, linewidth=2.5,
                label=f'Mode {i + 1}' if i < 3 else '')

    # Plot ground truth (thick green)
    ax.plot(gt_future[:, 0], gt_future[:, 1],
            'g-', linewidth=4, label='Ground Truth', zorder=100)
    ax.scatter(gt_future[-1, 0], gt_future[-1, 1],
               s=200, c='green', marker='*', zorder=101,
               edgecolors='black', linewidth=2)

    # Plot history (dark gray dashed)
    ax.plot(history[:, 0], history[:, 1],
            color='#333', linewidth=3.5, linestyle='--',
            label='History', zorder=99)
    ax.scatter(history[0, 0], history[0, 1],
               s=150, c='black', marker='o', zorder=100)

    # Plot agents
    for agent in agents:
        pos = agent['position']
        if agent['type'] == 'TRUCK':
            width, height = 2.5, 8.0
            color = 'orange'
        elif agent['type'] == 'BUS':
            width, height = 2.5, 10.0
            color = 'purple'
        else:
            width, height = 2.0, 4.5
            color = 'red'

        rect = patches.Rectangle(
            (pos[0] - width / 2, pos[1] - height / 2),
            width, height,
            linewidth=2,
            edgecolor='black',
            facecolor=color,
            alpha=0.7,
            zorder=95
        )
        ax.add_patch(rect)

        ax.text(pos[0], pos[1] + height / 2 + 1.5,
                agent['type'],
                ha='center', va='bottom',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', alpha=0.8))

    # Decode scene/lane context
    scene_tokens = sample.get('scene_context', [])
    lane_tokens = sample.get('lane_context', [])

    scene_names = []
    lane_names = []

    if hasattr(processor.config, 'semantic_token_map'):
        for t in scene_tokens:
            name = processor.config.semantic_token_map.get(t)
            if name:
                scene_names.append(name)
        for t in lane_tokens:
            name = processor.config.semantic_token_map.get(t)
            if name:
                lane_names.append(name)

    scene_str = ', '.join(scene_names) if scene_names else 'UNKNOWN'
    lane_str = ', '.join(lane_names) if lane_names else 'UNKNOWN'

    # Title
    title = f"{scenario_type.description}\n"
    title += f"Scene: {scene_str} | Lane: {lane_str} | Agents: {len(agents)}\n"
    title += f"Best ADE: {metrics['best_ade']:.2f}m | FDE: {metrics['best_fde']:.2f}m | "
    title += f"Diversity: {metrics['diversity']:.2f}m"

    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.set_xlabel('X (meters)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y (meters)', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', fontsize=9, framealpha=0.9)

    # Set limits
    all_points = [history, gt_future] + modes
    all_points = [p for p in all_points if len(p) > 0]
    if all_points:
        all_points = np.vstack(all_points)
        margin = 5
        ax.set_xlim(all_points[:, 0].min() - margin,
                    all_points[:, 0].max() + margin)
        ax.set_ylim(all_points[:, 1].min() - margin,
                    all_points[:, 1].max() + margin)

    ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =====================================================================
# Scenario Search
# =====================================================================

def search_for_scenarios(
        model: TrajectoryGPT,
        processor: TrajectoryProcessor,
        samples: List[Dict],
        device: torch.device,
        max_ade: float = 4.0,
        min_diversity: float = 1.0,
        num_per_type: int = 2,
        temperature: float = 0.8,
) -> Dict[str, List[Tuple[int, Dict, List[np.ndarray], Dict]]]:
    """Search for good examples of each scenario type"""

    results = {st.name: [] for st in SCENARIO_TYPES}

    print("\nSearching for showcase scenarios...")
    print(f"Criteria: Best ADE < {max_ade}m, Diversity > {min_diversity}m")
    print("=" * 70)

    for scenario_type in SCENARIO_TYPES:
        print(f"\nSearching for: {scenario_type.description}")
        found = 0

        for idx, sample in enumerate(samples):
            # Check if sample matches scenario type
            if not scenario_type.detector_fn(sample):
                continue

            # Generate predictions
            modes, metrics = generate_predictions(
                model, processor, sample, device,
                num_modes=10,
                temperature=temperature,
                top_p=0.9
            )

            # Check if passes quality criteria
            if (metrics['best_ade'] <= max_ade and
                    metrics['diversity'] >= min_diversity and
                    metrics['num_modes'] >= 5):

                results[scenario_type.name].append((idx, sample, modes, metrics))
                found += 1

                print(f"  ✓ Found #{found}: Sample {idx} - "
                      f"ADE={metrics['best_ade']:.2f}m, "
                      f"Div={metrics['diversity']:.2f}m, "
                      f"Modes={metrics['num_modes']}")

                if found >= num_per_type:
                    break

        if found == 0:
            print(f"  ⚠ No good examples found")
        elif found < num_per_type:
            print(f"  ⚠ Only found {found}/{num_per_type} examples")

    return results


# =====================================================================
# Main
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description='TrajectoryGPT Demo - Showcase Best Predictions'
    )
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to trajectory data')
    parser.add_argument('--max-ade', type=float, default=4.0,
                        help='Maximum ADE for demo examples')
    parser.add_argument('--min-diversity', type=float, default=1.0,
                        help='Minimum diversity for demo examples')
    parser.add_argument('--num-per-type', type=int, default=2,
                        help='Number of examples per scenario type')
    parser.add_argument('--temperature', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--output', type=str, default='./demo_showcase',
                        help='Output directory')

    args = parser.parse_args()

    print("=" * 70)
    print("TrajectoryGPT Showcase Demo")
    print("=" * 70)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load data
    print(f"\nLoading data from {args.data}...")
    with open(args.data, 'rb') as f:
        data = pickle.load(f)

    samples = data['data']
    saved_cfg = data['config']

    print(f"✓ Loaded {len(samples)} samples")
    print(f"  Grid: {saved_cfg.grid_size}×{saved_cfg.grid_size}")
    print(f"  Vocab: {saved_cfg.vocab_size}")

    processor = TrajectoryProcessor(saved_cfg)

    # Load model
    print(f"\nLoading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Create model with correct config
    model_cfg = TrajectoryGPTConfig(
        vocab_size=saved_cfg.vocab_size,
        n_layer=6,
        n_embd=384,
        n_head=6
    )
    model = TrajectoryGPT(model_cfg)

    # Load weights
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model on {device}")
    print(f"  Parameters: {n_params / 1e6:.2f}M")

    # Search for good scenarios
    results = search_for_scenarios(
        model, processor, samples, device,
        max_ade=args.max_ade,
        min_diversity=args.min_diversity,
        num_per_type=args.num_per_type,
        temperature=args.temperature
    )

    # Check if any results found
    total_found = sum(len(v) for v in results.values())
    if total_found == 0:
        print("\n⚠ No scenarios found matching criteria.")
        print("Try loosening constraints (--max-ade 6.0 --min-diversity 0.5)")
        return

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Visualize results
    print("\n" + "=" * 70)
    print("Generating Visualizations")
    print("=" * 70)

    for scenario_type in SCENARIO_TYPES:
        scenario_results = results[scenario_type.name]

        if not scenario_results:
            continue

        print(f"\n{scenario_type.description}:")

        for i, (idx, sample, modes, metrics) in enumerate(scenario_results, 1):
            output_path = os.path.join(
                args.output,
                f"{scenario_type.name}_{i}.png"
            )

            visualize_scenario(
                sample, modes, metrics, processor,
                scenario_type, output_path
            )

            print(f"  ✓ Sample {idx}: ADE={metrics['best_ade']:.2f}m, "
                  f"Div={metrics['diversity']:.2f}m → {output_path}")

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print(f"\n✓ {total_found} visualizations saved to: {args.output}")
    print("\nScenarios showcased:")
    for scenario_type in SCENARIO_TYPES:
        count = len(results[scenario_type.name])
        if count > 0:
            print(f"  - {scenario_type.description}: {count} examples")


if __name__ == '__main__':
    main()