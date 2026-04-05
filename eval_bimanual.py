"""
Evaluate a trained bimanual Instant Policy model on PerAct2 RLBench tasks.

Usage examples:

  # Evaluate on a single task (100 rollouts, 2 demos)
  python eval_bimanual.py \
      --checkpoint ./checkpoints_bimanual/bimanual_final.pt \
      --tasks bimanual_push_box \
      --num_rollouts 100

  # Evaluate on all PerAct2 bimanual tasks
  python eval_bimanual.py \
      --checkpoint ./checkpoints_bimanual/bimanual_final.pt \
      --num_rollouts 100

  # Quick test (fewer rollouts)
  python eval_bimanual.py \
      --checkpoint ./checkpoints_bimanual/bimanual_final.pt \
      --tasks bimanual_push_box bimanual_lift_tray \
      --num_rollouts 10

Evaluation protocol (aligned with Instant Policy paper, Section 4.1):
  - N=2 demonstrations provided as in-context examples.
  - Object poses randomised at each rollout via task.reset().
  - Closed-loop execution with T=8 action horizon.
  - Success determined by RLBench's built-in task conditions.
  - Metric: task success rate = num_successes / num_rollouts.
"""
import argparse
import json
import os
import torch

from ip.bimanual.config import BimanualIPConfig
from ip.bimanual.model import BimanualGraphDiffusionPolicy
from bimanual_sim_utils import (
    rollout_bimanual_model,
    evaluate_all_tasks,
    print_results_table,
    BIMANUAL_TASK_MAP,
)


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load a trained bimanual model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device)

    if 'cfg' in ckpt and ckpt['cfg'] is not None:
        cfg = ckpt['cfg']
    else:
        cfg = BimanualIPConfig()

    model = BimanualGraphDiffusionPolicy(cfg).to(device)

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate bimanual Instant Policy on PerAct2 tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained bimanual model checkpoint (.pt)',
    )
    parser.add_argument(
        '--tasks', type=str, nargs='*', default=None,
        help='Task names to evaluate (default: all 13 PerAct2 tasks)',
    )
    parser.add_argument(
        '--num_demos', type=int, default=2,
        help='Number of demonstrations as context (default: 2)',
    )
    parser.add_argument(
        '--num_rollouts', type=int, default=100,
        help='Rollouts per task (default: 100, paper uses 100)',
    )
    parser.add_argument(
        '--max_steps', type=int, default=30,
        help='Max observe-predict-execute cycles per rollout (default: 30)',
    )
    parser.add_argument(
        '--execution_horizon', type=int, default=8,
        help='Actions executed per cycle (default: 8)',
    )
    parser.add_argument(
        '--num_traj_wp', type=int, default=10,
        help='Demo waypoints after downsampling (default: 10)',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Torch device (default: cuda)',
    )
    parser.add_argument(
        '--headless', action='store_true', default=True,
        help='Run simulation without GUI (default: True)',
    )
    parser.add_argument(
        '--no_headless', action='store_true',
        help='Run simulation WITH GUI (for debugging)',
    )
    parser.add_argument(
        '--save_results', type=str, default=None,
        help='Path to save results as JSON (optional)',
    )
    args = parser.parse_args()

    headless = not args.no_headless
    if not torch.cuda.is_available() and args.device == 'cuda':
        print("CUDA not available, using CPU.")
        args.device = 'cpu'

    # Load model
    model = load_model(args.checkpoint, args.device)

    # Determine tasks
    task_names = args.tasks
    if task_names is None:
        task_names = list(BIMANUAL_TASK_MAP.keys())
        print(f"\nEvaluating all {len(task_names)} PerAct2 bimanual tasks.")
    else:
        print(f"\nEvaluating {len(task_names)} task(s): {task_names}")

    print(f"  Demos per task (N):     {args.num_demos}")
    print(f"  Rollouts per task:      {args.num_rollouts}")
    print(f"  Execution horizon (T):  {args.execution_horizon}")
    print(f"  Demo waypoints (L):     {args.num_traj_wp}")
    print(f"  Device:                 {args.device}")

    # Run evaluation
    if len(task_names) == 1:
        # Single task
        result = rollout_bimanual_model(
            model, task_names[0],
            num_demos=args.num_demos,
            num_rollouts=args.num_rollouts,
            max_execution_steps=args.max_steps,
            execution_horizon=args.execution_horizon,
            num_traj_wp=args.num_traj_wp,
            headless=headless,
            device=args.device,
        )
        summary = {
            'per_task': {task_names[0]: result},
            'avg_success_rate': result['success_rate'],
            'num_tasks_evaluated': 1,
            'num_tasks_total': 1,
        }
    else:
        # Multiple tasks
        summary = evaluate_all_tasks(
            model,
            task_names=task_names,
            num_demos=args.num_demos,
            num_rollouts=args.num_rollouts,
            max_execution_steps=args.max_steps,
            execution_horizon=args.execution_horizon,
            num_traj_wp=args.num_traj_wp,
            headless=headless,
            device=args.device,
        )

    # Print results
    print_results_table(summary)

    # Save results
    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results) or '.', exist_ok=True)
        with open(args.save_results, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\nResults saved to {args.save_results}")


if __name__ == '__main__':
    main()
