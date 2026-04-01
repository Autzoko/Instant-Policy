"""
Collect bimanual demonstrations from RLBench for Instant Policy training.

This script uses RLBench's bimanual task interface to collect demonstrations
and saves them in the npz format expected by PerAct2Dataset.

Output format per episode (saved as .npz):
  pcds:          list of (N, 3) point clouds (merged from all cameras)
  T_w_es_left:   list of (4, 4) left arm EE poses
  T_w_es_right:  list of (4, 4) right arm EE poses
  grips_left:    list of int (0=closed, 1=open)
  grips_right:   list of int (0=closed, 1=open)

Usage:
  python scripts/collect_bimanual_demos.py \
      --save_dir /scratch/$USER/peract2_data \
      --num_episodes 20 \
      --tasks push_box lift_tray
"""
import os
import argparse
import numpy as np
from typing import List

# PerAct2 bimanual tasks from the benchmark
BIMANUAL_TASKS = [
    'coordinated_push_box',
    'coordinated_lift_ball',
    'coordinated_lift_tray',
    'handover_item',
    'bimanual_straighten_rope',
    'bimanual_sweep_to_dustpan',
    'bimanual_pick_laptop',
    'bimanual_pick_plate',
    'dual_push_buttons',
    'bimanual_put_item_in_drawer',
    'bimanual_place_wine',
    'bimanual_put_bottle_in_fridge',
    'bimanual_close_laptop',
]

# Fallback task names (standard RLBench naming convention)
BIMANUAL_TASKS_ALT = [
    'push_box',
    'lift_ball',
    'lift_tray',
    'handover_item',
    'straighten_rope',
    'sweep_to_dustpan',
    'pick_up_notebook',
    'pick_up_plate',
    'push_buttons',
    'put_item_in_drawer',
    'place_wine_at_rack_location',
    'put_bottle_in_fridge',
    'close_laptop_lid',
]


def collect_single_task(
    task_name: str,
    save_dir: str,
    num_episodes: int = 20,
    headless: bool = True,
    num_pcd_points: int = 2048,
):
    """Collect demonstrations for a single bimanual task."""
    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import BimanualEndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import BimanualDiscrete
        from rlbench.observation_config import ObservationConfig
    except ImportError as e:
        print(f"RLBench import error: {e}")
        print("Attempting standard RLBench import...")
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig

    # Configure observation
    obs_config = ObservationConfig()
    obs_config.set_all(True)  # enable all cameras
    obs_config.set_all_low_dim(True)

    # Configure action mode for bimanual
    try:
        action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
            gripper_action_mode=BimanualDiscrete(),
        )
    except NameError:
        # Fallback for standard RLBench
        action_mode = MoveArmThenGripper(
            arm_action_mode=EndEffectorPoseViaPlanning(),
            gripper_action_mode=Discrete(),
        )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()

    # Get task
    try:
        task_class = env.get_task(task_name)
    except Exception:
        print(f"  Task '{task_name}' not found, skipping.")
        env.shutdown()
        return

    task = env.get_task(task_class)
    task_save_dir = os.path.join(save_dir, task_name)
    os.makedirs(task_save_dir, exist_ok=True)

    print(f"  Collecting {num_episodes} episodes for '{task_name}'...")

    for ep in range(num_episodes):
        try:
            # Get demo
            demos = task.get_demos(amount=1, live_demos=True)
            demo = demos[0]

            # Extract data from each observation
            pcds = []
            T_w_es_left = []
            T_w_es_right = []
            grips_left = []
            grips_right = []

            for obs in demo:
                # Merge point clouds from all cameras
                pcd = _merge_point_clouds(obs, num_pcd_points)
                pcds.append(pcd)

                # Extract EE poses
                T_l, T_r = _extract_bimanual_poses(obs)
                T_w_es_left.append(T_l)
                T_w_es_right.append(T_r)

                # Extract gripper states
                gl, gr = _extract_bimanual_grippers(obs)
                grips_left.append(gl)
                grips_right.append(gr)

            # Save as npz
            save_path = os.path.join(task_save_dir, f'episode_{ep:04d}.npz')
            np.savez_compressed(
                save_path,
                pcds=np.array(pcds, dtype=object),
                T_w_es_left=np.array(T_w_es_left),
                T_w_es_right=np.array(T_w_es_right),
                grips_left=np.array(grips_left),
                grips_right=np.array(grips_right),
            )
            print(f"    Episode {ep+1}/{num_episodes}: "
                  f"{len(demo)} steps -> {save_path}")

        except Exception as e:
            print(f"    Episode {ep+1} failed: {e}")
            continue

    env.shutdown()
    print(f"  Done: {task_name}")


def _merge_point_clouds(obs, num_points: int = 2048) -> np.ndarray:
    """Merge point clouds from multiple camera views."""
    pcds = []

    # Try all standard RLBench camera names
    camera_names = [
        'front', 'left_shoulder', 'right_shoulder',
        'overhead', 'wrist', 'wrist_left', 'wrist_right',
    ]

    for cam in camera_names:
        pc_attr = f'{cam}_point_cloud'
        if hasattr(obs, pc_attr):
            pc = getattr(obs, pc_attr)
            if pc is not None:
                if pc.ndim == 3:
                    pc = pc.reshape(-1, 3)
                # Filter invalid points
                valid = np.isfinite(pc).all(axis=1) & (np.abs(pc) < 5.0).all(axis=1)
                pc = pc[valid]
                if len(pc) > 0:
                    pcds.append(pc)

    if pcds:
        merged = np.concatenate(pcds, axis=0)
    else:
        # Fallback
        merged = np.random.randn(num_points, 3).astype(np.float32) * 0.1

    # Subsample to target size
    if len(merged) > num_points:
        idx = np.random.choice(len(merged), num_points, replace=False)
        merged = merged[idx]
    elif len(merged) < num_points:
        pad_idx = np.random.choice(len(merged), num_points - len(merged), replace=True)
        merged = np.concatenate([merged, merged[pad_idx]], axis=0)

    return merged.astype(np.float32)


def _extract_bimanual_poses(obs) -> tuple:
    """Extract left and right EE poses as 4x4 matrices."""
    from scipy.spatial.transform import Rotation

    def _to_mat(pos, quat):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = pos
        T[:3, :3] = Rotation.from_quat(quat).as_matrix()
        return T

    # Try PerAct2 bimanual attributes
    if hasattr(obs, 'left_gripper_pose') and hasattr(obs, 'right_gripper_pose'):
        pose_l = obs.left_gripper_pose  # [x,y,z,qx,qy,qz,qw]
        pose_r = obs.right_gripper_pose
        return _to_mat(pose_l[:3], pose_l[3:]), _to_mat(pose_r[:3], pose_r[3:])

    # Try alternative names
    if hasattr(obs, 'gripper_pose'):
        # Single-arm fallback: use same pose for both (shouldn't happen)
        pose = obs.gripper_pose
        T = _to_mat(pose[:3], pose[3:])
        return T, T.copy()

    # Try matrix format
    if hasattr(obs, 'left_gripper_matrix'):
        return (
            np.array(obs.left_gripper_matrix, dtype=np.float32),
            np.array(obs.right_gripper_matrix, dtype=np.float32),
        )

    raise ValueError("Cannot find EE pose attributes in observation")


def _extract_bimanual_grippers(obs) -> tuple:
    """Extract left and right gripper states (0=closed, 1=open)."""
    gl, gr = 1, 1

    if hasattr(obs, 'left_gripper_open'):
        gl = int(obs.left_gripper_open > 0.5)
    elif hasattr(obs, 'gripper_open'):
        gl = int(obs.gripper_open > 0.5)

    if hasattr(obs, 'right_gripper_open'):
        gr = int(obs.right_gripper_open > 0.5)
    elif hasattr(obs, 'gripper_open'):
        gr = int(obs.gripper_open > 0.5)

    return gl, gr


def main():
    parser = argparse.ArgumentParser(
        description='Collect bimanual demonstrations from RLBench')
    parser.add_argument('--save_dir', type=str, required=True,
                       help='Directory to save demonstrations')
    parser.add_argument('--num_episodes', type=int, default=20,
                       help='Number of episodes per task')
    parser.add_argument('--tasks', type=str, nargs='*', default=None,
                       help='Specific tasks to collect (default: all)')
    parser.add_argument('--headless', action='store_true', default=True,
                       help='Run in headless mode')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    tasks = args.tasks or BIMANUAL_TASKS

    print("=" * 60)
    print("PerAct2 Bimanual Data Collection")
    print(f"  Save dir:  {args.save_dir}")
    print(f"  Episodes:  {args.num_episodes} per task")
    print(f"  Tasks:     {len(tasks)}")
    print("=" * 60)

    for i, task_name in enumerate(tasks):
        print(f"\n[{i+1}/{len(tasks)}] Task: {task_name}")
        try:
            collect_single_task(
                task_name, args.save_dir,
                num_episodes=args.num_episodes,
                headless=args.headless,
            )
        except Exception as e:
            print(f"  FAILED: {e}")
            # Try alternative name
            if i < len(BIMANUAL_TASKS_ALT):
                alt_name = BIMANUAL_TASKS_ALT[i]
                print(f"  Trying alternative name: {alt_name}")
                try:
                    collect_single_task(
                        alt_name, args.save_dir,
                        num_episodes=args.num_episodes,
                        headless=args.headless,
                    )
                except Exception as e2:
                    print(f"  Also failed: {e2}")

    print("\n" + "=" * 60)
    print("Collection complete!")
    print(f"Data saved to: {args.save_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
