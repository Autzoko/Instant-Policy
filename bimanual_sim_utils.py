"""
Bimanual evaluation utilities for Instant Policy on PerAct2 RLBench tasks.

Provides environment setup, demonstration collection, closed-loop rollout,
and success rate computation for dual-arm manipulation tasks.

Evaluation protocol (aligned with the Instant Policy paper, Section 4):
  1. Create bimanual RLBench environment for a given task.
  2. Collect N live demonstrations as in-context examples.
  3. For each rollout:
     a. Reset task (randomises object poses).
     b. Closed-loop execution: observe → predict → execute → repeat.
     c. Record binary success (task completion with positive reward).
  4. Report success rate = num_successes / num_rollouts.
"""
import numpy as np
import torch
from typing import List, Tuple, Optional
from tqdm import tqdm, trange

from utils import pose_to_transform, transform_to_pose, subsample_pcd, transform_pcd


# ── PerAct2 bimanual tasks ───────────────────────────────────────────
# 13 tasks from the PerAct2 benchmark (Grotz et al., 2024)
BIMANUAL_TASK_MAP = {
    'coordinated_push_box':          'CoordinatedPushBox',
    'coordinated_lift_ball':         'CoordinatedLiftBall',
    'coordinated_lift_tray':         'CoordinatedLiftTray',
    'handover_item':                 'HandoverItem',
    'bimanual_straighten_rope':      'BimanualStraightenRope',
    'bimanual_sweep_to_dustpan':     'BimanualSweepToDustpan',
    'bimanual_pick_laptop':          'BimanualPickLaptop',
    'bimanual_pick_plate':           'BimanualPickPlate',
    'dual_push_buttons':             'DualPushButtons',
    'bimanual_put_item_in_drawer':   'BimanualPutItemInDrawer',
    'bimanual_place_wine':           'BimanualPlaceWine',
    'bimanual_put_bottle_in_fridge': 'BimanualPutBottleInFridge',
    'bimanual_close_laptop':         'BimanualCloseLaptop',
}


# ── Point cloud extraction ───────────────────────────────────────────

def get_bimanual_point_cloud(
    obs,
    camera_names: Tuple[str, ...] = ('front', 'left_shoulder', 'right_shoulder'),
    use_mask: bool = True,
    mask_threshold: int = 60,
) -> np.ndarray:
    """
    Merge point clouds from multiple cameras.

    When use_mask=True, only points belonging to task-relevant objects
    are kept (using RLBench's ground-truth segmentation masks).
    This mirrors the single-arm sim_utils.get_point_cloud().
    """
    pcds = []
    for cam in camera_names:
        pc_attr = f'{cam}_point_cloud'
        if not hasattr(obs, pc_attr):
            continue
        pc = getattr(obs, pc_attr)
        if pc is None:
            continue
        if pc.ndim == 3:
            pc = pc.reshape(-1, 3)

        if use_mask:
            mask_attr = f'{cam}_mask'
            if hasattr(obs, mask_attr):
                mask = getattr(obs, mask_attr)
                if mask is not None:
                    if mask.ndim > 1:
                        mask = mask.reshape(-1)
                    # Keep only object pixels (same heuristic as single-arm)
                    valid_mask = mask > mask_threshold
                    if valid_mask.shape[0] == pc.shape[0]:
                        pc = pc[valid_mask]

        # Filter invalid points
        valid = np.isfinite(pc).all(axis=1) & (np.abs(pc) < 5.0).all(axis=1)
        pc = pc[valid]
        if len(pc) > 0:
            pcds.append(pc)

    if pcds:
        return np.concatenate(pcds, axis=0)
    return np.random.randn(100, 3).astype(np.float32) * 0.1


# ── Pose extraction ──────────────────────────────────────────────────

def extract_bimanual_poses(obs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract left and right end-effector poses as (4, 4) matrices.
    Handles both PerAct2 bimanual and standard RLBench observation formats.
    """
    # PerAct2 bimanual format: 7D pose [x,y,z,qx,qy,qz,qw]
    if hasattr(obs, 'left_gripper_pose') and hasattr(obs, 'right_gripper_pose'):
        T_l = pose_to_transform(obs.left_gripper_pose)
        T_r = pose_to_transform(obs.right_gripper_pose)
        return T_l, T_r

    # Alternative attribute names
    if hasattr(obs, 'left_ee_pose') and hasattr(obs, 'right_ee_pose'):
        T_l = pose_to_transform(obs.left_ee_pose)
        T_r = pose_to_transform(obs.right_ee_pose)
        return T_l, T_r

    # Matrix format
    if hasattr(obs, 'left_gripper_matrix') and hasattr(obs, 'right_gripper_matrix'):
        return (
            np.array(obs.left_gripper_matrix, dtype=np.float32).reshape(4, 4),
            np.array(obs.right_gripper_matrix, dtype=np.float32).reshape(4, 4),
        )

    raise ValueError(
        "Cannot extract bimanual EE poses from observation. "
        "Expected attributes: left_gripper_pose/right_gripper_pose "
        "or left_ee_pose/right_ee_pose."
    )


def extract_bimanual_grippers(obs) -> Tuple[int, int]:
    """Extract left and right gripper states (0=closed, 1=open)."""
    gl = 1
    gr = 1

    if hasattr(obs, 'left_gripper_open'):
        gl = int(float(obs.left_gripper_open) > 0.5)
    if hasattr(obs, 'right_gripper_open'):
        gr = int(float(obs.right_gripper_open) > 0.5)

    return gl, gr


# ── Demo processing ──────────────────────────────────────────────────

def bimanual_demo_to_sample(demo, num_pcd_points: int = 2048) -> dict:
    """
    Convert a raw RLBench bimanual demo into the dict format expected
    by BimanualGraphDiffusionPolicy.

    Returns:
        {
            'pcds':          list of (2048, 3) np.ndarray
            'T_w_es_left':   list of (4, 4) np.ndarray
            'T_w_es_right':  list of (4, 4) np.ndarray
            'grips_left':    list of int
            'grips_right':   list of int
        }
    """
    sample = {
        'pcds': [],
        'T_w_es_left': [],
        'T_w_es_right': [],
        'grips_left': [],
        'grips_right': [],
    }

    for obs in demo:
        pcd = get_bimanual_point_cloud(obs)
        pcd = subsample_pcd(pcd, num_pcd_points)
        sample['pcds'].append(pcd)

        T_l, T_r = extract_bimanual_poses(obs)
        sample['T_w_es_left'].append(T_l)
        sample['T_w_es_right'].append(T_r)

        gl, gr = extract_bimanual_grippers(obs)
        sample['grips_left'].append(gl)
        sample['grips_right'].append(gr)

    return sample


def downsample_bimanual_sample(sample: dict, target_length: int = 10) -> dict:
    """
    Downsample a bimanual demonstration to target_length waypoints.

    Key-frame selection (aligned with PerAct2 Section 3.5):
      1. Always include start and end.
      2. Include timesteps where EITHER arm's gripper state changed.
      3. Include timesteps where EITHER arm velocity dropped significantly.
      4. Fill remaining slots with evenly spaced intermediates.
    """
    L = len(sample['pcds'])
    if L <= target_length:
        result = {k: list(v) for k, v in sample.items()}
        while len(result['pcds']) < target_length:
            for k in result:
                result[k].append(result[k][-1])
        return result

    key_indices = {0, L - 1}

    # Gripper state changes on EITHER arm
    for arm in ('left', 'right'):
        grips = sample[f'grips_{arm}']
        for i in range(1, L):
            if grips[i] != grips[i - 1]:
                key_indices.add(i)

    # Velocity slowdowns on EITHER arm
    if L > 2:
        for arm in ('left', 'right'):
            poses = sample[f'T_w_es_{arm}']
            velocities = []
            for i in range(1, L):
                diff = np.linalg.norm(poses[i][:3, 3] - poses[i - 1][:3, 3])
                velocities.append(diff)
            if velocities:
                mean_vel = np.mean(velocities)
                if mean_vel > 1e-6:
                    for i, v in enumerate(velocities):
                        if v < mean_vel * 0.3:
                            key_indices.add(i + 1)

    key_indices = sorted(key_indices)

    # Fill with evenly spaced intermediates
    while len(key_indices) < target_length:
        max_gap = 0
        max_gap_idx = 0
        for i in range(len(key_indices) - 1):
            gap = key_indices[i + 1] - key_indices[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
        mid = (key_indices[max_gap_idx] + key_indices[max_gap_idx + 1]) // 2
        key_indices.insert(max_gap_idx + 1, mid)

    # Trim if too many (keep start, end, and gripper change frames)
    if len(key_indices) > target_length:
        must_keep = {0, L - 1}
        for arm in ('left', 'right'):
            grips = sample[f'grips_{arm}']
            for i in range(1, L):
                if grips[i] != grips[i - 1]:
                    must_keep.add(i)
        optional = [idx for idx in key_indices if idx not in must_keep]
        needed = target_length - len(must_keep)
        if needed > 0 and optional:
            step = max(1, len(optional) // needed)
            selected = optional[::step][:needed]
            key_indices = sorted(must_keep | set(selected))
        else:
            key_indices = sorted(must_keep)
        key_indices = key_indices[:target_length]

    return {k: [v[i] for i in key_indices] for k, v in sample.items()}


def sample_to_tensors(sample: dict) -> dict:
    """Convert numpy sample to torch tensors for model input."""
    return {
        'pcds': [torch.from_numpy(p).float() for p in sample['pcds']],
        'T_w_es_left': [torch.from_numpy(t).float()
                        for t in sample['T_w_es_left']],
        'T_w_es_right': [torch.from_numpy(t).float()
                         for t in sample['T_w_es_right']],
        'grips_left': list(sample['grips_left']),
        'grips_right': list(sample['grips_right']),
    }


# ── Environment creation ─────────────────────────────────────────────

def create_bimanual_env(task_name: str, headless: bool = True):
    """
    Create a PerAct2 bimanual RLBench environment.

    Returns: (env, task) ready for demo collection and rollout.
    """
    from rlbench.observation_config import ObservationConfig
    from rlbench.environment import Environment

    # Try bimanual action modes first (PerAct2 fork)
    try:
        from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import (
            BimanualEndEffectorPoseViaIK,
        )
        from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

        action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaIK(),
            gripper_action_mode=BimanualDiscrete(),
        )
    except ImportError:
        # Fallback: try IK variant naming
        from rlbench.action_modes.action_mode import BimanualMoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import (
            BimanualEndEffectorPoseViaPlanning,
        )
        from rlbench.action_modes.gripper_action_modes import BimanualDiscrete

        action_mode = BimanualMoveArmThenGripper(
            arm_action_mode=BimanualEndEffectorPoseViaPlanning(),
            gripper_action_mode=BimanualDiscrete(),
        )

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    obs_config.set_all_low_dim(True)

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()

    # Resolve task class
    task_cls = _resolve_task_class(task_name, env)
    task = env.get_task(task_cls)

    return env, task


def _resolve_task_class(task_name: str, env):
    """Resolve task name to RLBench task class, trying multiple naming conventions."""
    import rlbench.tasks as rlbench_tasks

    # Try direct class name from our mapping
    if task_name in BIMANUAL_TASK_MAP:
        cls_name = BIMANUAL_TASK_MAP[task_name]
        if hasattr(rlbench_tasks, cls_name):
            return getattr(rlbench_tasks, cls_name)

    # Try CamelCase conversion of task_name
    camel = ''.join(word.capitalize() for word in task_name.split('_'))
    if hasattr(rlbench_tasks, camel):
        return getattr(rlbench_tasks, camel)

    # Try as-is
    if hasattr(rlbench_tasks, task_name):
        return getattr(rlbench_tasks, task_name)

    raise ValueError(
        f"Task '{task_name}' not found in rlbench.tasks. "
        f"Available bimanual tasks: {list(BIMANUAL_TASK_MAP.keys())}"
    )


# ── Core rollout function ────────────────────────────────────────────

def rollout_bimanual_model(
    model,
    task_name: str,
    num_demos: int = 2,
    num_rollouts: int = 100,
    max_execution_steps: int = 30,
    execution_horizon: int = 8,
    num_traj_wp: int = 10,
    num_pcd_points: int = 2048,
    headless: bool = True,
    device: str = 'cuda',
) -> dict:
    """
    Evaluate a bimanual model on a PerAct2 RLBench task.

    Protocol (aligned with Instant Policy paper Section 4.1):
      1. Create environment and collect N live demonstrations.
      2. For each rollout:
         a. Reset task (object poses randomised).
         b. Closed-loop: observe -> predict T actions -> execute -> repeat.
         c. Record success = (task terminated AND reward > 0).
      3. Return success rate and per-rollout results.

    Args:
        model:                Trained BimanualGraphDiffusionPolicy (eval mode).
        task_name:            PerAct2 bimanual task name.
        num_demos:            Number of demonstrations as context (N).
        num_rollouts:         Number of evaluation rollouts per task.
        max_execution_steps:  Maximum observe-predict-execute cycles.
        execution_horizon:    Actions executed per cycle (T).
        num_traj_wp:          Demo waypoints after downsampling (L).
        num_pcd_points:       Points per point cloud observation.
        headless:             Run without GUI.
        device:               Torch device.

    Returns:
        {
            'task': task_name,
            'success_rate': float,
            'num_successes': int,
            'num_rollouts': int,
            'per_rollout': list of int (0 or 1),
        }
    """
    # ── 1. Create environment ────────────────────────────────────
    env, task = create_bimanual_env(task_name, headless=headless)

    # ── 2. Collect demonstrations as context ─────────────────────
    context_demos = []
    pbar_demo = tqdm(
        range(num_demos), desc=f'[{task_name}] Collecting demos',
        total=num_demos, leave=False,
    )
    for i in pbar_demo:
        collected = False
        for attempt in range(20):
            try:
                raw_demos = task.get_demos(1, live_demos=True)
                raw_sample = bimanual_demo_to_sample(
                    raw_demos[0], num_pcd_points)
                ds_sample = downsample_bimanual_sample(
                    raw_sample, num_traj_wp)
                tensor_sample = sample_to_tensors(ds_sample)
                assert len(tensor_sample['pcds']) == num_traj_wp
                context_demos.append(tensor_sample)
                collected = True
                break
            except Exception as e:
                if attempt == 19:
                    print(f"  Warning: failed to collect demo {i} "
                          f"after 20 attempts: {e}")
                continue
        if not collected:
            # Use a dummy demo as last resort
            print(f"  Demo {i} collection failed, using last valid demo.")
            if context_demos:
                context_demos.append(context_demos[-1])

    if not context_demos:
        env.shutdown()
        raise RuntimeError(
            f"Failed to collect any demos for task '{task_name}'")

    # ── 3. Run rollouts ──────────────────────────────────────────
    successes = []
    pbar = trange(
        num_rollouts,
        desc=f'[{task_name}] SR: 0/0',
        leave=False,
    )

    for rollout_idx in pbar:
        # Reset task (randomises object poses)
        reset_ok = False
        for attempt in range(10):
            try:
                task.reset()
                reset_ok = True
                break
            except Exception:
                continue
        if not reset_ok:
            successes.append(0)
            continue

        success = 0
        for step in range(max_execution_steps):
            # ── Observe ──────────────────────────────────────
            try:
                curr_obs = task.get_observation()
            except Exception:
                break

            # Extract current observation
            pcd_raw = get_bimanual_point_cloud(curr_obs)
            pcd = subsample_pcd(pcd_raw, num_pcd_points)
            T_l, T_r = extract_bimanual_poses(curr_obs)
            gl, gr = extract_bimanual_grippers(curr_obs)

            # Build model input
            full_sample = {
                'demos': context_demos,
                'current': {
                    'pcd': torch.from_numpy(pcd).float(),
                    'T_w_e_left': torch.from_numpy(T_l).float(),
                    'T_w_e_right': torch.from_numpy(T_r).float(),
                    'grip_left': gl,
                    'grip_right': gr,
                },
            }

            # ── Predict ──────────────────────────────────────
            try:
                with torch.no_grad():
                    actions_l, grips_l, actions_r, grips_r = \
                        model.predict_actions(full_sample)
            except Exception as e:
                print(f"  Rollout {rollout_idx}, step {step}: "
                      f"predict_actions failed: {e}")
                break

            # Move to CPU numpy
            actions_l = actions_l.cpu().numpy()
            actions_r = actions_r.cpu().numpy()
            grips_l = grips_l.cpu().numpy()
            grips_r = grips_r.cpu().numpy()

            # ── Execute ──────────────────────────────────────
            terminated = False
            for j in range(min(execution_horizon, len(actions_l))):
                # Absolute target poses = current_EE × relative_action
                target_l = T_l @ actions_l[j]
                target_r = T_r @ actions_r[j]

                # Convert to 7D pose [x,y,z,qx,qy,qz,qw]
                pose_l = transform_to_pose(target_l)
                pose_r = transform_to_pose(target_r)

                # Gripper commands (binary)
                grip_cmd_l = float(grips_l[j] > 0.5)
                grip_cmd_r = float(grips_r[j] > 0.5)

                # Bimanual action: [left_pose(7), left_grip(1),
                #                   right_pose(7), right_grip(1)]
                env_action = np.concatenate([
                    pose_l, [grip_cmd_l],
                    pose_r, [grip_cmd_r],
                ])

                try:
                    curr_obs, reward, terminate = task.step(env_action)
                    # Update EE poses for next action in this horizon
                    T_l, T_r = extract_bimanual_poses(curr_obs)
                    success = int(terminate and reward > 0.0)
                except Exception:
                    terminate = True

                if terminate:
                    terminated = True
                    break

            if terminated:
                break

        successes.append(success)
        n_done = len(successes)
        n_succ = sum(successes)
        pbar.set_description(
            f'[{task_name}] SR: {n_succ}/{n_done} '
            f'({100 * n_succ / n_done:.0f}%)'
        )

    pbar.close()
    env.shutdown()

    sr = sum(successes) / len(successes) if successes else 0.0
    return {
        'task': task_name,
        'success_rate': sr,
        'num_successes': sum(successes),
        'num_rollouts': len(successes),
        'per_rollout': successes,
    }


# ── Multi-task evaluation ────────────────────────────────────────────

def evaluate_all_tasks(
    model,
    task_names: Optional[List[str]] = None,
    num_demos: int = 2,
    num_rollouts: int = 100,
    headless: bool = True,
    device: str = 'cuda',
    **kwargs,
) -> dict:
    """
    Evaluate a bimanual model on multiple PerAct2 tasks.

    Returns a summary dict with per-task and aggregate results.
    """
    if task_names is None:
        task_names = list(BIMANUAL_TASK_MAP.keys())

    results = {}
    for i, task_name in enumerate(task_names):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(task_names)}] Evaluating: {task_name}")
        print(f"{'='*60}")

        try:
            result = rollout_bimanual_model(
                model, task_name,
                num_demos=num_demos,
                num_rollouts=num_rollouts,
                headless=headless,
                device=device,
                **kwargs,
            )
            results[task_name] = result
            print(f"  -> {task_name}: "
                  f"{result['num_successes']}/{result['num_rollouts']} "
                  f"({result['success_rate']*100:.1f}%)")
        except Exception as e:
            print(f"  -> {task_name}: FAILED ({e})")
            results[task_name] = {
                'task': task_name,
                'success_rate': 0.0,
                'num_successes': 0,
                'num_rollouts': 0,
                'per_rollout': [],
                'error': str(e),
            }

    # Aggregate
    valid = [r for r in results.values() if r['num_rollouts'] > 0]
    avg_sr = np.mean([r['success_rate'] for r in valid]) if valid else 0.0

    return {
        'per_task': results,
        'avg_success_rate': avg_sr,
        'num_tasks_evaluated': len(valid),
        'num_tasks_total': len(task_names),
    }


def print_results_table(summary: dict):
    """Print a formatted results table (aligned with Instant Policy Table 1)."""
    print("\n" + "=" * 50)
    print("Bimanual Evaluation Results")
    print("=" * 50)
    print(f"{'Task':<40} {'SR':>8}")
    print("-" * 50)

    for task_name, result in summary['per_task'].items():
        if 'error' in result:
            sr_str = 'ERROR'
        else:
            n = result['num_successes']
            t = result['num_rollouts']
            sr_str = f'{n}/{t} ({result["success_rate"]*100:.0f}%)'
        print(f"  {task_name:<38} {sr_str:>8}")

    print("-" * 50)
    print(f"  {'Average':<38} "
          f"{summary['avg_success_rate']*100:.1f}%")
    print(f"  Tasks evaluated: "
          f"{summary['num_tasks_evaluated']}/{summary['num_tasks_total']}")
    print("=" * 50)
