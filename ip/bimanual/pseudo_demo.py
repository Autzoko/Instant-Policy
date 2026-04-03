"""
Bimanual pseudo-demonstration generator.

Extends the single-arm pseudo-demo system with dual-arm task primitives:
  - bimanual_grasp:    both arms approach from opposite sides, grasp, lift
  - bimanual_handover: left grasps, transfers to right (or vice versa)
  - bimanual_place:    coordinated placement with both arms
  - bimanual_push:     push a large object with both arms
  - bimanual_open:     one arm holds, other pulls/opens

Also supports random bimanual waypoints for diversity.
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation, Slerp

from ..pseudo_demo import (
    load_shapenet_meshes,
    sample_scene,
    render_point_clouds,
    augment_trajectory,
)


# ──────────────────────────────────────────────────────────────────────
# Bimanual waypoint sampling
# ──────────────────────────────────────────────────────────────────────

def sample_bimanual_waypoints_random(
    objects: List[dict],
    num_waypoints: int = None,
) -> List[dict]:
    """
    Sample random bimanual waypoints near objects.

    Returns: list of waypoints, each with left and right arm data:
      {
        'position_left': (3,), 'rotation_left': (3,3),
        'grip_change_left': bool,
        'position_right': (3,), 'rotation_right': (3,3),
        'grip_change_right': bool,
      }
    """
    if num_waypoints is None:
        num_waypoints = np.random.randint(2, 7)

    waypoints = []
    for _ in range(num_waypoints):
        wp = {}
        for arm in ('left', 'right'):
            obj = objects[np.random.randint(len(objects))]
            offset = np.random.randn(3) * 0.03
            pos = obj['position'] + offset
            pos[2] = max(pos[2], 0.01)

            rot = Rotation.random().as_matrix()
            euler = Rotation.from_matrix(rot).as_euler('xyz')
            euler = np.clip(euler, -np.pi / 3, np.pi / 3)
            rot = Rotation.from_euler('xyz', euler).as_matrix()

            wp[f'position_{arm}'] = pos
            wp[f'rotation_{arm}'] = rot
            wp[f'grip_change_{arm}'] = np.random.random() < 0.3

        waypoints.append(wp)
    return waypoints


def sample_bimanual_waypoints_biased(
    objects: List[dict],
    task_type: str = None,
) -> List[dict]:
    """
    Biased bimanual waypoints toward common bimanual manipulation
    primitives.

    task_type: 'bimanual_grasp', 'bimanual_handover', 'bimanual_place',
               'bimanual_push', 'bimanual_open', or None (random).
    """
    if task_type is None:
        task_type = np.random.choice([
            'bimanual_grasp', 'bimanual_handover',
            'bimanual_place', 'bimanual_push', 'bimanual_open',
        ])

    obj = objects[0]
    obj_pos = obj['position'].copy()
    # Opposite approach directions for left and right
    approach_offset_l = np.array([-0.06, 0, 0])
    approach_offset_r = np.array([0.06, 0, 0])

    if task_type == 'bimanual_grasp':
        # Both arms approach from opposite sides, close grippers, lift
        waypoints = [
            # Approach
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.08],
                obj_pos + approach_offset_r + [0, 0, 0.08],
            ),
            # Grasp
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.02],
                obj_pos + approach_offset_r + [0, 0, 0.02],
                grip_change_left=True, grip_change_right=True,
            ),
            # Lift
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.12],
                obj_pos + approach_offset_r + [0, 0, 0.12],
            ),
        ]

    elif task_type == 'bimanual_handover':
        # Left grasps, moves to centre, right grasps, left releases
        centre = obj_pos + np.array([0, 0, 0.10])
        waypoints = [
            # Left approaches
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.08],
                obj_pos + approach_offset_r + [0, 0, 0.15],
            ),
            # Left grasps
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.02],
                obj_pos + approach_offset_r + [0, 0, 0.15],
                grip_change_left=True,
            ),
            # Left lifts to centre, right approaches
            _bim_wp(
                centre + approach_offset_l * 0.5,
                centre + approach_offset_r * 0.5,
            ),
            # Right grasps
            _bim_wp(
                centre + approach_offset_l * 0.5,
                centre + approach_offset_r * 0.3,
                grip_change_right=True,
            ),
            # Left releases
            _bim_wp(
                centre + approach_offset_l,
                centre + approach_offset_r * 0.3,
                grip_change_left=True,
            ),
        ]

    elif task_type == 'bimanual_place':
        # Both arms carry object to a target location
        target = objects[min(1, len(objects) - 1)]
        target_pos = target['position'].copy()
        waypoints = [
            # Approach
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.08],
                obj_pos + approach_offset_r + [0, 0, 0.08],
            ),
            # Grasp
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.02],
                obj_pos + approach_offset_r + [0, 0, 0.02],
                grip_change_left=True, grip_change_right=True,
            ),
            # Lift
            _bim_wp(
                obj_pos + approach_offset_l + [0, 0, 0.12],
                obj_pos + approach_offset_r + [0, 0, 0.12],
            ),
            # Move to target
            _bim_wp(
                target_pos + approach_offset_l + [0, 0, 0.12],
                target_pos + approach_offset_r + [0, 0, 0.12],
            ),
            # Place
            _bim_wp(
                target_pos + approach_offset_l + [0, 0, 0.03],
                target_pos + approach_offset_r + [0, 0, 0.03],
                grip_change_left=True, grip_change_right=True,
            ),
        ]

    elif task_type == 'bimanual_push':
        # Both arms push object in the same direction
        start_l = obj_pos + np.array([-0.08, 0.04, 0.02])
        start_r = obj_pos + np.array([-0.08, -0.04, 0.02])
        end_l = obj_pos + np.array([0.05, 0.04, 0.02])
        end_r = obj_pos + np.array([0.05, -0.04, 0.02])
        waypoints = [
            _bim_wp(start_l, start_r),
            _bim_wp(end_l, end_r),
        ]

    else:  # bimanual_open
        # Left arm holds, right arm pulls/opens
        hold_pos = obj_pos + np.array([-0.04, 0, 0.03])
        pull_start = obj_pos + np.array([0.04, 0, 0.03])
        pull_end = obj_pos + np.array([0.04, 0.08, 0.08])
        waypoints = [
            _bim_wp(hold_pos, pull_start,
                     grip_change_left=True, grip_change_right=True),
            _bim_wp(hold_pos, pull_end),
        ]

    # Add small random perturbations to rotations
    for wp in waypoints:
        for arm in ('left', 'right'):
            perturb = Rotation.from_rotvec(np.random.randn(3) * 0.1).as_matrix()
            wp[f'rotation_{arm}'] = perturb @ wp[f'rotation_{arm}']

    return waypoints


def _bim_wp(pos_left, pos_right,
            grip_change_left=False, grip_change_right=False):
    """Helper to create a bimanual waypoint dict."""
    return {
        'position_left': np.array(pos_left, dtype=np.float64),
        'rotation_left': np.eye(3),
        'grip_change_left': grip_change_left,
        'position_right': np.array(pos_right, dtype=np.float64),
        'rotation_right': np.eye(3),
        'grip_change_right': grip_change_right,
    }


# ──────────────────────────────────────────────────────────────────────
# Bimanual trajectory interpolation
# ──────────────────────────────────────────────────────────────────────

def interpolate_bimanual_trajectory(
    waypoints: List[dict],
    spacing_trans: float = 0.01,
    spacing_rot_deg: float = 3.0,
) -> List[dict]:
    """
    Interpolate both arms independently with matching timesteps.

    Returns dense trajectory: list of
      {T_we_left: (4,4), T_we_right: (4,4), grip_left: int, grip_right: int}
    """
    # Compute per-arm dense trajectories
    traj_left = _interpolate_single_arm(waypoints, 'left', spacing_trans)
    traj_right = _interpolate_single_arm(waypoints, 'right', spacing_trans)

    # Synchronise to the longer trajectory length
    max_len = max(len(traj_left), len(traj_right))
    traj_left = _resample(traj_left, max_len)
    traj_right = _resample(traj_right, max_len)

    # Merge
    trajectory = []
    for sl, sr in zip(traj_left, traj_right):
        trajectory.append({
            'T_we_left': sl['T_we'],
            'T_we_right': sr['T_we'],
            'grip_left': sl['grip'],
            'grip_right': sr['grip'],
        })
    return trajectory


def _interpolate_single_arm(waypoints: List[dict], arm: str,
                            spacing_trans: float) -> List[dict]:
    """Interpolate one arm's trajectory from bimanual waypoints."""
    positions = np.array([wp[f'position_{arm}'] for wp in waypoints])
    rotations = [Rotation.from_matrix(wp[f'rotation_{arm}']) for wp in waypoints]

    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = segment_lengths.sum()
    num_steps = max(int(total_length / spacing_trans), len(waypoints))

    t_wp = np.linspace(0, 1, len(waypoints))
    t_dense = np.linspace(0, 1, num_steps)

    # Position interpolation (linear)
    dense_positions = np.array([
        np.interp(t_dense, t_wp, positions[:, d])
        for d in range(3)
    ]).T

    # Rotation interpolation (slerp)
    key_rots = Rotation.concatenate(rotations)
    slerp = Slerp(t_wp, key_rots)
    dense_rotations = slerp(t_dense)

    # Gripper state
    grip_changes = [wp.get(f'grip_change_{arm}', False) for wp in waypoints]
    grip_state = 1  # start open
    grip_states = []
    last_flipped_wp = -1
    for i, t in enumerate(t_dense):
        wp_idx = np.argmin(np.abs(t_wp - t))
        if (grip_changes[wp_idx] and wp_idx != last_flipped_wp
                and abs(t - t_wp[wp_idx]) < 0.01):
            grip_state = 1 - grip_state
            last_flipped_wp = wp_idx
        grip_states.append(grip_state)

    trajectory = []
    for i in range(num_steps):
        T_we = np.eye(4)
        T_we[:3, :3] = dense_rotations[i].as_matrix()
        T_we[:3, 3] = dense_positions[i]
        trajectory.append({'T_we': T_we, 'grip': grip_states[i]})
    return trajectory


def _resample(trajectory: List[dict], target_len: int) -> List[dict]:
    """Resample trajectory to target_len by linear index interpolation."""
    n = len(trajectory)
    if n == target_len:
        return trajectory
    indices = np.linspace(0, n - 1, target_len)
    result = []
    for idx in indices:
        i = int(idx)
        i = min(i, n - 1)
        result.append(trajectory[i])
    return result


# ──────────────────────────────────────────────────────────────────────
# Bimanual data augmentation
# ──────────────────────────────────────────────────────────────────────

def augment_bimanual_trajectory(
    trajectory: List[dict],
    disturbance_prob: float = 0.3,
    grip_flip_prob: float = 0.1,
) -> List[dict]:
    """Apply data augmentation to a bimanual trajectory."""
    aug = [dict(t) for t in trajectory]

    # Local disturbances (independently per arm)
    for arm in ('left', 'right'):
        if np.random.random() < disturbance_prob:
            num_disturb = np.random.randint(1, max(2, len(aug) // 5))
            for _ in range(num_disturb):
                idx = np.random.randint(1, len(aug) - 1)
                T = aug[idx][f'T_we_{arm}'].copy()
                T[:3, 3] += np.random.randn(3) * 0.005
                perturb = Rotation.from_rotvec(
                    np.random.randn(3) * 0.02
                ).as_matrix()
                T[:3, :3] = perturb @ T[:3, :3]
                aug[idx][f'T_we_{arm}'] = T

    # Gripper flip (independently per arm)
    for arm in ('left', 'right'):
        if np.random.random() < grip_flip_prob:
            flip_idx = np.random.randint(0, len(aug))
            aug[flip_idx][f'grip_{arm}'] = 1 - aug[flip_idx][f'grip_{arm}']

    return aug


# ──────────────────────────────────────────────────────────────────────
# Full bimanual pseudo-demonstration generation
# ──────────────────────────────────────────────────────────────────────

def generate_bimanual_pseudo_task(
    mesh_paths: List[str],
    num_demos: int = 3,
) -> List[List[dict]]:
    """
    Generate a complete bimanual pseudo-task: multiple demonstrations
    performing the same task with randomised starting poses.

    Returns: list of num_demos demonstrations, each a list of
      {pcd: (N,3), T_we_left: (4,4), T_we_right: (4,4),
       grip_left: int, grip_right: int}
    """
    demos = []
    base_objects = sample_scene(mesh_paths, num_objects=2)

    use_biased = np.random.random() < 0.5
    if use_biased:
        base_waypoints = sample_bimanual_waypoints_biased(base_objects)
    else:
        base_waypoints = sample_bimanual_waypoints_random(base_objects)

    for _ in range(num_demos):
        # Slightly randomise object poses
        objects = []
        for obj in base_objects:
            new_obj = dict(obj)
            new_obj['position'] = obj['position'] + np.random.randn(3) * 0.02
            perturb = Rotation.from_rotvec(np.random.randn(3) * 0.05).as_matrix()
            new_obj['rotation'] = perturb @ obj['rotation']
            objects.append(new_obj)

        # Adjust waypoints
        waypoints = []
        for wp in base_waypoints:
            new_wp = dict(wp)
            for arm in ('left', 'right'):
                new_wp[f'position_{arm}'] = (
                    wp[f'position_{arm}'] + np.random.randn(3) * 0.01
                )
            waypoints.append(new_wp)

        # Interpolate
        trajectory = interpolate_bimanual_trajectory(waypoints)
        trajectory = augment_bimanual_trajectory(trajectory)

        # Render point clouds (shared scene, using midpoint of both EEs)
        demo = []
        for step in trajectory:
            # Use midpoint pose for rendering viewpoint
            mid_T = np.eye(4)
            mid_T[:3, 3] = 0.5 * (
                step['T_we_left'][:3, 3] + step['T_we_right'][:3, 3]
            )
            pcd = render_point_clouds(objects, mid_T)
            demo.append({
                'pcd': pcd,
                'T_we_left': step['T_we_left'].astype(np.float32),
                'T_we_right': step['T_we_right'].astype(np.float32),
                'grip_left': step['grip_left'],
                'grip_right': step['grip_right'],
            })
        demos.append(demo)

    return demos


def generate_bimanual_pseudo_demo_batch(
    mesh_paths: List[str],
    batch_size: int = 32,
    num_demos_range: Tuple[int, int] = (1, 5),
) -> List[dict]:
    """
    Generate a batch of bimanual training samples.

    Each sample: N-1 demos as context, predict bimanual actions for the Nth.
    """
    samples = []
    for _ in range(batch_size):
        N = np.random.randint(num_demos_range[0] + 1, num_demos_range[1] + 2)
        task_demos = generate_bimanual_pseudo_task(mesh_paths, num_demos=N)

        context_demos = task_demos[:N - 1]
        target_demo = task_demos[N - 1]

        # Need at least 2 frames (1 current + 1 action)
        if len(target_demo) < 2:
            continue

        T = min(8, len(target_demo) - 1)
        obs_idx = np.random.randint(0, max(1, len(target_demo) - T))

        samples.append({
            'demos': [
                {
                    'pcds': [s['pcd'] for s in demo],
                    'T_w_es_left': [s['T_we_left'] for s in demo],
                    'T_w_es_right': [s['T_we_right'] for s in demo],
                    'grips_left': [s['grip_left'] for s in demo],
                    'grips_right': [s['grip_right'] for s in demo],
                }
                for demo in context_demos
            ],
            'current': {
                'pcd': target_demo[obs_idx]['pcd'],
                'T_w_e_left': target_demo[obs_idx]['T_we_left'],
                'T_w_e_right': target_demo[obs_idx]['T_we_right'],
                'grip_left': target_demo[obs_idx]['grip_left'],
                'grip_right': target_demo[obs_idx]['grip_right'],
            },
            'actions': {
                'T_EAs_left': [
                    np.linalg.inv(target_demo[obs_idx]['T_we_left']) @
                    target_demo[obs_idx + j + 1]['T_we_left']
                    for j in range(T)
                ],
                'T_EAs_right': [
                    np.linalg.inv(target_demo[obs_idx]['T_we_right']) @
                    target_demo[obs_idx + j + 1]['T_we_right']
                    for j in range(T)
                ],
                'grips_left': [
                    target_demo[obs_idx + j + 1]['grip_left'] for j in range(T)
                ],
                'grips_right': [
                    target_demo[obs_idx + j + 1]['grip_right'] for j in range(T)
                ],
            },
        })

    return samples
