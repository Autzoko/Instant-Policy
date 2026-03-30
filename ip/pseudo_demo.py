"""
Pseudo-demonstration generator (Section 3.4, Appendix D).

Generates procedural robot trajectories with ShapeNet objects in simulation:
  1. Sample 2 objects from ShapeNet, place randomly on a plane.
  2. Sample 2-6 waypoints on/near objects.
  3. Assign gripper state changes to simulate grasping/releasing.
  4. Interpolate between waypoints (linear, cubic, spherical).
  5. Render segmented point clouds + gripper poses via PyRender.

Half the pseudo-demonstrations use biased sampling (grasping, pick-and-place),
the other half use completely random waypoints.

Data augmentation:
  - 30% of trajectories include local disturbances (recovery behaviour).
  - 10% flip the gripper open-close state.
"""
import numpy as np
import os
from typing import List, Dict, Tuple, Optional
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import CubicSpline


# ──────────────────────────────────────────────────────────────────────
# ShapeNet mesh loading
# ──────────────────────────────────────────────────────────────────────

def load_shapenet_meshes(shapenet_root: str, category: str = None,
                         max_objects: int = 100) -> List[str]:
    """
    Scan ShapeNet directory for .obj mesh paths.
    Returns list of mesh file paths.
    """
    mesh_paths = []
    for root, dirs, files in os.walk(shapenet_root):
        for f in files:
            if f.endswith('.obj') and len(mesh_paths) < max_objects:
                mesh_paths.append(os.path.join(root, f))
    return mesh_paths


# ──────────────────────────────────────────────────────────────────────
# Scene setup
# ──────────────────────────────────────────────────────────────────────

def sample_scene(mesh_paths: List[str], num_objects: int = 2,
                 plane_extent: float = 0.3) -> List[dict]:
    """
    Sample objects and their placements on a table plane.
    Returns list of {mesh_path, position, rotation, scale}.
    """
    objects = []
    chosen = np.random.choice(len(mesh_paths), size=num_objects, replace=False)
    for idx in chosen:
        pos = np.array([
            np.random.uniform(-plane_extent, plane_extent),
            np.random.uniform(-plane_extent, plane_extent),
            0.0,  # on the table
        ])
        rot = Rotation.random().as_matrix()
        # Restrict to ±60 degrees (Appendix F)
        euler = Rotation.from_matrix(rot).as_euler('xyz')
        euler = np.clip(euler, -np.pi / 3, np.pi / 3)
        rot = Rotation.from_euler('xyz', euler).as_matrix()
        scale = np.random.uniform(0.03, 0.08)
        objects.append({
            'mesh_path': mesh_paths[idx],
            'position': pos,
            'rotation': rot,
            'scale': scale,
        })
    return objects


# ──────────────────────────────────────────────────────────────────────
# Waypoint sampling
# ──────────────────────────────────────────────────────────────────────

def sample_waypoints_random(objects: List[dict],
                             num_waypoints: int = None) -> List[dict]:
    """
    Sample random waypoints near objects.
    Returns: list of {position: (3,), rotation: (3,3), grip_change: bool}
    """
    if num_waypoints is None:
        num_waypoints = np.random.randint(2, 7)

    waypoints = []
    for i in range(num_waypoints):
        obj = objects[np.random.randint(len(objects))]
        # Position near object surface
        offset = np.random.randn(3) * 0.03
        pos = obj['position'] + offset
        pos[2] = max(pos[2], 0.01)  # stay above table

        rot = Rotation.random().as_matrix()
        euler = Rotation.from_matrix(rot).as_euler('xyz')
        euler = np.clip(euler, -np.pi / 3, np.pi / 3)
        rot = Rotation.from_euler('xyz', euler).as_matrix()

        # Random gripper state change
        grip_change = np.random.random() < 0.3
        waypoints.append({
            'position': pos,
            'rotation': rot,
            'grip_change': grip_change,
        })
    return waypoints


def sample_waypoints_biased(objects: List[dict],
                             task_type: str = None) -> List[dict]:
    """
    Biased sampling toward common manipulation primitives (Appendix D).
    task_type: 'grasp', 'pick_place', 'push', 'open_close' or None (random).
    """
    if task_type is None:
        task_type = np.random.choice(['grasp', 'pick_place', 'push', 'open_close'])

    obj = objects[np.random.randint(len(objects))]
    obj_pos = obj['position'].copy()

    if task_type == 'grasp':
        # Approach → grasp → lift
        approach = obj_pos + np.array([0, 0, 0.08])
        grasp_pos = obj_pos + np.array([0, 0, 0.02])
        lift_pos = obj_pos + np.array([0, 0, 0.10])
        waypoints = [
            {'position': approach, 'rotation': np.eye(3), 'grip_change': False},
            {'position': grasp_pos, 'rotation': np.eye(3), 'grip_change': True},
            {'position': lift_pos, 'rotation': np.eye(3), 'grip_change': False},
        ]

    elif task_type == 'pick_place':
        target = objects[(np.random.randint(len(objects) - 1) + 1) % len(objects)]
        target_pos = target['position'].copy()
        approach = obj_pos + np.array([0, 0, 0.08])
        grasp_pos = obj_pos + np.array([0, 0, 0.02])
        lift = obj_pos + np.array([0, 0, 0.12])
        place_above = target_pos + np.array([0, 0, 0.12])
        place = target_pos + np.array([0, 0, 0.03])
        waypoints = [
            {'position': approach, 'rotation': np.eye(3), 'grip_change': False},
            {'position': grasp_pos, 'rotation': np.eye(3), 'grip_change': True},
            {'position': lift, 'rotation': np.eye(3), 'grip_change': False},
            {'position': place_above, 'rotation': np.eye(3), 'grip_change': False},
            {'position': place, 'rotation': np.eye(3), 'grip_change': True},
        ]

    elif task_type == 'push':
        start = obj_pos + np.array([0.05, 0, 0.02])
        end = obj_pos + np.array([-0.05, 0, 0.02])
        waypoints = [
            {'position': start, 'rotation': np.eye(3), 'grip_change': False},
            {'position': end, 'rotation': np.eye(3), 'grip_change': False},
        ]

    else:  # open_close
        pos1 = obj_pos + np.array([0, 0, 0.03])
        pos2 = obj_pos + np.array([0, 0.05, 0.08])
        waypoints = [
            {'position': pos1, 'rotation': np.eye(3), 'grip_change': True},
            {'position': pos2, 'rotation': np.eye(3), 'grip_change': False},
        ]

    # Add small random perturbations to rotations
    for wp in waypoints:
        perturb = Rotation.from_rotvec(np.random.randn(3) * 0.1).as_matrix()
        wp['rotation'] = perturb @ wp['rotation']

    return waypoints


# ──────────────────────────────────────────────────────────────────────
# Trajectory interpolation
# ──────────────────────────────────────────────────────────────────────

def interpolate_trajectory(waypoints: List[dict],
                           spacing_trans: float = 0.01,
                           spacing_rot_deg: float = 3.0,
                           method: str = None) -> List[dict]:
    """
    Interpolate between waypoints with constant spacing (Appendix D).
    method: 'linear', 'cubic', or 'slerp' (random if None).
    Returns dense trajectory: list of {T_we: (4,4), grip: int}.
    """
    if method is None:
        method = np.random.choice(['linear', 'cubic', 'slerp'])

    positions = np.array([wp['position'] for wp in waypoints])
    rotations = [Rotation.from_matrix(wp['rotation']) for wp in waypoints]

    # Compute total arc length for position interpolation
    diffs = np.diff(positions, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = segment_lengths.sum()
    num_steps = max(int(total_length / spacing_trans), len(waypoints))

    t_wp = np.linspace(0, 1, len(waypoints))
    t_dense = np.linspace(0, 1, num_steps)

    # Position interpolation
    if method == 'cubic' and len(waypoints) >= 4:
        cs = CubicSpline(t_wp, positions, axis=0)
        dense_positions = cs(t_dense)
    else:
        dense_positions = np.array([
            np.interp(t_dense, t_wp, positions[:, d])
            for d in range(3)
        ]).T

    # Rotation interpolation (always slerp for smooth rotations)
    key_rots = Rotation.concatenate(rotations)
    slerp = Slerp(t_wp, key_rots)
    dense_rotations = slerp(t_dense)

    # Build gripper state sequence
    grip_changes = [wp.get('grip_change', False) for wp in waypoints]
    grip_state = 1  # start open
    grip_states = []
    for i, t in enumerate(t_dense):
        # Find nearest waypoint
        wp_idx = np.argmin(np.abs(t_wp - t))
        if grip_changes[wp_idx] and abs(t - t_wp[wp_idx]) < 0.01:
            grip_state = 1 - grip_state
        grip_states.append(grip_state)

    # Build trajectory
    trajectory = []
    for i in range(num_steps):
        T_we = np.eye(4)
        T_we[:3, :3] = dense_rotations[i].as_matrix()
        T_we[:3, 3] = dense_positions[i]
        trajectory.append({
            'T_we': T_we,
            'grip': grip_states[i],
        })

    return trajectory


# ──────────────────────────────────────────────────────────────────────
# Data augmentation (Appendix D)
# ──────────────────────────────────────────────────────────────────────

def augment_trajectory(trajectory: List[dict],
                       disturbance_prob: float = 0.3,
                       grip_flip_prob: float = 0.1) -> List[dict]:
    """
    Apply data augmentation:
      - 30%: add local disturbances with recovery.
      - 10%: flip gripper state.
    """
    aug = [dict(t) for t in trajectory]

    # Local disturbances
    if np.random.random() < disturbance_prob:
        num_disturb = np.random.randint(1, max(2, len(aug) // 5))
        for _ in range(num_disturb):
            idx = np.random.randint(1, len(aug) - 1)
            T = aug[idx]['T_we'].copy()
            T[:3, 3] += np.random.randn(3) * 0.005  # 5mm perturbation
            perturb = Rotation.from_rotvec(np.random.randn(3) * 0.02).as_matrix()
            T[:3, :3] = perturb @ T[:3, :3]
            aug[idx]['T_we'] = T

    # Gripper flip
    if np.random.random() < grip_flip_prob:
        flip_idx = np.random.randint(0, len(aug))
        aug[flip_idx]['grip'] = 1 - aug[flip_idx]['grip']

    return aug


# ──────────────────────────────────────────────────────────────────────
# Point cloud rendering (placeholder using PyRender)
# ──────────────────────────────────────────────────────────────────────

def render_point_clouds(objects: List[dict],
                        T_we: np.ndarray,
                        num_cameras: int = 3,
                        img_size: int = 256) -> np.ndarray:
    """
    Render segmented point clouds from multiple cameras.

    In a full implementation, this uses PyRender (Matl, 2019) to render
    depth images from 3 cameras, backproject to 3D, and segment by object.

    For now, returns a synthetic point cloud from the object meshes
    transformed to their scene positions.

    Returns: (N, 3) point cloud of object surfaces.
    """
    try:
        import trimesh
        pcds = []
        for obj in objects:
            mesh = trimesh.load(obj['mesh_path'], force='mesh')
            # Transform mesh to scene pose
            mesh.apply_scale(obj['scale'])
            T = np.eye(4)
            T[:3, :3] = obj['rotation']
            T[:3, 3] = obj['position']
            mesh.apply_transform(T)
            # Sample points on surface
            pts = mesh.sample(1024)
            pcds.append(pts)
        if pcds:
            return np.concatenate(pcds, axis=0).astype(np.float32)
    except ImportError:
        pass

    # Fallback: random point cloud around object positions
    pcds = []
    for obj in objects:
        pts = obj['position'] + np.random.randn(1024, 3) * obj['scale']
        pcds.append(pts)
    return np.concatenate(pcds, axis=0).astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Full pseudo-demonstration generation
# ──────────────────────────────────────────────────────────────────────

def generate_pseudo_task(mesh_paths: List[str],
                          num_demos: int = 3) -> List[List[dict]]:
    """
    Generate a complete pseudo-task: multiple demonstrations performing
    the same task with randomised starting poses and object placements.

    Returns: list of num_demos demonstrations, each is a list of
             {pcd: (N,3), T_we: (4,4), grip: int}.
    """
    demos = []
    # Base scene (same objects for all demos of this task)
    base_objects = sample_scene(mesh_paths, num_objects=2)

    # Sample waypoints (same task structure)
    use_biased = np.random.random() < 0.5
    if use_biased:
        base_waypoints = sample_waypoints_biased(base_objects)
    else:
        base_waypoints = sample_waypoints_random(base_objects)

    for _ in range(num_demos):
        # Randomise object poses slightly
        objects = []
        for obj in base_objects:
            new_obj = dict(obj)
            new_obj['position'] = obj['position'] + np.random.randn(3) * 0.02
            perturb = Rotation.from_rotvec(np.random.randn(3) * 0.05).as_matrix()
            new_obj['rotation'] = perturb @ obj['rotation']
            objects.append(new_obj)

        # Adjust waypoints to new object positions
        waypoints = []
        for wp in base_waypoints:
            new_wp = dict(wp)
            new_wp['position'] = wp['position'] + np.random.randn(3) * 0.01
            waypoints.append(new_wp)

        # Interpolate
        trajectory = interpolate_trajectory(waypoints)
        trajectory = augment_trajectory(trajectory)

        # Render point clouds
        demo = []
        for step in trajectory:
            pcd = render_point_clouds(objects, step['T_we'])
            demo.append({
                'pcd': pcd,
                'T_we': step['T_we'].astype(np.float32),
                'grip': step['grip'],
            })
        demos.append(demo)

    return demos


def generate_pseudo_demo_batch(mesh_paths: List[str],
                                batch_size: int = 32,
                                num_demos_range: Tuple[int, int] = (1, 5)
                                ) -> List[dict]:
    """
    Generate a batch of training samples from pseudo-demonstrations.

    Each sample: N-1 demos as context, predict actions for the Nth.
    """
    samples = []
    for _ in range(batch_size):
        N = np.random.randint(num_demos_range[0] + 1, num_demos_range[1] + 2)
        task_demos = generate_pseudo_task(mesh_paths, num_demos=N)

        # Use N-1 demos as context, last demo for action prediction
        context_demos = task_demos[:N - 1]
        target_demo = task_demos[N - 1]

        # Pick a random point in the target demo as "current observation"
        T = min(8, len(target_demo) - 1)
        obs_idx = np.random.randint(0, len(target_demo) - T)

        samples.append({
            'demos': [
                {
                    'pcds': [s['pcd'] for s in demo],
                    'T_w_es': [s['T_we'] for s in demo],
                    'grips': [s['grip'] for s in demo],
                }
                for demo in context_demos
            ],
            'current': {
                'pcd': target_demo[obs_idx]['pcd'],
                'T_w_e': target_demo[obs_idx]['T_we'],
                'grip': target_demo[obs_idx]['grip'],
            },
            'actions': {
                'T_EAs': [
                    np.linalg.inv(target_demo[obs_idx]['T_we']) @
                    target_demo[obs_idx + j + 1]['T_we']
                    for j in range(T)
                ],
                'grips': [target_demo[obs_idx + j + 1]['grip'] for j in range(T)],
                'pcds': [target_demo[obs_idx + j + 1]['pcd'] for j in range(T)],
            },
        })

    return samples
