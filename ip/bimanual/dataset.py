"""
Bimanual training datasets.

Two modes:
  1. On-the-fly bimanual pseudo-demonstration generation (Phase 2 training).
  2. Loading from PerAct2 bimanual demonstrations (fine-tuning / evaluation).

Demo downsampling is extended to handle bimanual data: key indices are
selected based on the union of important events from BOTH arms (gripper
state changes, velocity slowdowns) so that both arms stay synchronised.
"""
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from typing import List, Dict, Optional

from ..config import IPConfig
from .config import BimanualIPConfig


# ──────────────────────────────────────────────────────────────────────
# Bimanual demo downsampling
# ──────────────────────────────────────────────────────────────────────

def downsample_bimanual_demo(demo: dict, target_length: int = 10) -> dict:
    """
    Downsample a bimanual demonstration to target_length waypoints.

    Key indices are chosen from the UNION of important events across
    both arms to keep them synchronised:
      1. Always include start and end.
      2. Include waypoints where EITHER arm's gripper state changed.
      3. Include waypoints where EITHER arm slowed down significantly.
      4. Fill remaining with evenly spaced intermediates.

    demo: {
      'pcds': [...],
      'T_w_es_left': [...], 'T_w_es_right': [...],
      'grips_left': [...],  'grips_right': [...],
    }
    Returns: same format with exactly target_length entries.
    """
    L_orig = len(demo['pcds'])
    if L_orig <= target_length:
        result = {k: list(v) for k, v in demo.items()}
        while len(result['pcds']) < target_length:
            for k in result:
                result[k].append(result[k][-1])
        return result

    key_indices = set()
    key_indices.add(0)
    key_indices.add(L_orig - 1)

    # Gripper state changes on EITHER arm
    for arm in ('left', 'right'):
        grips = demo[f'grips_{arm}']
        for i in range(1, L_orig):
            g_prev = grips[i - 1] if isinstance(grips[i - 1], (int, float)) else grips[i - 1].item()
            g_curr = grips[i] if isinstance(grips[i], (int, float)) else grips[i].item()
            if g_prev != g_curr:
                key_indices.add(i)

    # Velocity slowdowns on EITHER arm
    if L_orig > 2:
        for arm in ('left', 'right'):
            T_w_es = demo[f'T_w_es_{arm}']
            velocities = []
            for i in range(1, L_orig):
                t_prev = T_w_es[i - 1]
                t_curr = T_w_es[i]
                if isinstance(t_prev, np.ndarray):
                    diff = np.linalg.norm(t_curr[:3, 3] - t_prev[:3, 3])
                else:
                    diff = (t_curr[:3, 3] - t_prev[:3, 3]).norm().item()
                velocities.append(diff)

            if velocities:
                mean_vel = np.mean(velocities)
                for i, v in enumerate(velocities):
                    if v < mean_vel * 0.3:
                        key_indices.add(i + 1)

    # Fill remaining with evenly spaced intermediates
    key_indices = sorted(key_indices)
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

    # Trim if too many
    if len(key_indices) > target_length:
        must_keep = {0, L_orig - 1}
        for arm in ('left', 'right'):
            grips = demo[f'grips_{arm}']
            for i in range(1, L_orig):
                g_prev = grips[i - 1] if isinstance(grips[i - 1], (int, float)) else grips[i - 1].item()
                g_curr = grips[i] if isinstance(grips[i], (int, float)) else grips[i].item()
                if g_prev != g_curr:
                    must_keep.add(i)
        remaining = [idx for idx in key_indices if idx not in must_keep]
        needed = target_length - len(must_keep)
        if needed > 0 and remaining:
            step = max(1, len(remaining) // needed)
            selected = remaining[::step][:needed]
            key_indices = sorted(must_keep | set(selected))
        else:
            key_indices = sorted(must_keep)

    key_indices = key_indices[:target_length]

    result = {}
    for k in demo:
        result[k] = [demo[k][i] for i in key_indices]
    return result


# ──────────────────────────────────────────────────────────────────────
# Bimanual pseudo-demo dataset (on-the-fly generation)
# ──────────────────────────────────────────────────────────────────────

class BimanualPseudoDemoDataset(IterableDataset):
    """
    Infinite dataset generating bimanual pseudo-demonstrations on-the-fly.
    """

    def __init__(self, shapenet_root: str, cfg: BimanualIPConfig = None):
        super().__init__()
        self.cfg = cfg or BimanualIPConfig()
        self.shapenet_root = shapenet_root
        self._mesh_paths = None

    @property
    def mesh_paths(self):
        if self._mesh_paths is None:
            from ..pseudo_demo import load_shapenet_meshes
            import time
            print("  Loading ShapeNet mesh paths (this may take a few minutes)...")
            t0 = time.time()
            self._mesh_paths = load_shapenet_meshes(self.shapenet_root,
                                                       max_objects=50000)
            print(f"  Loaded {len(self._mesh_paths)} meshes in {time.time()-t0:.1f}s")
            if not self._mesh_paths:
                raise RuntimeError(
                    f"No .obj meshes found in {self.shapenet_root}")
        return self._mesh_paths

    def __iter__(self):
        from .pseudo_demo import generate_bimanual_pseudo_demo_batch
        while True:
            samples = generate_bimanual_pseudo_demo_batch(
                self.mesh_paths, batch_size=1,
                num_demos_range=(1, self.cfg.max_demos),
            )
            for sample in samples:
                yield self._to_tensors(sample)

    def _to_tensors(self, sample: dict) -> dict:
        """Convert numpy arrays to tensors."""
        cfg = self.cfg

        # Process demos
        demos = []
        for demo in sample['demos']:
            d = downsample_bimanual_demo(demo, cfg.num_traj_waypoints)
            demos.append({
                'pcds': [torch.from_numpy(p).float() for p in d['pcds']],
                'T_w_es_left': [torch.from_numpy(t).float() for t in d['T_w_es_left']],
                'T_w_es_right': [torch.from_numpy(t).float() for t in d['T_w_es_right']],
                'grips_left': [int(g) for g in d['grips_left']],
                'grips_right': [int(g) for g in d['grips_right']],
            })

        # Current observation
        current = {
            'pcd': torch.from_numpy(sample['current']['pcd']).float(),
            'T_w_e_left': torch.from_numpy(sample['current']['T_w_e_left']).float(),
            'T_w_e_right': torch.from_numpy(sample['current']['T_w_e_right']).float(),
            'grip_left': int(sample['current']['grip_left']),
            'grip_right': int(sample['current']['grip_right']),
        }

        # Actions
        T = min(cfg.pred_horizon, len(sample['actions']['T_EAs_left']))
        actions = {
            'T_EAs_left': torch.stack([
                torch.from_numpy(t).float()
                for t in sample['actions']['T_EAs_left'][:T]
            ]),
            'T_EAs_right': torch.stack([
                torch.from_numpy(t).float()
                for t in sample['actions']['T_EAs_right'][:T]
            ]),
            'grips_left': torch.tensor(
                sample['actions']['grips_left'][:T], dtype=torch.float32),
            'grips_right': torch.tensor(
                sample['actions']['grips_right'][:T], dtype=torch.float32),
        }

        # Data augmentation: flip gripper state (independently per arm)
        if np.random.random() < cfg.gripper_flip_prob:
            current['grip_left'] = 1 - current['grip_left']
        if np.random.random() < cfg.gripper_flip_prob:
            current['grip_right'] = 1 - current['grip_right']

        return {
            'demos': demos,
            'current': current,
            'actions': actions,
        }


# ──────────────────────────────────────────────────────────────────────
# PerAct2 bimanual dataset (for fine-tuning / evaluation)
# ──────────────────────────────────────────────────────────────────────

class PerAct2Dataset(Dataset):
    """
    Dataset loading bimanual demonstrations from PerAct2 / RLBench bimanual.

    Expects pre-collected demonstrations saved as numpy files with keys:
      pcds, T_w_es_left, T_w_es_right, grips_left, grips_right

    The dataset can also load from PerAct2's raw format:
      - Each episode directory contains timestep folders
      - Each timestep has: front_rgb, front_depth, front_point_cloud,
        left_shoulder_*, right_shoulder_*, wrist_*, etc.
      - Joint states and gripper states from low_dim_obs.pkl
    """

    def __init__(self, data_dir: str, task_names: List[str] = None,
                 cfg: BimanualIPConfig = None,
                 data_format: str = 'npz'):
        """
        Args:
            data_dir: root directory containing task folders
            task_names: list of task names to load (None = all)
            cfg: bimanual config
            data_format: 'npz' for pre-processed, 'peract2' for raw format
        """
        super().__init__()
        self.cfg = cfg or BimanualIPConfig()
        self.data_dir = data_dir
        self.data_format = data_format
        self.samples = []
        self._load_data(task_names)

    def _load_data(self, task_names: Optional[List[str]]):
        """Load demonstration file paths."""
        import os

        if task_names is None:
            task_names = [d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d))]

        for task in task_names:
            task_dir = os.path.join(self.data_dir, task)
            if not os.path.isdir(task_dir):
                continue

            if self.data_format == 'npz':
                demo_files = sorted([
                    f for f in os.listdir(task_dir) if f.endswith('.npz')
                ])
                for demo_file in demo_files:
                    self.samples.append({
                        'path': os.path.join(task_dir, demo_file),
                        'task': task,
                        'format': 'npz',
                    })
            elif self.data_format == 'peract2':
                # PerAct2 raw format: task_dir/episodes/episode{N}/
                episodes_dir = os.path.join(task_dir, 'all_variations',
                                            'episodes')
                if not os.path.isdir(episodes_dir):
                    episodes_dir = task_dir
                episode_dirs = sorted([
                    d for d in os.listdir(episodes_dir)
                    if os.path.isdir(os.path.join(episodes_dir, d))
                       and d.startswith('episode')
                ])
                for ep_dir in episode_dirs:
                    self.samples.append({
                        'path': os.path.join(episodes_dir, ep_dir),
                        'task': task,
                        'format': 'peract2',
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        if info['format'] == 'npz':
            return self._load_npz(info['path'])
        else:
            return self._load_peract2(info['path'], info['task'])

    def _load_npz(self, path: str) -> dict:
        """Load pre-processed bimanual demonstration."""
        data = np.load(path, allow_pickle=True)
        demo = {
            'pcds': list(data['pcds']),
            'T_w_es_left': list(data['T_w_es_left']),
            'T_w_es_right': list(data['T_w_es_right']),
            'grips_left': list(data['grips_left']),
            'grips_right': list(data['grips_right']),
        }
        demo = downsample_bimanual_demo(demo, self.cfg.num_traj_waypoints)
        return {
            'pcds': [torch.from_numpy(p).float() for p in demo['pcds']],
            'T_w_es_left': [torch.from_numpy(t).float()
                            for t in demo['T_w_es_left']],
            'T_w_es_right': [torch.from_numpy(t).float()
                             for t in demo['T_w_es_right']],
            'grips_left': [int(g) for g in demo['grips_left']],
            'grips_right': [int(g) for g in demo['grips_right']],
        }

    def _load_peract2(self, episode_dir: str, task: str) -> dict:
        """
        Load a bimanual demonstration from PerAct2's raw format.

        PerAct2 episode structure:
          episode_dir/
            low_dim_obs.pkl    -- proprioceptive data (joint angles, EE poses,
                                  gripper states for both arms)
            {timestep}/
              front_point_cloud.npy
              left_shoulder_point_cloud.npy
              right_shoulder_point_cloud.npy
              ...

        The EE poses and gripper states are extracted from low_dim_obs.pkl.
        Point clouds are merged from multiple camera views.
        """
        import os
        import pickle

        # Load low-dimensional observations
        low_dim_path = os.path.join(episode_dir, 'low_dim_obs.pkl')
        with open(low_dim_path, 'rb') as f:
            low_dim_obs = pickle.load(f)

        # Extract per-timestep data
        pcds = []
        T_w_es_left = []
        T_w_es_right = []
        grips_left = []
        grips_right = []

        # PerAct2 stores observations as a list of dicts or an object
        # with attributes per timestep
        if isinstance(low_dim_obs, list):
            timesteps = low_dim_obs
        elif hasattr(low_dim_obs, '__iter__'):
            timesteps = list(low_dim_obs)
        else:
            # Single observation object with arrays
            timesteps = [low_dim_obs]

        for t_idx, obs in enumerate(timesteps):
            # Extract EE poses (4x4 matrices)
            if isinstance(obs, dict):
                T_l = obs.get('left_ee_pose',
                             obs.get('gripper_left_pose',
                                     np.eye(4, dtype=np.float32)))
                T_r = obs.get('right_ee_pose',
                             obs.get('gripper_right_pose',
                                     np.eye(4, dtype=np.float32)))
                g_l = obs.get('left_gripper_open',
                             obs.get('gripper_left_open', 1))
                g_r = obs.get('right_gripper_open',
                             obs.get('gripper_right_open', 1))
            else:
                # Object with attributes
                T_l = getattr(obs, 'left_ee_pose',
                             getattr(obs, 'gripper_left_pose',
                                     np.eye(4, dtype=np.float32)))
                T_r = getattr(obs, 'right_ee_pose',
                             getattr(obs, 'gripper_right_pose',
                                     np.eye(4, dtype=np.float32)))
                g_l = getattr(obs, 'left_gripper_open',
                             getattr(obs, 'gripper_left_open', 1))
                g_r = getattr(obs, 'right_gripper_open',
                             getattr(obs, 'gripper_right_open', 1))

            # Convert 7D pose (pos + quat) to 4x4 if needed
            if isinstance(T_l, np.ndarray) and T_l.shape == (7,):
                T_l = _pose7_to_mat(T_l)
            if isinstance(T_r, np.ndarray) and T_r.shape == (7,):
                T_r = _pose7_to_mat(T_r)

            T_w_es_left.append(np.array(T_l, dtype=np.float32))
            T_w_es_right.append(np.array(T_r, dtype=np.float32))
            grips_left.append(int(g_l > 0.5) if isinstance(g_l, float) else int(g_l))
            grips_right.append(int(g_r > 0.5) if isinstance(g_r, float) else int(g_r))

            # Load and merge point clouds from available cameras
            ts_dir = os.path.join(episode_dir, str(t_idx))
            pcd = self._load_merged_pcd(ts_dir)
            pcds.append(pcd)

        demo = {
            'pcds': pcds,
            'T_w_es_left': T_w_es_left,
            'T_w_es_right': T_w_es_right,
            'grips_left': grips_left,
            'grips_right': grips_right,
        }
        demo = downsample_bimanual_demo(demo, self.cfg.num_traj_waypoints)
        return {
            'pcds': [torch.from_numpy(p).float() for p in demo['pcds']],
            'T_w_es_left': [torch.from_numpy(t).float()
                            for t in demo['T_w_es_left']],
            'T_w_es_right': [torch.from_numpy(t).float()
                             for t in demo['T_w_es_right']],
            'grips_left': [int(g) for g in demo['grips_left']],
            'grips_right': [int(g) for g in demo['grips_right']],
        }

    def _load_merged_pcd(self, timestep_dir: str) -> np.ndarray:
        """
        Load and merge point clouds from multiple cameras.
        Falls back to random points if files are missing.
        """
        import os
        pcd_files = [
            'front_point_cloud.npy',
            'left_shoulder_point_cloud.npy',
            'right_shoulder_point_cloud.npy',
            'overhead_point_cloud.npy',
            'wrist_point_cloud.npy',
            'wrist_left_point_cloud.npy',
            'wrist_right_point_cloud.npy',
        ]
        pcds = []
        for pcd_file in pcd_files:
            path = os.path.join(timestep_dir, pcd_file)
            if os.path.exists(path):
                pts = np.load(path)
                if pts.ndim == 3:
                    # (H, W, 3) -> (H*W, 3)
                    pts = pts.reshape(-1, 3)
                # Filter invalid points
                valid = np.isfinite(pts).all(axis=1) & (np.abs(pts) < 5.0).all(axis=1)
                pts = pts[valid]
                if len(pts) > 0:
                    pcds.append(pts)

        if pcds:
            merged = np.concatenate(pcds, axis=0)
        else:
            # Fallback
            merged = np.random.randn(2048, 3).astype(np.float32) * 0.1

        # Subsample to target size
        target = self.cfg.num_pcd_points
        if len(merged) > target:
            idx = np.random.choice(len(merged), target, replace=False)
            merged = merged[idx]
        elif len(merged) < target:
            # Pad by repeating
            pad_idx = np.random.choice(len(merged), target - len(merged), replace=True)
            merged = np.concatenate([merged, merged[pad_idx]], axis=0)

        return merged.astype(np.float32)


def _pose7_to_mat(pose7: np.ndarray) -> np.ndarray:
    """Convert [x, y, z, qx, qy, qz, qw] to 4x4 matrix."""
    from scipy.spatial.transform import Rotation
    T = np.eye(4, dtype=np.float32)
    T[:3, 3] = pose7[:3]
    T[:3, :3] = Rotation.from_quat(pose7[3:]).as_matrix()
    return T
