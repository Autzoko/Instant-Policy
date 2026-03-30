"""
Training dataset for Instant Policy.

Two modes:
  1. On-the-fly pseudo-demonstration generation (for main IP training).
  2. Loading from RLBench demonstrations (for fine-tuning and evaluation).

Implements demo processing / downsampling from Appendix E:
  - Record at 25Hz (sim) or 10Hz (real).
  - Keep start/end, gripper-state-change points, slow-down points.
  - Pad to L waypoints with interpolated intermediates.
"""
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from typing import List, Dict, Optional, Tuple

from .config import IPConfig


# ──────────────────────────────────────────────────────────────────────
# Demo downsampling (Appendix E: Demo Processing)
# ──────────────────────────────────────────────────────────────────────

def downsample_demo(demo: dict, target_length: int = 10) -> dict:
    """
    Downsample a demonstration to target_length waypoints.

    Strategy (Appendix E):
      1. Always include start and end.
      2. Include waypoints where gripper state changed.
      3. Include waypoints where gripper slowed down significantly.
      4. Fill remaining with evenly spaced intermediates.

    demo: {'pcds': [...], 'T_w_es': [...], 'grips': [...]}
    Returns: same format with exactly target_length entries.
    """
    L_orig = len(demo['pcds'])
    if L_orig <= target_length:
        # Pad by repeating last frame
        result = {k: list(v) for k, v in demo.items()}
        while len(result['pcds']) < target_length:
            for k in result:
                result[k].append(result[k][-1])
        return result

    # Select key indices
    key_indices = set()
    key_indices.add(0)
    key_indices.add(L_orig - 1)

    # Gripper state changes
    grips = demo['grips']
    for i in range(1, L_orig):
        g_prev = grips[i - 1] if isinstance(grips[i - 1], (int, float)) else grips[i - 1].item()
        g_curr = grips[i] if isinstance(grips[i], (int, float)) else grips[i].item()
        if g_prev != g_curr:
            key_indices.add(i)

    # Slow-down points (based on velocity)
    if L_orig > 2:
        T_w_es = demo['T_w_es']
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
                if v < mean_vel * 0.3:  # significant slowdown
                    key_indices.add(i + 1)

    # Fill remaining with evenly spaced indices
    key_indices = sorted(key_indices)
    while len(key_indices) < target_length:
        # Find largest gap and insert midpoint
        max_gap = 0
        max_gap_idx = 0
        for i in range(len(key_indices) - 1):
            gap = key_indices[i + 1] - key_indices[i]
            if gap > max_gap:
                max_gap = gap
                max_gap_idx = i
        mid = (key_indices[max_gap_idx] + key_indices[max_gap_idx + 1]) // 2
        key_indices.insert(max_gap_idx + 1, mid)

    # If too many key indices, keep the most important ones
    if len(key_indices) > target_length:
        # Always keep first and last, subsample the rest
        must_keep = {0, L_orig - 1}
        # Keep gripper change points
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
# Pseudo-demo dataset (on-the-fly generation)
# ──────────────────────────────────────────────────────────────────────

class PseudoDemoDataset(IterableDataset):
    """
    Infinite dataset that generates pseudo-demonstrations on-the-fly.
    Each item is a training sample with demos (context) + current obs + actions.
    """

    def __init__(self, shapenet_root: str, cfg: IPConfig = None):
        super().__init__()
        self.cfg = cfg or IPConfig()
        self.shapenet_root = shapenet_root
        self._mesh_paths = None

    @property
    def mesh_paths(self):
        if self._mesh_paths is None:
            from .pseudo_demo import load_shapenet_meshes
            self._mesh_paths = load_shapenet_meshes(self.shapenet_root)
            if not self._mesh_paths:
                raise RuntimeError(
                    f"No .obj meshes found in {self.shapenet_root}")
        return self._mesh_paths

    def __iter__(self):
        from .pseudo_demo import generate_pseudo_demo_batch
        while True:
            samples = generate_pseudo_demo_batch(
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
            d = downsample_demo(demo, cfg.num_traj_waypoints)
            demos.append({
                'pcds': [torch.from_numpy(p).float() for p in d['pcds']],
                'T_w_es': [torch.from_numpy(t).float() for t in d['T_w_es']],
                'grips': [int(g) for g in d['grips']],
            })

        # Current observation
        current = {
            'pcd': torch.from_numpy(sample['current']['pcd']).float(),
            'T_w_e': torch.from_numpy(sample['current']['T_w_e']).float(),
            'grip': int(sample['current']['grip']),
        }

        # Actions
        T = min(cfg.pred_horizon, len(sample['actions']['T_EAs']))
        actions = {
            'T_EAs': torch.stack([
                torch.from_numpy(t).float()
                for t in sample['actions']['T_EAs'][:T]
            ]),
            'grips': torch.tensor(
                sample['actions']['grips'][:T], dtype=torch.float32
            ),
            'pcds': [
                torch.from_numpy(p).float()
                for p in sample['actions']['pcds'][:T]
            ],
        }

        # Data augmentation: flip gripper state (Appendix E)
        if np.random.random() < cfg.gripper_flip_prob:
            current['grip'] = 1 - current['grip']

        return {
            'demos': demos,
            'current': current,
            'actions': actions,
        }


# ──────────────────────────────────────────────────────────────────────
# RLBench dataset (for fine-tuning)
# ──────────────────────────────────────────────────────────────────────

class RLBenchDataset(Dataset):
    """
    Dataset loading demonstrations from RLBench.
    Expects pre-collected demonstrations saved as numpy files.
    """

    def __init__(self, data_dir: str, task_names: List[str] = None,
                 cfg: IPConfig = None):
        super().__init__()
        self.cfg = cfg or IPConfig()
        self.data_dir = data_dir
        self.samples = []
        self._load_data(task_names)

    def _load_data(self, task_names: Optional[List[str]]):
        """Load pre-collected demonstrations."""
        import os
        if task_names is None:
            task_names = [d for d in os.listdir(self.data_dir)
                         if os.path.isdir(os.path.join(self.data_dir, d))]

        for task in task_names:
            task_dir = os.path.join(self.data_dir, task)
            if not os.path.isdir(task_dir):
                continue
            demo_files = sorted([f for f in os.listdir(task_dir)
                                if f.endswith('.npz')])
            for demo_file in demo_files:
                self.samples.append(os.path.join(task_dir, demo_file))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        data = np.load(self.samples[idx], allow_pickle=True)
        # Expects saved format with keys: pcds, T_w_es, grips
        demo = {
            'pcds': list(data['pcds']),
            'T_w_es': list(data['T_w_es']),
            'grips': list(data['grips']),
        }
        demo = downsample_demo(demo, self.cfg.num_traj_waypoints)
        return {
            'pcds': [torch.from_numpy(p).float() for p in demo['pcds']],
            'T_w_es': [torch.from_numpy(t).float() for t in demo['T_w_es']],
            'grips': [int(g) for g in demo['grips']],
        }
