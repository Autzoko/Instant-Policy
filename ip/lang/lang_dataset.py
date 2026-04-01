"""
Language-annotated dataset for modality transfer training (Appendix J).

Data sources:
  1. RLBench demonstrations with language annotations
     (each task has natural-language variation descriptions).
  2. Successful rollouts from the pre-trained IP model, labelled with
     the task language description.

Each sample provides:
  - Language description (text string)
  - Current observation (point cloud + gripper pose/state)
  - Demonstrations (for computing the target bottleneck via frozen φ)
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from typing import List, Dict, Optional, Tuple

from ..config import IPConfig
from ..dataset import downsample_demo


# ──────────────────────────────────────────────────────────────────────
# RLBench task descriptions (subset used in the paper)
# ──────────────────────────────────────────────────────────────────────

RLBENCH_TASK_DESCRIPTIONS = {
    'open_box': 'open the box',
    'close_box': 'close the box',
    'close_jar': 'close the jar',
    'close_microwave': 'close the microwave',
    'open_microwave': 'open the microwave',
    'toilet_seat_down': 'put the toilet seat down',
    'toilet_seat_up': 'put the toilet seat up',
    'toilet_roll_off': 'take the toilet roll off',
    'phone_on_base': 'put the phone on the base',
    'push_button': 'push the button',
    'lift_lid': 'lift the lid',
    'slide_block': 'slide the block',
    'basketball': 'put the ball in the hoop',
    'lamp_on': 'turn on the lamp',
    'put_rubbish': 'put the rubbish in the bin',
    'umbrella_out': 'take the umbrella out of the stand',
    'buzz': 'press the buzzer',
    'plate_out': 'take the plate out',
    'close_laptop': 'close the laptop',
    'open_drawer': 'open the drawer',
    'meat_off_grill': 'take the meat off the grill',
    'flip_switch': 'flip the switch',
    'paper_roll_off': 'take the paper roll off',
    'put_umbrella': 'put the umbrella in the stand',
}


class LangAnnotatedDataset(Dataset):
    """
    Dataset for training φ_lang.

    Each item contains:
      - text:     language description of the task
      - current:  {pcd, T_w_e, grip}  a random observation from the demo
      - demos:    list of demonstrations (for bottleneck target computation)

    The target bottleneck is computed on-the-fly by the frozen IP model
    during training (not stored in the dataset).
    """

    def __init__(self, data_dir: str,
                 task_descriptions: Dict[str, str] = None,
                 cfg: IPConfig = None,
                 demos_per_task: int = 20,
                 num_context_demos: int = 2):
        super().__init__()
        self.cfg = cfg or IPConfig()
        self.data_dir = data_dir
        self.task_descs = task_descriptions or RLBENCH_TASK_DESCRIPTIONS
        self.num_context_demos = num_context_demos
        self.samples = []
        self._load_data(demos_per_task)

    def _load_data(self, demos_per_task: int):
        """
        Load demonstrations organised by task.
        Expected directory structure:
          data_dir/
            task_name/
              demo_0.npz
              demo_1.npz
              ...
        """
        for task_name, description in self.task_descs.items():
            task_dir = os.path.join(self.data_dir, task_name)
            if not os.path.isdir(task_dir):
                continue
            demo_files = sorted([
                f for f in os.listdir(task_dir) if f.endswith('.npz')
            ])[:demos_per_task]

            # Each demo file becomes multiple training samples
            # (different "current" timesteps)
            demo_paths = [os.path.join(task_dir, f) for f in demo_files]
            if len(demo_paths) < self.num_context_demos + 1:
                continue

            for i in range(len(demo_paths)):
                self.samples.append({
                    'task': task_name,
                    'text': description,
                    'demo_paths': demo_paths,
                    'target_idx': i,
                })

        print(f"LangAnnotatedDataset: {len(self.samples)} samples "
              f"from {len(set(s['task'] for s in self.samples))} tasks")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> dict:
        sample = self.samples[idx]
        cfg = self.cfg

        # Load all demos for this task
        all_demo_paths = sample['demo_paths']
        target_idx = sample['target_idx']

        # Context demos: exclude the target demo
        context_indices = [i for i in range(len(all_demo_paths))
                          if i != target_idx]
        np.random.shuffle(context_indices)
        context_indices = context_indices[:self.num_context_demos]

        # Load and process context demos
        demos = []
        for ci in context_indices:
            data = np.load(all_demo_paths[ci], allow_pickle=True)
            demo = {
                'pcds': list(data['pcds']),
                'T_w_es': list(data['T_w_es']),
                'grips': list(data['grips']),
            }
            demo = downsample_demo(demo, cfg.num_traj_waypoints)
            demos.append({
                'pcds': [torch.from_numpy(p).float() for p in demo['pcds']],
                'T_w_es': [torch.from_numpy(t).float() for t in demo['T_w_es']],
                'grips': [int(g) for g in demo['grips']],
            })

        # Load target demo for current observation
        target_data = np.load(all_demo_paths[target_idx], allow_pickle=True)
        target_demo = {
            'pcds': list(target_data['pcds']),
            'T_w_es': list(target_data['T_w_es']),
            'grips': list(target_data['grips']),
        }

        # Pick a random timestep as "current observation"
        obs_idx = np.random.randint(0, len(target_demo['pcds']))
        current = {
            'pcd': torch.from_numpy(target_demo['pcds'][obs_idx]).float(),
            'T_w_e': torch.from_numpy(target_demo['T_w_es'][obs_idx]).float(),
            'grip': int(target_demo['grips'][obs_idx]),
        }

        return {
            'text': sample['text'],
            'current': current,
            'demos': demos,
        }


# ──────────────────────────────────────────────────────────────────────
# Utility: collect RLBench demonstrations with language
# ──────────────────────────────────────────────────────────────────────

def collect_rlbench_lang_data(save_dir: str,
                                task_names: List[str] = None,
                                demos_per_task: int = 20,
                                headless: bool = True):
    """
    Collect demonstrations from RLBench and save with language annotations.

    Requires RLBench and CoppeliaSim to be installed.
    """
    try:
        from rlbench.environment import Environment
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaIK
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.observation_config import ObservationConfig
    except ImportError:
        print("RLBench not installed. Cannot collect demonstrations.")
        return

    if task_names is None:
        task_names = list(RLBENCH_TASK_DESCRIPTIONS.keys())

    obs_config = ObservationConfig()
    obs_config.set_all(True)

    action_mode = MoveArmThenGripper(
        arm_action_mode=EndEffectorPoseViaIK(),
        gripper_action_mode=Discrete()
    )

    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=headless,
    )
    env.launch()

    for task_name in task_names:
        task_dir = os.path.join(save_dir, task_name)
        os.makedirs(task_dir, exist_ok=True)

        try:
            # Convert snake_case task name to CamelCase class name
            # e.g. 'close_microwave' -> 'CloseMicrowave'
            class_name = ''.join(w.capitalize() for w in task_name.split('_'))
            task_module = __import__(f'rlbench.tasks.{task_name}',
                                     fromlist=[class_name])
            task_class = getattr(task_module, class_name)
            task = env.get_task(task_class)
        except Exception as e:
            print(f"Skipping {task_name}: {e}")
            continue

        # Get task descriptions
        descriptions, _ = task.reset()
        print(f"Task: {task_name} — '{descriptions[0]}'")

        for demo_idx in range(demos_per_task):
            try:
                demo = task.get_demos(1, live_demos=True)[0]
                pcds = []
                T_w_es = []
                grips = []

                for obs in demo:
                    # Combine point clouds from 3 cameras
                    pcd_parts = []
                    for cam in ['front', 'left_shoulder', 'right_shoulder']:
                        depth = getattr(obs, f'{cam}_depth')
                        mask = getattr(obs, f'{cam}_mask')
                        pcd_cam = getattr(obs, f'{cam}_point_cloud')
                        # Filter by mask
                        valid = mask.flatten() > 60
                        if valid.any():
                            pcd_parts.append(
                                pcd_cam.reshape(-1, 3)[valid]
                            )
                    if pcd_parts:
                        pcd = np.concatenate(pcd_parts, axis=0)
                    else:
                        pcd = np.zeros((100, 3), dtype=np.float32)

                    # Subsample to 2048
                    if pcd.shape[0] > 2048:
                        idx = np.random.choice(pcd.shape[0], 2048, replace=False)
                        pcd = pcd[idx]
                    elif pcd.shape[0] < 2048:
                        pad = np.random.choice(pcd.shape[0], 2048 - pcd.shape[0])
                        pcd = np.concatenate([pcd, pcd[pad]], axis=0)

                    pcds.append(pcd.astype(np.float32))
                    T_w_es.append(obs.gripper_matrix.astype(np.float32))
                    grips.append(int(obs.gripper_open > 0.5))

                save_path = os.path.join(task_dir, f'demo_{demo_idx}.npz')
                np.savez_compressed(save_path,
                                    pcds=np.array(pcds),
                                    T_w_es=np.array(T_w_es),
                                    grips=np.array(grips))
                print(f"  Saved demo {demo_idx}")

            except Exception as e:
                print(f"  Demo {demo_idx} failed: {e}")

    env.shutdown()
    print(f"Data collection complete. Saved to {save_dir}")
