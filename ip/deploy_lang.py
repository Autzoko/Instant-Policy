"""
Language-guided deployment script.

Instead of providing demonstrations, the user gives a natural language
description of the task.  φ_lang produces the bottleneck, and the frozen
ψ predicts actions via the standard diffusion denoising loop.

Usage:
    python -m ip.deploy_lang \
        --ip_checkpoint checkpoints_ip/model_final.pt \
        --lang_checkpoint checkpoints_lang/phi_lang_final.pt \
        --task "close the microwave"
"""
import torch
import numpy as np
import argparse
from typing import Tuple

from .config import IPConfig
from .model import GraphDiffusionPolicy
from .lang.phi_lang import PhiLang
from .lang.encoder import LanguageEncoder
from .geometry_encoder import GeometryEncoder
from .graph_builder import (
    GripperNodeEncoder, build_local_graph, build_action_edges,
    gripper_keypoints_ee, gripper_keypoints_world,
    DEFAULT_GRIPPER_KEYPOINTS, _fully_connected_edges, _make_edge_attr,
)
from .networks import PsiNetwork, DiffusionStepEmbedding
from .diffusion import NoiseSchedule, ddim_reverse_step
from .se3_utils import invert_se3, transform_points


class LanguageGuidedPolicy:
    """
    Deployment wrapper for language-guided Instant Policy.

    Replaces the demo-based pipeline with:
      language description + current observation → φ_lang → bottleneck → ψ → actions
    """

    def __init__(self, ip_checkpoint: str, lang_checkpoint: str,
                 device: str = 'cuda'):
        self.device = device

        # Load IP model (we only need geo_encoder, gripper_encoder, σ, ψ)
        ckpt = torch.load(ip_checkpoint, map_location=device)
        cfg = ckpt.get('cfg', IPConfig())
        self.cfg = cfg

        self.ip_model = GraphDiffusionPolicy(cfg).to(device)
        if 'model' in ckpt:
            self.ip_model.load_state_dict(ckpt['model'])
        else:
            self.ip_model.load_state_dict(ckpt)
        self.ip_model.eval()

        # Freeze everything in IP model
        for param in self.ip_model.parameters():
            param.requires_grad = False

        # Load φ_lang
        lang_ckpt = torch.load(lang_checkpoint, map_location=device)
        self.phi_lang = PhiLang(cfg).to(device)
        self.phi_lang.load_state_dict(lang_ckpt['phi_lang'])
        self.phi_lang.eval()

        # Inference noise schedule
        self.schedule = NoiseSchedule(
            cfg.num_diffusion_steps_infer * 5,
            cfg.beta_start, cfg.beta_end, device
        )

        self.kp = DEFAULT_GRIPPER_KEYPOINTS.to(device)

    @torch.no_grad()
    def get_bottleneck(self, task_description: str,
                       pcd: np.ndarray,
                       T_w_e: np.ndarray,
                       grip: int) -> torch.Tensor:
        """
        Compute bottleneck from language + current observation.

        Args:
            task_description: e.g. "close the microwave"
            pcd:    (N, 3) segmented point cloud in world frame
            T_w_e:  (4, 4) end-effector pose
            grip:   0 (closed) or 1 (open)

        Returns: (K, hidden_dim) bottleneck tensor
        """
        device = self.device
        cfg = self.cfg
        kp = self.kp

        pcd_t = torch.from_numpy(pcd).float().to(device)
        T_we_t = torch.from_numpy(T_w_e).float().to(device)
        grip_t = torch.tensor(grip, device=device)

        # Point cloud → EE frame
        T_ew = invert_se3(T_we_t.unsqueeze(0)).squeeze(0)
        pcd_ee = transform_points(T_ew.unsqueeze(0), pcd_t.unsqueeze(0)).squeeze(0)

        # Geometry encoder
        centroids, feat = self.ip_model._encode_pcd(pcd_ee.unsqueeze(0))
        centroids, feat = centroids.squeeze(0), feat.squeeze(0)

        # Gripper features
        gripper_feat = self.ip_model.gripper_encoder(grip_t.unsqueeze(0))[0]

        # Language embedding
        lang_feat = self.phi_lang.lang_encoder(
            texts=[task_description], device=device
        )

        # Build edges
        M, K = centroids.shape[0], kp.shape[0]
        ei = _fully_connected_edges(M, K, device=device)
        ea = _make_edge_attr(centroids, kp, ei[0], ei[1], cfg.edge_freq_bands)

        # φ_lang → bottleneck
        bottleneck = self.phi_lang(
            scene_feat=feat,
            scene_pos=centroids,
            gripper_feat=gripper_feat,
            gripper_pos=kp,
            lang_feat=lang_feat,
            scene_gripper_edge_index=ei,
            scene_gripper_edge_attr=ea,
        )
        return bottleneck

    @torch.no_grad()
    def predict_actions(self, task_description: str,
                         pcd: np.ndarray,
                         T_w_e: np.ndarray,
                         grip: int,
                         num_diffusion_steps: int = None,
                         ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full inference: language + observation → actions.

        Args:
            task_description: e.g. "close the microwave"
            pcd:    (N, 3) segmented point cloud (world frame)
            T_w_e:  (4, 4) current end-effector pose
            grip:   0 or 1

        Returns:
            actions: (T, 4, 4) relative end-effector transforms
            grips:   (T,) gripper commands (0=close, 1=open)
        """
        cfg = self.cfg
        K = cfg.num_gripper_keypoints
        T_pred = cfg.pred_horizon
        device = self.device
        kp = self.kp
        num_steps = num_diffusion_steps or cfg.num_diffusion_steps_infer

        # 1. Get bottleneck from language
        bottleneck = self.get_bottleneck(task_description, pcd, T_w_e, grip)

        # 2. Current gripper world positions
        T_we_t = torch.from_numpy(T_w_e).float().to(device)
        cur_g_pos = gripper_keypoints_world(T_we_t.unsqueeze(0), kp).squeeze(0)

        # 3. Initialise noisy action positions
        action_pos = cur_g_pos.unsqueeze(0).expand(T_pred, -1, -1) + \
            torch.randn(T_pred, K, 3, device=device) * 0.01
        action_pos = action_pos.reshape(T_pred * K, 3)
        grip_pred = torch.randn(T_pred, device=device)

        # 4. Denoising loop
        step_indices = torch.linspace(
            self.schedule.num_steps - 1, 0, num_steps, device=device
        ).long()
        T_accum = torch.eye(4, device=device).unsqueeze(0).expand(T_pred, -1, -1).clone()

        pcd_t = torch.from_numpy(pcd).float().to(device)
        T_ew = invert_se3(T_we_t.unsqueeze(0)).squeeze(0)
        pcd_ee = transform_points(T_ew.unsqueeze(0), pcd_t.unsqueeze(0)).squeeze(0)

        for i, k_idx in enumerate(step_indices):
            k = k_idx.item()

            # Build action subgraphs with current noisy positions
            action_feats_list = []
            for t in range(T_pred):
                T_ea_est = T_accum[t]
                T_ea_inv = invert_se3(T_ea_est.unsqueeze(0)).squeeze(0)
                pcd_action = transform_points(
                    T_ea_inv.unsqueeze(0), pcd_ee.unsqueeze(0)
                ).squeeze(0)
                cent, feat = self.ip_model._encode_pcd(pcd_action.unsqueeze(0))
                cent, feat = cent.squeeze(0), feat.squeeze(0)

                grip_t_int = torch.tensor(int((grip_pred[t] > 0).item()), device=device)
                g_feat = self.ip_model._build_and_run_sigma(
                    cent, feat, kp, grip_t_int
                )
                action_feats_list.append(g_feat)

            action_gripper_feats = torch.cat(action_feats_list, dim=0)

            # Run ψ
            flow = self.ip_model._run_psi_single_step(
                bottleneck, action_gripper_feats,
                cur_g_pos, action_pos, k
            )
            flow_reshaped = flow.reshape(T_pred, K, 7)

            # DDIM step
            action_pos_reshaped = action_pos.reshape(T_pred, K, 3)
            T_step, action_pos_new, grip_pred = ddim_reverse_step(
                action_pos_reshaped, flow_reshaped, grip_pred,
                k, self.schedule, kp
            )
            action_pos = action_pos_new.reshape(T_pred * K, 3)
            T_accum = T_step @ T_accum

        # 5. Output
        actions = T_accum.cpu().numpy()
        grips = (grip_pred > 0).float().cpu().numpy()
        return actions, grips


# ──────────────────────────────────────────────────────────────────────
# CLI demo
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Language-guided Instant Policy deployment'
    )
    parser.add_argument('--ip_checkpoint', type=str, required=True)
    parser.add_argument('--lang_checkpoint', type=str, required=True)
    parser.add_argument('--task', type=str, required=True,
                       help='Task description, e.g. "close the microwave"')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    policy = LanguageGuidedPolicy(
        args.ip_checkpoint, args.lang_checkpoint, args.device
    )

    # Example: create dummy observation (replace with real robot data)
    print(f"\nLanguage-guided policy ready.")
    print(f"Task: '{args.task}'")
    print(f"Waiting for observations...\n")

    # Dummy observation for demonstration
    pcd = np.random.randn(2048, 3).astype(np.float32) * 0.1
    T_w_e = np.eye(4, dtype=np.float32)
    T_w_e[:3, 3] = [0.4, 0.0, 0.3]  # example gripper position
    grip = 1  # open

    actions, grips = policy.predict_actions(args.task, pcd, T_w_e, grip)
    print(f"Predicted {len(actions)} actions:")
    for i, (a, g) in enumerate(zip(actions, grips)):
        pos = a[:3, 3]
        g_str = "OPEN" if g > 0.5 else "CLOSE"
        print(f"  Step {i}: delta_pos=({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})  "
              f"gripper={g_str}")


if __name__ == '__main__':
    main()
