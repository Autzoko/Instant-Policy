"""
Full GraphDiffusionPolicy model (Section 3, Appendix C/E).

Integrates:
  - Geometry encoder φ_e (frozen, pre-trained)
  - GripperNodeEncoder
  - σ, φ, ψ networks
  - SE(3) diffusion process

Training forward pass:
  1. Encode all point clouds (demos, current, actions) with φ_e.
  2. Build local graphs, process with σ.
  3. Build context graph, process with φ → bottleneck.
  4. Add noise to ground-truth actions → build action local graphs → σ.
  5. Build action graph, combine with bottleneck → ψ → flow predictions.
  6. Compute loss against ground-truth flow.

Inference:
  1-3. Same as above (using demonstrations as context).
  4.   Sample noisy actions from N(0,I).
  5.   Iteratively denoise using σ→ψ loop.
  6.   Extract SE(3) actions via SVD.
"""
import torch
import torch.nn as nn
import torch.nn.functional as TF
from typing import Dict, List, Tuple, Optional

from .config import IPConfig
from .geometry_encoder import GeometryEncoder
from .graph_builder import (
    GripperNodeEncoder, build_local_graph, build_context_edges,
    build_action_edges, gripper_keypoints_world, gripper_keypoints_ee,
    DEFAULT_GRIPPER_KEYPOINTS,
)
from .networks import SigmaNetwork, PhiNetwork, PsiNetwork, DiffusionStepEmbedding
from .diffusion import (
    NoiseSchedule, forward_diffusion_se3, compute_flow_targets,
    ddim_reverse_step, diffusion_loss,
)
from .se3_utils import transform_points, invert_se3
from .pos_encoding import nerf_positional_encoding


class GraphDiffusionPolicy(nn.Module):
    """
    Instant Policy: In-Context Graph Diffusion model.
    """

    def __init__(self, cfg: IPConfig = None):
        super().__init__()
        self.cfg = cfg or IPConfig()
        c = self.cfg

        # ── Components ────────────────────────────────────────────
        self.geo_encoder = GeometryEncoder(c)
        self.gripper_encoder = GripperNodeEncoder(
            num_kp=c.num_gripper_keypoints,
            distinction_dim=c.gripper_feat_dim,
            state_embed_dim=c.gripper_state_embed_dim,
        )
        self.sigma = SigmaNetwork(c)
        self.phi = PhiNetwork(c)
        self.psi = PsiNetwork(c)
        self.step_embed = DiffusionStepEmbedding(c.hidden_dim)

        # Noise schedule (will be moved to device in forward)
        self.register_buffer('_dummy', torch.tensor(0.0))  # for device tracking
        self._schedule = None

        # Gripper keypoints in EE frame (fixed)
        self.register_buffer('kp',
                             DEFAULT_GRIPPER_KEYPOINTS.clone(),
                             persistent=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def schedule(self) -> NoiseSchedule:
        if self._schedule is None or self._schedule.betas.device != self.device:
            self._schedule = NoiseSchedule(
                self.cfg.num_diffusion_steps_train,
                self.cfg.beta_start, self.cfg.beta_end, self.device
            )
        return self._schedule

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _encode_pcd(self, pcd: torch.Tensor):
        """pcd: (B, 2048, 3) → centroids (B, M, 3), features (B, M, 512)."""
        return self.geo_encoder(pcd)

    def _build_and_run_sigma(self, scene_pos, scene_feat,
                              gripper_pos, grip_state):
        """
        Build a local graph and run σ.
        scene_pos:  (M, 3)
        scene_feat: (M, feat_dim)
        gripper_pos: (K, 3)
        grip_state: scalar tensor (int)
        Returns: gripper features (K, hidden_dim)
        """
        gripper_feat = self.gripper_encoder(grip_state.unsqueeze(0))[0]
        local_g = build_local_graph(
            scene_pos, scene_feat, gripper_pos, gripper_feat,
            freq_bands=self.cfg.edge_freq_bands
        )
        return self.sigma(local_g)

    def _process_all_subgraphs(self, sample: dict):
        """
        Process all local subgraphs (demos + current) through geometry
        encoder + σ.

        sample format:
          {
            'demos': [
              {'pcds': [(2048,3),...], 'T_w_es': [(4,4),...], 'grips': [int,...]},
              ...
            ],
            'current': {'pcd': (2048,3), 'T_w_e': (4,4), 'grip': int},
          }

        Returns:
          demo_gripper_feats: list[list[Tensor(K, hidden)]]  per demo, per wp
          demo_gripper_pos_world: list[list[Tensor(K, 3)]]
          current_gripper_feat: (K, hidden_dim)
          current_gripper_pos_world: (K, 3)
        """
        device = self.device
        kp = self.kp

        demo_gripper_feats = []
        demo_gripper_pos_world = []

        for demo in sample['demos']:
            feats_list = []
            pos_list = []
            for pcd, T_we, grip in zip(demo['pcds'], demo['T_w_es'], demo['grips']):
                pcd_t = pcd.to(device).unsqueeze(0) if pcd.dim() == 2 else pcd.to(device)
                T_we_t = T_we.to(device)
                grip_t = torch.tensor(grip, device=device) if not isinstance(grip, torch.Tensor) else grip.to(device)

                # Point cloud in EE frame
                T_ew = invert_se3(T_we_t.unsqueeze(0)).squeeze(0)
                pcd_ee = transform_points(T_ew.unsqueeze(0), pcd_t).squeeze(0)

                # Encode
                centroids, feat = self._encode_pcd(pcd_ee.unsqueeze(0))
                centroids = centroids.squeeze(0)
                feat = feat.squeeze(0)

                # Gripper keypoints in EE frame = kp
                g_feat = self._build_and_run_sigma(
                    centroids, feat, kp, grip_t
                )
                feats_list.append(g_feat)

                # World-frame keypoint positions (for context edges)
                g_pos_world = gripper_keypoints_world(T_we_t.unsqueeze(0), kp).squeeze(0)
                pos_list.append(g_pos_world)

            demo_gripper_feats.append(feats_list)
            demo_gripper_pos_world.append(pos_list)

        # Current observation
        cur = sample['current']
        pcd_cur = cur['pcd'].to(device)
        T_we_cur = cur['T_w_e'].to(device)
        grip_cur = cur['grip'] if isinstance(cur['grip'], torch.Tensor) else torch.tensor(cur['grip'], device=device)

        T_ew_cur = invert_se3(T_we_cur.unsqueeze(0)).squeeze(0)
        pcd_ee_cur = transform_points(T_ew_cur.unsqueeze(0), pcd_cur.unsqueeze(0)).squeeze(0)
        cent_cur, feat_cur = self._encode_pcd(pcd_ee_cur.unsqueeze(0))
        cent_cur, feat_cur = cent_cur.squeeze(0), feat_cur.squeeze(0)

        cur_g_feat = self._build_and_run_sigma(cent_cur, feat_cur, kp, grip_cur)
        cur_g_pos_world = gripper_keypoints_world(T_we_cur.unsqueeze(0), kp).squeeze(0)

        return (demo_gripper_feats, demo_gripper_pos_world,
                cur_g_feat, cur_g_pos_world)

    def _run_phi(self, demo_gripper_feats, demo_gripper_pos_world,
                 cur_g_feat, cur_g_pos_world):
        """Run φ network on context graph → bottleneck."""
        K = self.cfg.num_gripper_keypoints

        # Flatten all gripper features
        all_feats = []
        for demo_feats in demo_gripper_feats:
            for gf in demo_feats:
                all_feats.append(gf)
        all_feats.append(cur_g_feat)
        all_gripper_feats = torch.cat(all_feats, dim=0)  # (total_nodes, hidden)

        # Build context edges
        ei_dict, ea_dict = build_context_edges(
            demo_gripper_pos_world, cur_g_pos_world,
            freq_bands=self.cfg.edge_freq_bands,
        )

        current_start = all_gripper_feats.shape[0] - K
        current_slice = slice(current_start, current_start + K)

        bottleneck = self.phi(all_gripper_feats, ei_dict, ea_dict, current_slice)
        return bottleneck  # (K, hidden_dim)

    def _run_psi_single_step(self, bottleneck, action_gripper_feats,
                              cur_g_pos_world, action_gripper_pos,
                              diffusion_step_k):
        """Run ψ on action graph for one diffusion step."""
        K = self.cfg.num_gripper_keypoints
        T_pred = action_gripper_feats.shape[0] // K

        # Add diffusion step embedding to action node features
        step_emb = self.step_embed(diffusion_step_k, self.cfg.hidden_dim,
                                    self.device, bottleneck.dtype)
        action_gripper_feats = action_gripper_feats + step_emb.unsqueeze(0)

        # Concatenate current (bottleneck) + action gripper features
        all_feats = torch.cat([bottleneck, action_gripper_feats], dim=0)

        # Build action edges
        action_pos_list = [
            action_gripper_pos[t * K: (t + 1) * K]
            for t in range(T_pred)
        ]
        ei_dict, ea_dict = build_action_edges(
            cur_g_pos_world, action_pos_list,
            freq_bands=self.cfg.edge_freq_bands,
        )

        action_slice = slice(K, K + T_pred * K)
        flow = self.psi(all_feats, ei_dict, ea_dict, action_slice)
        return flow  # (T*K, 7)

    # ──────────────────────────────────────────────────────────────
    # Training forward
    # ──────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Training forward pass.

        batch: {
          'demos': [...],
          'current': {...},
          'actions': {
            'T_EAs': (T, 4, 4),    ground truth relative actions
            'grips': (T,),          ground truth gripper states
            'pcds':  [(2048,3),...] point clouds at action poses (for σ)
          }
        }

        Returns: scalar loss.
        """
        c = self.cfg
        K = c.num_gripper_keypoints
        kp = self.kp
        device = self.device

        # 1. Process demo + current subgraphs → bottleneck
        (demo_g_feats, demo_g_pos, cur_g_feat, cur_g_pos) = \
            self._process_all_subgraphs(batch)
        bottleneck = self._run_phi(demo_g_feats, demo_g_pos,
                                    cur_g_feat, cur_g_pos)

        # 2. Sample random diffusion step
        k = torch.randint(1, c.num_diffusion_steps_train + 1, (1,)).item()

        # 3. Add noise to ground truth actions
        T_EA_gt = batch['actions']['T_EAs'].to(device).unsqueeze(0)  # (1,T,4,4)
        grip_gt = batch['actions']['grips'].to(device).unsqueeze(0)  # (1,T)

        T_EA_k, grip_k, noise_se3, noise_grip = forward_diffusion_se3(
            T_EA_gt, grip_gt, k, self.schedule,
            c.max_rotation_rad, c.max_translation
        )
        T_EA_k = T_EA_k.squeeze(0)  # (T, 4, 4)
        grip_k = grip_k.squeeze(0)  # (T,)

        # 4. Build action local subgraphs with noisy actions, run σ
        action_feats_list = []
        action_pos_list = []
        cur_T_we = batch['current']['T_w_e'].to(device)

        for t in range(c.pred_horizon):
            # Scene in EE frame, transformed by inverse action (Appendix E)
            T_ea_inv = invert_se3(T_EA_k[t].unsqueeze(0)).squeeze(0)
            pcd_ee_cur = batch['actions']['pcds'][t].to(device)
            pcd_action = transform_points(T_ea_inv.unsqueeze(0),
                                           pcd_ee_cur.unsqueeze(0)).squeeze(0)
            cent, feat = self._encode_pcd(pcd_action.unsqueeze(0))
            cent, feat = cent.squeeze(0), feat.squeeze(0)

            grip_t = torch.tensor(
                int((grip_k[t] > 0).item()), device=device
            )
            g_feat = self._build_and_run_sigma(cent, feat, kp, grip_t)
            action_feats_list.append(g_feat)

            # Action keypoint positions (world frame)
            T_wa = cur_T_we @ T_EA_k[t]
            g_pos = gripper_keypoints_world(T_wa.unsqueeze(0), kp).squeeze(0)
            action_pos_list.append(g_pos)

        action_gripper_feats = torch.cat(action_feats_list, dim=0)  # (T*K, h)
        action_gripper_pos = torch.cat(action_pos_list, dim=0)      # (T*K, 3)

        # 5. Run ψ → flow predictions
        flow_pred = self._run_psi_single_step(
            bottleneck, action_gripper_feats,
            cur_g_pos, action_gripper_pos, k
        )  # (T*K, 7)
        flow_pred = flow_pred.reshape(1, c.pred_horizon, K, 7)

        # 6. Compute flow targets
        flow_target = compute_flow_targets(
            T_EA_gt, T_EA_k.unsqueeze(0),
            grip_gt, grip_k.unsqueeze(0), kp
        )  # (1, T, K, 7)

        # 7. Normalise both target and prediction to [-1, 1] (Appendix E)
        # Translation flow: normalise by 2 * max_translation (capped range)
        # Rotation flow: normalise by 1 (already small, kept as-is)
        # Gripper gradient: already in [-2, 2], kept as-is
        norm_t = 2 * c.max_translation
        flow_target[..., :3] = flow_target[..., :3] / norm_t
        flow_pred[..., :3] = flow_pred[..., :3] / norm_t

        return diffusion_loss(flow_pred, flow_target)

    # ──────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_actions(self, sample: dict,
                        num_diffusion_steps: int = None
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict actions given demonstrations + current observation.

        sample: same format as training but without 'actions' key.
        Returns:
            actions: (T, 4, 4) relative end-effector transforms
            grips:   (T,)      gripper commands
        """
        c = self.cfg
        K = c.num_gripper_keypoints
        T_pred = c.pred_horizon
        kp = self.kp
        device = self.device
        num_steps = num_diffusion_steps or c.num_diffusion_steps_infer

        # 1. Process demos + current → bottleneck
        (demo_g_feats, demo_g_pos, cur_g_feat, cur_g_pos) = \
            self._process_all_subgraphs(sample)
        bottleneck = self._run_phi(demo_g_feats, demo_g_pos,
                                    cur_g_feat, cur_g_pos)

        # 2. Initialise noisy actions from N(0, I)
        # Action keypoint positions: sample around the current keypoint positions
        cur_T_we = sample['current']['T_w_e'].to(device)
        cur_kp_world = cur_g_pos  # (K, 3)
        action_pos = cur_kp_world.unsqueeze(0).expand(T_pred, -1, -1) + \
            torch.randn(T_pred, K, 3, device=device) * 0.01
        action_pos = action_pos.reshape(T_pred * K, 3)
        grip = torch.randn(T_pred, device=device)

        # Build inference schedule
        infer_schedule = NoiseSchedule(
            num_steps * 5,  # use larger schedule, subsample
            c.beta_start, c.beta_end, device
        )
        step_indices = torch.linspace(
            infer_schedule.num_steps - 1, 0, num_steps, device=device
        ).long()

        # 3. Iterative denoising
        T_accum = torch.eye(4, device=device).unsqueeze(0).expand(T_pred, -1, -1).clone()

        for i, k_idx in enumerate(step_indices):
            k = k_idx.item()
            k_prev = step_indices[i + 1].item() if i + 1 < len(step_indices) else -1

            # Build action subgraphs with current noisy positions
            pcd_ee_cur = sample['current']['pcd'].to(device)
            action_feats_list = []
            for t in range(T_pred):
                # For inference, reuse current observation point cloud
                # Transform by estimated inverse action
                T_ea_est = T_accum[t]
                T_ea_inv = invert_se3(T_ea_est.unsqueeze(0)).squeeze(0)
                pcd_action = transform_points(T_ea_inv.unsqueeze(0),
                                               pcd_ee_cur.unsqueeze(0)).squeeze(0)
                cent, feat = self._encode_pcd(pcd_action.unsqueeze(0))
                cent, feat = cent.squeeze(0), feat.squeeze(0)

                grip_t = torch.tensor(int((grip[t] > 0).item()), device=device)
                g_feat = self._build_and_run_sigma(cent, feat, kp, grip_t)
                action_feats_list.append(g_feat)

            action_gripper_feats = torch.cat(action_feats_list, dim=0)

            # Run ψ
            flow = self._run_psi_single_step(
                bottleneck, action_gripper_feats,
                cur_g_pos, action_pos, k
            )  # (T*K, 7)
            flow_reshaped = flow.reshape(T_pred, K, 7)

            # DDIM step
            action_pos_reshaped = action_pos.reshape(T_pred, K, 3)
            T_step, action_pos_new, grip = ddim_reverse_step(
                action_pos_reshaped, flow_reshaped, grip,
                k, k_prev, infer_schedule, kp
            )
            action_pos = action_pos_new.reshape(T_pred * K, 3)
            T_accum = T_step @ T_accum

        # 4. Discretise gripper
        grips = (grip > 0).float()

        return T_accum, grips
