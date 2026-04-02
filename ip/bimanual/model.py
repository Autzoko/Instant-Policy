"""
Bimanual GraphDiffusionPolicy model.

Extends the single-arm GraphDiffusionPolicy for dual-arm manipulation.
Both arms share the geometry encoder and (optionally) the gripper encoder,
but have separate action predictions connected via cross-arm graph edges
for implicit coordination.

Training forward pass:
  1. Encode shared scene PCD with phi_e (frozen).
  2. Build bimanual local graphs (scene + left + right), process with sigma.
  3. Build bimanual context graph, process with phi -> (bottleneck_L, bottleneck_R).
  4. Add noise to ground-truth actions for BOTH arms independently.
  5. Build bimanual action graph with noisy actions, process with psi -> flow.
  6. Compute loss (MSE on per-arm flow predictions).

Inference:
  1-3. Same as training.
  4.   Sample noisy actions from N(0,I) for both arms.
  5.   Iteratively denoise using sigma->psi loop with sync edges.
  6.   Extract SE(3) actions via SVD per arm.
"""
import torch
import torch.nn as nn
from typing import Tuple

from ..geometry_encoder import GeometryEncoder
from ..graph_builder import (
    GripperNodeEncoder,
    gripper_keypoints_world,
    DEFAULT_GRIPPER_KEYPOINTS,
)
from ..networks import DiffusionStepEmbedding
from ..diffusion import (
    NoiseSchedule,
    forward_diffusion_se3,
    compute_flow_targets,
    ddim_reverse_step,
    diffusion_loss,
)
from ..se3_utils import transform_points, invert_se3
from ..pos_encoding import nerf_positional_encoding

from .config import BimanualIPConfig
from .graph_builder import (
    build_bimanual_local_graph,
    build_bimanual_context_edges,
    build_bimanual_action_edges,
)
from .networks import BimanualSigmaNetwork, BimanualPhiNetwork, BimanualPsiNetwork


def _midpoint_frame(T_left: torch.Tensor,
                    T_right: torch.Tensor) -> torch.Tensor:
    """
    Compute a midpoint reference frame between two SE(3) poses.
    Translation: average.  Rotation: identity (for simplicity and stability).

    T_left, T_right: (4, 4)
    Returns: (4, 4)
    """
    T_mid = torch.eye(4, device=T_left.device, dtype=T_left.dtype)
    T_mid[:3, 3] = 0.5 * (T_left[:3, 3] + T_right[:3, 3])
    return T_mid


class BimanualGraphDiffusionPolicy(nn.Module):
    """
    Bimanual Instant Policy: dual-arm in-context graph diffusion model.
    """

    def __init__(self, cfg: BimanualIPConfig = None):
        super().__init__()
        self.cfg = cfg or BimanualIPConfig()
        c = self.cfg

        # ── Shared components ────────────────────────────────────
        self.geo_encoder = GeometryEncoder(c)
        self.gripper_encoder = GripperNodeEncoder(
            num_kp=c.num_gripper_keypoints,
            distinction_dim=c.gripper_feat_dim,
            state_embed_dim=c.gripper_state_embed_dim,
        )
        if not c.share_gripper_encoder:
            self.gripper_encoder_right = GripperNodeEncoder(
                num_kp=c.num_gripper_keypoints,
                distinction_dim=c.gripper_feat_dim,
                state_embed_dim=c.gripper_state_embed_dim,
            )

        # ── Bimanual sub-networks ────────────────────────────────
        self.sigma = BimanualSigmaNetwork(c)
        self.phi = BimanualPhiNetwork(c)
        self.psi = BimanualPsiNetwork(c)
        self.step_embed = DiffusionStepEmbedding(c.hidden_dim)

        # ── Bookkeeping ──────────────────────────────────────────
        self.register_buffer('_dummy', torch.tensor(0.0))
        self._schedule = None
        self.register_buffer('kp', DEFAULT_GRIPPER_KEYPOINTS.clone(),
                             persistent=False)

    @property
    def device(self):
        return self._dummy.device

    @property
    def schedule(self) -> NoiseSchedule:
        if self._schedule is None or self._schedule.betas.device != self.device:
            self._schedule = NoiseSchedule(
                self.cfg.num_diffusion_steps_train,
                self.cfg.beta_start, self.cfg.beta_end, self.device,
            )
        return self._schedule

    # ──────────────────────────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────────────────────────

    def _encode_pcd(self, pcd: torch.Tensor):
        """pcd: (B, 2048, 3) -> centroids (B, M, 3), features (B, M, 512)."""
        return self.geo_encoder(pcd)

    def _get_gripper_feat(self, grip_state: torch.Tensor, arm: str = 'left'):
        """Encode gripper state.  grip_state: scalar tensor."""
        if not self.cfg.share_gripper_encoder and arm == 'right':
            return self.gripper_encoder_right(grip_state.unsqueeze(0))[0]
        return self.gripper_encoder(grip_state.unsqueeze(0))[0]

    def _transform_pcd_to_frame(self, pcd_world: torch.Tensor,
                                T_frame: torch.Tensor) -> torch.Tensor:
        """Transform PCD from world frame to a reference frame."""
        T_inv = invert_se3(T_frame.unsqueeze(0)).squeeze(0)
        return transform_points(T_inv.unsqueeze(0),
                                pcd_world.unsqueeze(0)).squeeze(0)

    def _compute_scene_frame(self, T_we_left: torch.Tensor,
                             T_we_right: torch.Tensor) -> torch.Tensor:
        """Compute the reference frame for scene encoding."""
        if self.cfg.scene_frame == 'midpoint':
            return _midpoint_frame(T_we_left, T_we_right)
        else:
            return torch.eye(4, device=self.device, dtype=T_we_left.dtype)

    def _build_and_run_sigma(self, scene_pos, scene_feat,
                             kp_left, grip_left,
                             kp_right, grip_right):
        """
        Build a bimanual local graph and run sigma.

        Returns: (left_feat, right_feat) each (K, hidden_dim)
        """
        left_feat = self._get_gripper_feat(grip_left, 'left')
        right_feat = self._get_gripper_feat(grip_right, 'right')

        local_g = build_bimanual_local_graph(
            scene_pos, scene_feat, kp_left, left_feat, kp_right, right_feat,
            freq_bands=self.cfg.edge_freq_bands,
            enable_coordinate_edges=self.cfg.enable_coordinate_edges,
        )
        return self.sigma(local_g)

    # ──────────────────────────────────────────────────────────────
    # Process all subgraphs (demos + current)
    # ──────────────────────────────────────────────────────────────

    def _process_all_subgraphs(self, sample: dict):
        """
        Process all bimanual local subgraphs through geo encoder + sigma.

        sample format:
          {
            'demos': [
              {
                'pcds': [(2048,3),...],
                'T_w_es_left':  [(4,4),...],  'T_w_es_right':  [(4,4),...],
                'grips_left':   [int,...],     'grips_right':   [int,...],
              },
              ...
            ],
            'current': {
              'pcd': (2048,3),
              'T_w_e_left':  (4,4),  'T_w_e_right':  (4,4),
              'grip_left': int,      'grip_right': int,
            },
          }

        Returns:
          demo_g_feats_left:  list[list[Tensor(K,h)]]
          demo_g_feats_right: list[list[Tensor(K,h)]]
          demo_g_pos_left:    list[list[Tensor(K,3)]]
          demo_g_pos_right:   list[list[Tensor(K,3)]]
          cur_g_feat_left:    (K, h)
          cur_g_feat_right:   (K, h)
          cur_g_pos_left:     (K, 3)
          cur_g_pos_right:    (K, 3)
        """
        device = self.device
        kp = self.kp

        demo_g_feats_left, demo_g_feats_right = [], []
        demo_g_pos_left, demo_g_pos_right = [], []

        for demo in sample['demos']:
            fl_list, fr_list = [], []
            pl_list, pr_list = [], []

            for i, pcd in enumerate(demo['pcds']):
                pcd_t = pcd.to(device)
                if pcd_t.dim() == 2:
                    pcd_t = pcd_t.unsqueeze(0)

                T_we_l = demo['T_w_es_left'][i].to(device) if isinstance(
                    demo['T_w_es_left'][i], torch.Tensor
                ) else torch.tensor(demo['T_w_es_left'][i], device=device, dtype=torch.float32)
                T_we_r = demo['T_w_es_right'][i].to(device) if isinstance(
                    demo['T_w_es_right'][i], torch.Tensor
                ) else torch.tensor(demo['T_w_es_right'][i], device=device, dtype=torch.float32)

                grip_l = demo['grips_left'][i]
                grip_r = demo['grips_right'][i]
                grip_l_t = torch.tensor(grip_l, device=device) if not isinstance(
                    grip_l, torch.Tensor) else grip_l.to(device)
                grip_r_t = torch.tensor(grip_r, device=device) if not isinstance(
                    grip_r, torch.Tensor) else grip_r.to(device)

                # Transform PCD to reference frame
                T_frame = self._compute_scene_frame(T_we_l, T_we_r)
                pcd_frame = self._transform_pcd_to_frame(pcd_t.squeeze(0), T_frame)

                # Encode scene
                cent, feat = self._encode_pcd(pcd_frame.unsqueeze(0))
                cent, feat = cent.squeeze(0), feat.squeeze(0)

                # Gripper keypoints in reference frame
                T_frame_inv = invert_se3(T_frame.unsqueeze(0)).squeeze(0)
                # kp in reference frame = T_frame_inv @ T_we @ kp_ee
                T_fl = T_frame_inv @ T_we_l
                T_fr = T_frame_inv @ T_we_r
                kp_left_frame = gripper_keypoints_world(T_fl.unsqueeze(0), kp).squeeze(0)
                kp_right_frame = gripper_keypoints_world(T_fr.unsqueeze(0), kp).squeeze(0)

                # Run sigma
                g_feat_l, g_feat_r = self._build_and_run_sigma(
                    cent, feat,
                    kp_left_frame, grip_l_t,
                    kp_right_frame, grip_r_t,
                )
                fl_list.append(g_feat_l)
                fr_list.append(g_feat_r)

                # World-frame keypoint positions (for context edges)
                pl_list.append(
                    gripper_keypoints_world(T_we_l.unsqueeze(0), kp).squeeze(0))
                pr_list.append(
                    gripper_keypoints_world(T_we_r.unsqueeze(0), kp).squeeze(0))

            demo_g_feats_left.append(fl_list)
            demo_g_feats_right.append(fr_list)
            demo_g_pos_left.append(pl_list)
            demo_g_pos_right.append(pr_list)

        # Current observation
        cur = sample['current']
        pcd_cur = cur['pcd'].to(device)
        T_we_l_cur = cur['T_w_e_left'].to(device) if isinstance(
            cur['T_w_e_left'], torch.Tensor
        ) else torch.tensor(cur['T_w_e_left'], device=device, dtype=torch.float32)
        T_we_r_cur = cur['T_w_e_right'].to(device) if isinstance(
            cur['T_w_e_right'], torch.Tensor
        ) else torch.tensor(cur['T_w_e_right'], device=device, dtype=torch.float32)

        grip_l_cur = cur['grip_left']
        grip_r_cur = cur['grip_right']
        grip_l_cur_t = torch.tensor(grip_l_cur, device=device) if not isinstance(
            grip_l_cur, torch.Tensor) else grip_l_cur.to(device)
        grip_r_cur_t = torch.tensor(grip_r_cur, device=device) if not isinstance(
            grip_r_cur, torch.Tensor) else grip_r_cur.to(device)

        T_frame_cur = self._compute_scene_frame(T_we_l_cur, T_we_r_cur)
        pcd_frame_cur = self._transform_pcd_to_frame(pcd_cur, T_frame_cur)

        cent_cur, feat_cur = self._encode_pcd(pcd_frame_cur.unsqueeze(0))
        cent_cur, feat_cur = cent_cur.squeeze(0), feat_cur.squeeze(0)

        T_frame_inv_cur = invert_se3(T_frame_cur.unsqueeze(0)).squeeze(0)
        T_fl_cur = T_frame_inv_cur @ T_we_l_cur
        T_fr_cur = T_frame_inv_cur @ T_we_r_cur
        kp_l_frame_cur = gripper_keypoints_world(T_fl_cur.unsqueeze(0), kp).squeeze(0)
        kp_r_frame_cur = gripper_keypoints_world(T_fr_cur.unsqueeze(0), kp).squeeze(0)

        cur_g_feat_l, cur_g_feat_r = self._build_and_run_sigma(
            cent_cur, feat_cur,
            kp_l_frame_cur, grip_l_cur_t,
            kp_r_frame_cur, grip_r_cur_t,
        )
        cur_g_pos_l = gripper_keypoints_world(T_we_l_cur.unsqueeze(0), kp).squeeze(0)
        cur_g_pos_r = gripper_keypoints_world(T_we_r_cur.unsqueeze(0), kp).squeeze(0)

        return (demo_g_feats_left, demo_g_feats_right,
                demo_g_pos_left, demo_g_pos_right,
                cur_g_feat_l, cur_g_feat_r,
                cur_g_pos_l, cur_g_pos_r)

    # ──────────────────────────────────────────────────────────────
    # Phi (context aggregation)
    # ──────────────────────────────────────────────────────────────

    def _run_phi(self,
                 demo_g_feats_left, demo_g_feats_right,
                 demo_g_pos_left, demo_g_pos_right,
                 cur_g_feat_l, cur_g_feat_r,
                 cur_g_pos_l, cur_g_pos_r):
        """Run bimanual phi -> (bottleneck_left, bottleneck_right)."""
        K = self.cfg.num_gripper_keypoints

        # Flatten per-arm features
        def _flatten_feats(demo_feats_list, cur_feat):
            all_f = []
            for demo_feats in demo_feats_list:
                for gf in demo_feats:
                    all_f.append(gf)
            all_f.append(cur_feat)
            return torch.cat(all_f, dim=0)

        all_left = _flatten_feats(demo_g_feats_left, cur_g_feat_l)
        all_right = _flatten_feats(demo_g_feats_right, cur_g_feat_r)

        # Build context edges
        ei_dict, ea_dict = build_bimanual_context_edges(
            demo_g_pos_left, demo_g_pos_right,
            cur_g_pos_l, cur_g_pos_r,
            freq_bands=self.cfg.edge_freq_bands,
            enable_bimanual_edges=self.cfg.enable_bimanual_edges,
        )

        # Current nodes are the last K in each arm's array
        cur_start = all_left.shape[0] - K
        cur_slice = slice(cur_start, cur_start + K)

        bottleneck_l, bottleneck_r = self.phi(
            all_left, all_right, ei_dict, ea_dict,
            cur_slice, cur_slice,
        )
        return bottleneck_l, bottleneck_r

    # ──────────────────────────────────────────────────────────────
    # Psi (action denoising)
    # ──────────────────────────────────────────────────────────────

    def _run_psi_single_step(self,
                             bottleneck_l, bottleneck_r,
                             action_feats_l, action_feats_r,
                             cur_g_pos_l, cur_g_pos_r,
                             action_pos_l, action_pos_r,
                             diffusion_step_k):
        """Run psi -> (flow_left, flow_right) each (T*K, 7)."""
        K = self.cfg.num_gripper_keypoints
        T_pred = action_feats_l.shape[0] // K

        # Add diffusion step embedding
        step_emb = self.step_embed(diffusion_step_k, self.cfg.hidden_dim,
                                   self.device, bottleneck_l.dtype)
        action_feats_l = action_feats_l + step_emb.unsqueeze(0)
        action_feats_r = action_feats_r + step_emb.unsqueeze(0)

        # Concat bottleneck + action features per arm
        all_l = torch.cat([bottleneck_l, action_feats_l], dim=0)
        all_r = torch.cat([bottleneck_r, action_feats_r], dim=0)

        # Build action edges
        action_pos_l_list = [
            action_pos_l[t * K: (t + 1) * K] for t in range(T_pred)
        ]
        action_pos_r_list = [
            action_pos_r[t * K: (t + 1) * K] for t in range(T_pred)
        ]
        ei_dict, ea_dict = build_bimanual_action_edges(
            cur_g_pos_l, action_pos_l_list,
            cur_g_pos_r, action_pos_r_list,
            freq_bands=self.cfg.edge_freq_bands,
            enable_sync_edges=self.cfg.enable_sync_edges,
        )

        action_slice = slice(K, K + T_pred * K)
        flow_l, flow_r = self.psi(
            all_l, all_r, ei_dict, ea_dict,
            action_slice, action_slice,
        )
        return flow_l, flow_r

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
            'T_EAs_left':  (T, 4, 4),   'T_EAs_right':  (T, 4, 4),
            'grips_left':  (T,),         'grips_right':  (T,),
          }
        }

        Returns: scalar loss.
        """
        c = self.cfg
        K = c.num_gripper_keypoints
        kp = self.kp
        device = self.device
        T_pred = batch['actions']['T_EAs_left'].shape[0]  # actual action length

        # 1. Process demo + current subgraphs -> bottleneck
        (dg_fl, dg_fr, dg_pl, dg_pr,
         cur_fl, cur_fr, cur_pl, cur_pr) = self._process_all_subgraphs(batch)
        bottleneck_l, bottleneck_r = self._run_phi(
            dg_fl, dg_fr, dg_pl, dg_pr,
            cur_fl, cur_fr, cur_pl, cur_pr,
        )

        # 2. Sample random diffusion step
        k = torch.randint(1, c.num_diffusion_steps_train + 1, (1,)).item()

        # 3. Add noise to GT actions (independently per arm)
        T_EA_gt_l = batch['actions']['T_EAs_left'].to(device).unsqueeze(0)
        T_EA_gt_r = batch['actions']['T_EAs_right'].to(device).unsqueeze(0)
        grip_gt_l = batch['actions']['grips_left'].to(device).unsqueeze(0)
        grip_gt_r = batch['actions']['grips_right'].to(device).unsqueeze(0)

        T_EA_k_l, grip_k_l, _, _ = forward_diffusion_se3(
            T_EA_gt_l, grip_gt_l, k, self.schedule,
            c.max_rotation_rad, c.max_translation,
        )
        T_EA_k_r, grip_k_r, _, _ = forward_diffusion_se3(
            T_EA_gt_r, grip_gt_r, k, self.schedule,
            c.max_rotation_rad, c.max_translation,
        )
        T_EA_k_l = T_EA_k_l.squeeze(0)
        T_EA_k_r = T_EA_k_r.squeeze(0)
        grip_k_l = grip_k_l.squeeze(0)
        grip_k_r = grip_k_r.squeeze(0)

        # 4. Build action local subgraphs with noisy actions
        #    Same strategy as demo/current: use the midpoint of the two
        #    action world-frame poses as the reference frame so that
        #    scene centroids and gripper keypoints are in the same frame.
        cur_T_we_l = batch['current']['T_w_e_left'].to(device)
        cur_T_we_r = batch['current']['T_w_e_right'].to(device)
        pcd_cur_world = batch['current']['pcd'].to(device)

        action_feats_l_list, action_feats_r_list = [], []
        action_pos_l_list, action_pos_r_list = [], []

        for t in range(T_pred):
            # World-frame action poses for this timestep
            T_wa_l = cur_T_we_l @ T_EA_k_l[t]
            T_wa_r = cur_T_we_r @ T_EA_k_r[t]

            # Action midpoint frame (consistent with demo/current handling)
            T_action_mid = self._compute_scene_frame(T_wa_l, T_wa_r)

            # PCD in action midpoint frame
            pcd_action = self._transform_pcd_to_frame(
                pcd_cur_world, T_action_mid)
            cent, feat = self._encode_pcd(pcd_action.unsqueeze(0))
            cent, feat = cent.squeeze(0), feat.squeeze(0)

            grip_t_l = torch.tensor(
                int((grip_k_l[t] > 0).item()), device=device)
            grip_t_r = torch.tensor(
                int((grip_k_r[t] > 0).item()), device=device)

            # Gripper keypoints in action midpoint frame
            T_amid_inv = invert_se3(T_action_mid.unsqueeze(0)).squeeze(0)
            T_al_in_amid = T_amid_inv @ T_wa_l
            T_ar_in_amid = T_amid_inv @ T_wa_r
            kp_act_l = gripper_keypoints_world(
                T_al_in_amid.unsqueeze(0), kp).squeeze(0)
            kp_act_r = gripper_keypoints_world(
                T_ar_in_amid.unsqueeze(0), kp).squeeze(0)

            g_feat_l, g_feat_r = self._build_and_run_sigma(
                cent, feat, kp_act_l, grip_t_l, kp_act_r, grip_t_r,
            )
            action_feats_l_list.append(g_feat_l)
            action_feats_r_list.append(g_feat_r)

            # Action keypoint positions in world frame (for action graph edges)
            action_pos_l_list.append(
                gripper_keypoints_world(T_wa_l.unsqueeze(0), kp).squeeze(0))
            action_pos_r_list.append(
                gripper_keypoints_world(T_wa_r.unsqueeze(0), kp).squeeze(0))

        action_feats_l = torch.cat(action_feats_l_list, dim=0)
        action_feats_r = torch.cat(action_feats_r_list, dim=0)
        action_pos_l = torch.cat(action_pos_l_list, dim=0)
        action_pos_r = torch.cat(action_pos_r_list, dim=0)

        # 5. Run psi -> flow predictions
        flow_l, flow_r = self._run_psi_single_step(
            bottleneck_l, bottleneck_r,
            action_feats_l, action_feats_r,
            cur_pl, cur_pr,
            action_pos_l, action_pos_r, k,
        )
        flow_l = flow_l.reshape(1, T_pred, K, 7)
        flow_r = flow_r.reshape(1, T_pred, K, 7)

        # 6. Compute flow targets (independently per arm)
        flow_target_l = compute_flow_targets(
            T_EA_gt_l, T_EA_k_l.unsqueeze(0),
            grip_gt_l, grip_k_l.unsqueeze(0), kp,
        )
        flow_target_r = compute_flow_targets(
            T_EA_gt_r, T_EA_k_r.unsqueeze(0),
            grip_gt_r, grip_k_r.unsqueeze(0), kp,
        )

        # 7. Normalise and compute loss
        norm_t = 2 * c.max_translation
        flow_target_l[..., :3] /= norm_t
        flow_l[..., :3] = flow_l[..., :3] / norm_t
        flow_target_r[..., :3] /= norm_t
        flow_r[..., :3] = flow_r[..., :3] / norm_t

        loss_l = diffusion_loss(flow_l, flow_target_l)
        loss_r = diffusion_loss(flow_r, flow_target_r)
        return 0.5 * (loss_l + loss_r)

    # ──────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_actions(self, sample: dict,
                        num_diffusion_steps: int = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor,
                                   torch.Tensor, torch.Tensor]:
        """
        Predict bimanual actions given demonstrations + current observation.

        Returns:
            actions_left:  (T, 4, 4)
            grips_left:    (T,)
            actions_right: (T, 4, 4)
            grips_right:   (T,)
        """
        c = self.cfg
        K = c.num_gripper_keypoints
        T_pred = c.pred_horizon
        kp = self.kp
        device = self.device
        num_steps = num_diffusion_steps or c.num_diffusion_steps_infer

        # 1. Process demos + current -> bottleneck
        (dg_fl, dg_fr, dg_pl, dg_pr,
         cur_fl, cur_fr, cur_pl, cur_pr) = self._process_all_subgraphs(sample)
        bottleneck_l, bottleneck_r = self._run_phi(
            dg_fl, dg_fr, dg_pl, dg_pr,
            cur_fl, cur_fr, cur_pl, cur_pr,
        )

        # 2. Initialise noisy actions from N(0, I) for both arms
        action_pos_l = cur_pl.unsqueeze(0).expand(T_pred, -1, -1) + \
            torch.randn(T_pred, K, 3, device=device) * 0.01
        action_pos_r = cur_pr.unsqueeze(0).expand(T_pred, -1, -1) + \
            torch.randn(T_pred, K, 3, device=device) * 0.01
        action_pos_l = action_pos_l.reshape(T_pred * K, 3)
        action_pos_r = action_pos_r.reshape(T_pred * K, 3)
        grip_l = torch.randn(T_pred, device=device)
        grip_r = torch.randn(T_pred, device=device)

        # Build inference schedule
        infer_schedule = NoiseSchedule(
            num_steps * 5, c.beta_start, c.beta_end, device)
        step_indices = torch.linspace(
            infer_schedule.num_steps - 1, 0, num_steps, device=device
        ).long()

        # 3. Iterative denoising
        T_accum_l = torch.eye(4, device=device).unsqueeze(0).expand(
            T_pred, -1, -1).clone()
        T_accum_r = torch.eye(4, device=device).unsqueeze(0).expand(
            T_pred, -1, -1).clone()

        cur_T_we_l = sample['current']['T_w_e_left'].to(device)
        cur_T_we_r = sample['current']['T_w_e_right'].to(device)
        pcd_cur = sample['current']['pcd'].to(device)

        for i, k_idx in enumerate(step_indices):
            k_val = k_idx.item()
            k_prev = step_indices[i + 1].item() if i + 1 < len(step_indices) else -1

            # Build action subgraphs (same midpoint-frame strategy as training)
            action_feats_l_list, action_feats_r_list = [], []
            for t in range(T_pred):
                # World-frame estimated action poses
                T_wa_l = cur_T_we_l @ T_accum_l[t]
                T_wa_r = cur_T_we_r @ T_accum_r[t]

                # Action midpoint frame
                T_action_mid = self._compute_scene_frame(T_wa_l, T_wa_r)

                # PCD in action midpoint frame
                pcd_action = self._transform_pcd_to_frame(
                    pcd_cur, T_action_mid)
                cent, feat = self._encode_pcd(pcd_action.unsqueeze(0))
                cent, feat = cent.squeeze(0), feat.squeeze(0)

                grip_t_l = torch.tensor(
                    int((grip_l[t] > 0).item()), device=device)
                grip_t_r = torch.tensor(
                    int((grip_r[t] > 0).item()), device=device)

                # Gripper keypoints in action midpoint frame
                T_amid_inv = invert_se3(T_action_mid.unsqueeze(0)).squeeze(0)
                T_al_in_amid = T_amid_inv @ T_wa_l
                T_ar_in_amid = T_amid_inv @ T_wa_r
                kp_act_l = gripper_keypoints_world(
                    T_al_in_amid.unsqueeze(0), kp).squeeze(0)
                kp_act_r = gripper_keypoints_world(
                    T_ar_in_amid.unsqueeze(0), kp).squeeze(0)

                g_feat_l, g_feat_r = self._build_and_run_sigma(
                    cent, feat, kp_act_l, grip_t_l, kp_act_r, grip_t_r,
                )
                action_feats_l_list.append(g_feat_l)
                action_feats_r_list.append(g_feat_r)

            action_feats_l = torch.cat(action_feats_l_list, dim=0)
            action_feats_r = torch.cat(action_feats_r_list, dim=0)

            # Run psi
            flow_l, flow_r = self._run_psi_single_step(
                bottleneck_l, bottleneck_r,
                action_feats_l, action_feats_r,
                cur_pl, cur_pr,
                action_pos_l, action_pos_r, k_val,
            )
            flow_l = flow_l.reshape(T_pred, K, 7)
            flow_r = flow_r.reshape(T_pred, K, 7)

            # DDIM step (independently per arm)
            pos_l_reshaped = action_pos_l.reshape(T_pred, K, 3)
            pos_r_reshaped = action_pos_r.reshape(T_pred, K, 3)

            T_step_l, pos_new_l, grip_l = ddim_reverse_step(
                pos_l_reshaped, flow_l, grip_l,
                k_val, k_prev, infer_schedule, kp,
            )
            T_step_r, pos_new_r, grip_r = ddim_reverse_step(
                pos_r_reshaped, flow_r, grip_r,
                k_val, k_prev, infer_schedule, kp,
            )

            action_pos_l = pos_new_l.reshape(T_pred * K, 3)
            action_pos_r = pos_new_r.reshape(T_pred * K, 3)
            T_accum_l = T_step_l @ T_accum_l
            T_accum_r = T_step_r @ T_accum_r

        # 4. Discretise grippers
        grips_l = (grip_l > 0).float()
        grips_r = (grip_r > 0).float()

        return T_accum_l, grips_l, T_accum_r, grips_r
