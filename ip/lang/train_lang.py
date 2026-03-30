"""
Training script for language-conditioned modality transfer (Appendix J).

Training procedure:
  1. Load pre-trained IP model (frozen σ, φ, ψ, geometry encoder).
  2. For each training sample:
     a. Run frozen IP pipeline: demos + obs → φ → bottleneck_demo (target).
     b. Run φ_lang: language + obs → bottleneck_lang (prediction).
     c. Compute contrastive + MSE loss between bottleneck_lang and bottleneck_demo.
  3. Only φ_lang parameters and language projection MLP are trained.

At inference, φ_lang replaces the entire "demos + σ + φ" pipeline.
The frozen ψ takes the language-derived bottleneck and predicts actions.
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from typing import Optional

from ..config import IPConfig
from ..model import GraphDiffusionPolicy
from ..geometry_encoder import GeometryEncoder
from ..graph_builder import (
    GripperNodeEncoder, build_local_graph, gripper_keypoints_ee,
    DEFAULT_GRIPPER_KEYPOINTS,
)
from ..se3_utils import invert_se3, transform_points
from .phi_lang import PhiLang, BottleneckAlignmentLoss
from .lang_dataset import LangAnnotatedDataset


class LanguageTransferTrainer:
    """
    Manages the full language transfer training pipeline.
    """

    def __init__(self, ip_model: GraphDiffusionPolicy,
                 phi_lang: PhiLang,
                 cfg: IPConfig = None,
                 device: str = 'cuda'):
        self.cfg = cfg or IPConfig()
        self.device = device

        # Frozen IP model (used to compute target bottlenecks)
        self.ip_model = ip_model.to(device).eval()
        for param in self.ip_model.parameters():
            param.requires_grad = False

        # Trainable φ_lang
        self.phi_lang = phi_lang.to(device)

        # Loss
        self.loss_fn = BottleneckAlignmentLoss(
            temperature=self.cfg.lang_temperature,
            lambda_mse=self.cfg.lang_lambda_mse,
        )

    @torch.no_grad()
    def compute_target_bottleneck(self, sample: dict) -> torch.Tensor:
        """
        Run the frozen IP model to extract the bottleneck representation
        from demonstrations + current observation.

        Returns: (K, hidden_dim) target bottleneck.
        """
        # Use the IP model's internal methods
        (demo_g_feats, demo_g_pos, cur_g_feat, cur_g_pos) = \
            self.ip_model._process_all_subgraphs(sample)
        bottleneck = self.ip_model._run_phi(
            demo_g_feats, demo_g_pos, cur_g_feat, cur_g_pos
        )
        return bottleneck.detach()

    def compute_lang_bottleneck(self, sample: dict,
                                 text: str) -> torch.Tensor:
        """
        Run φ_lang to predict bottleneck from language + current observation.

        Returns: (K, hidden_dim) predicted bottleneck.
        """
        device = self.device
        cfg = self.cfg
        kp = DEFAULT_GRIPPER_KEYPOINTS.to(device)

        cur = sample['current']
        pcd = cur['pcd'].to(device)
        T_we = cur['T_w_e'].to(device)
        grip = cur['grip'] if isinstance(cur['grip'], torch.Tensor) \
            else torch.tensor(cur['grip'], device=device)

        # Encode point cloud in EE frame
        T_ew = invert_se3(T_we.unsqueeze(0)).squeeze(0)
        pcd_ee = transform_points(T_ew.unsqueeze(0), pcd.unsqueeze(0)).squeeze(0)
        centroids, feat = self.ip_model._encode_pcd(pcd_ee.unsqueeze(0))
        centroids, feat = centroids.squeeze(0), feat.squeeze(0)

        # Gripper features
        gripper_feat = self.ip_model.gripper_encoder(grip.unsqueeze(0))[0]

        # Language embedding
        lang_feat = self.phi_lang.lang_encoder(
            texts=[text], device=device
        )  # (1, hidden_dim)

        # Build scene→gripper edges (same as in σ)
        from ..graph_builder import _fully_connected_edges, _make_edge_attr
        M = centroids.shape[0]
        K = kp.shape[0]
        ei = _fully_connected_edges(M, K, device=device)
        ea = _make_edge_attr(centroids, kp, ei[0], ei[1], cfg.edge_freq_bands)

        # Run φ_lang
        bottleneck = self.phi_lang(
            scene_feat=feat,
            scene_pos=centroids,
            gripper_feat=gripper_feat,
            gripper_pos=kp,
            lang_feat=lang_feat,
            scene_gripper_edge_index=ei,
            scene_gripper_edge_attr=ea,
        )
        return bottleneck  # (K, hidden_dim)

    def train_step(self, batch: list) -> torch.Tensor:
        """
        One training step on a batch of samples.
        batch: list of dicts from LangAnnotatedDataset.
        Returns: scalar loss.
        """
        lang_bottlenecks = []
        demo_bottlenecks = []

        for sample in batch:
            # Target: frozen IP model bottleneck
            ip_sample = {
                'demos': sample['demos'],
                'current': sample['current'],
            }
            target = self.compute_target_bottleneck(ip_sample)
            demo_bottlenecks.append(target)

            # Prediction: φ_lang bottleneck
            pred = self.compute_lang_bottleneck(ip_sample, sample['text'])
            lang_bottlenecks.append(pred)

        # Stack into batch
        lang_bn = torch.stack(lang_bottlenecks)   # (B, K, hidden_dim)
        demo_bn = torch.stack(demo_bottlenecks)   # (B, K, hidden_dim)

        return self.loss_fn(lang_bn, demo_bn)


def train_language_transfer(
    ip_checkpoint: str,
    data_dir: str,
    save_dir: str,
    cfg: IPConfig = None,
    device: str = 'cuda',
    resume_from: Optional[str] = None,
):
    """
    Full training loop for language transfer.

    Args:
        ip_checkpoint: path to pre-trained IP model checkpoint.
        data_dir: directory with language-annotated RLBench demonstrations.
        save_dir: directory to save φ_lang checkpoints.
    """
    print("=" * 60)
    print("Language Transfer Training (Appendix J)")
    print("=" * 60)

    cfg = cfg or IPConfig()

    # Load pre-trained IP model
    print(f"Loading IP model from {ip_checkpoint}...")
    ckpt = torch.load(ip_checkpoint, map_location=device)
    if 'cfg' in ckpt:
        ip_cfg = ckpt['cfg']
    else:
        ip_cfg = cfg
    ip_model = GraphDiffusionPolicy(ip_cfg)
    if 'model' in ckpt:
        ip_model.load_state_dict(ckpt['model'])
    else:
        ip_model.load_state_dict(ckpt)
    print("IP model loaded.")

    # Create φ_lang
    phi_lang = PhiLang(cfg)

    # Resume if available
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        lang_ckpt = torch.load(resume_from, map_location=device)
        phi_lang.load_state_dict(lang_ckpt['phi_lang'])
        start_step = lang_ckpt.get('step', 0)
        print(f"Resumed φ_lang from step {start_step}")

    # Trainer
    trainer = LanguageTransferTrainer(ip_model, phi_lang, cfg, device)

    # Optimiser
    trainable_params = list(phi_lang.parameters())
    optimiser = torch.optim.AdamW(trainable_params, lr=cfg.lang_lr)
    scaler = GradScaler()

    # Dataset
    dataset = LangAnnotatedDataset(data_dir, cfg=cfg)
    loader = DataLoader(dataset, batch_size=cfg.batch_size,
                       shuffle=True, collate_fn=lambda x: x,
                       num_workers=0)

    # Training loop
    os.makedirs(save_dir, exist_ok=True)
    phi_lang.train()
    running_loss = 0.0
    step = start_step

    for epoch in range(1000):  # effectively infinite
        for batch in loader:
            step += 1
            if step > cfg.lang_train_steps:
                break

            with autocast():
                loss = trainer.train_step(batch)

            optimiser.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
            scaler.step(optimiser)
            scaler.update()

            running_loss += loss.item()

            if step % 50 == 0:
                avg = running_loss / 50
                print(f"  Step {step}/{cfg.lang_train_steps}  Loss: {avg:.5f}")
                running_loss = 0.0

            if step % 5000 == 0:
                ckpt_path = os.path.join(save_dir, f'phi_lang_step{step}.pt')
                torch.save({
                    'step': step,
                    'phi_lang': phi_lang.state_dict(),
                    'cfg': cfg,
                }, ckpt_path)
                print(f"  Saved {ckpt_path}")

        if step > cfg.lang_train_steps:
            break

    # Final save
    final_path = os.path.join(save_dir, 'phi_lang_final.pt')
    torch.save({
        'step': step,
        'phi_lang': phi_lang.state_dict(),
        'cfg': cfg,
    }, final_path)
    print(f"Language transfer training complete. Saved to {final_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Train language-conditioned modality transfer'
    )
    parser.add_argument('--ip_checkpoint', type=str, required=True,
                       help='Path to pre-trained IP model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory with language-annotated demonstrations')
    parser.add_argument('--save_dir', type=str,
                       default='./checkpoints_lang',
                       help='Directory to save φ_lang checkpoints')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    train_language_transfer(
        args.ip_checkpoint, args.data_dir, args.save_dir,
        device=args.device, resume_from=args.resume,
    )


if __name__ == '__main__':
    main()
