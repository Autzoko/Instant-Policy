"""
Bimanual Instant Policy training script.

Three-phase training (same structure as single-arm):
  Phase 1: Pre-train geometry encoder (reuse single-arm, unchanged).
  Phase 2: Train bimanual graph diffusion model on pseudo-demonstrations.
  Phase 3: (Future) Language transfer for bimanual tasks.

Hyperparameters follow Appendix E with the same defaults.
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from typing import Optional

from .config import BimanualIPConfig
from .model import BimanualGraphDiffusionPolicy
from .dataset import BimanualPseudoDemoDataset
from ..train import train_occupancy_network  # Phase 1 reused directly


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Bimanual Model Training
# ──────────────────────────────────────────────────────────────────────

def train_bimanual_model(
    shapenet_root: str,
    encoder_ckpt: str,
    save_dir: str,
    cfg: BimanualIPConfig = None,
    device: str = 'cuda',
    resume_from: Optional[str] = None,
):
    """
    Train the BimanualGraphDiffusionPolicy on bimanual pseudo-demonstrations.
    """
    print("=" * 60)
    print("Phase 2: Training Bimanual Graph Diffusion Policy")
    print("=" * 60)

    cfg = cfg or BimanualIPConfig()
    model = BimanualGraphDiffusionPolicy(cfg).to(device)

    # Load pre-trained geometry encoder and freeze it
    if os.path.exists(encoder_ckpt):
        state = torch.load(encoder_ckpt, map_location=device)
        model.geo_encoder.load_state_dict(state)
        print(f"Loaded geometry encoder from {encoder_ckpt}")
    else:
        print(f"Warning: encoder checkpoint not found at {encoder_ckpt}")

    for param in model.geo_encoder.parameters():
        param.requires_grad = False
    print("Geometry encoder frozen.")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params_list)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    # Optimiser
    optimiser = torch.optim.AdamW(
        trainable_params_list, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Learning rate schedule with cosine cooldown
    import math

    def lr_lambda(step):
        if step < cfg.max_optim_steps - cfg.cooldown_steps:
            return 1.0
        progress = (step - (cfg.max_optim_steps - cfg.cooldown_steps)) / cfg.cooldown_steps
        return max(0.01, 0.5 * (1.0 + math.cos(progress * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_fp16)

    # Dataset
    dataset = BimanualPseudoDemoDataset(shapenet_root, cfg)
    loader = DataLoader(dataset, batch_size=None)

    # Resume
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimiser.load_state_dict(ckpt['optimiser'])
        start_step = ckpt.get('step', 0)
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    os.makedirs(save_dir, exist_ok=True)

    running_loss = 0.0
    for step, batch in enumerate(loader, start=start_step + 1):
        if step > cfg.max_optim_steps:
            break

        with autocast(enabled=cfg.use_fp16):
            loss = model(batch)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(trainable_params_list, cfg.grad_clip)
        scaler.step(optimiser)
        scaler.update()
        scheduler.step()

        running_loss += loss.item()

        if step % 100 == 0:
            avg_loss = running_loss / 100
            lr = optimiser.param_groups[0]['lr']
            print(f"  Step {step}/{cfg.max_optim_steps}  "
                  f"Loss: {avg_loss:.5f}  LR: {lr:.2e}")
            running_loss = 0.0

        if step % 10_000 == 0:
            ckpt_path = os.path.join(save_dir, f'bimanual_step{step}.pt')
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'cfg': cfg,
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, 'bimanual_final.pt')
    torch.save({
        'step': step,
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'cfg': cfg,
    }, final_path)
    print(f"Training complete. Final model saved to {final_path}")


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train Bimanual Instant Policy')
    parser.add_argument('--shapenet_root', type=str, required=True,
                       help='Path to ShapeNet dataset')
    parser.add_argument('--save_dir', type=str,
                       default='./checkpoints_bimanual',
                       help='Directory to save checkpoints')
    parser.add_argument('--phase', type=int, default=0, choices=[0, 1, 2],
                       help='0=both, 1=encoder only, 2=model only')
    parser.add_argument('--encoder_ckpt', type=str,
                       default='./checkpoints_ip/geo_encoder.pt',
                       help='Path to pre-trained geometry encoder')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='cuda')

    # Bimanual-specific flags
    parser.add_argument('--no_coordinate_edges', action='store_true',
                       help='Disable cross-arm edges in sigma')
    parser.add_argument('--no_bimanual_edges', action='store_true',
                       help='Disable cross-arm edges in phi')
    parser.add_argument('--no_sync_edges', action='store_true',
                       help='Disable cross-arm edges in psi')
    parser.add_argument('--scene_frame', type=str, default='midpoint',
                       choices=['midpoint', 'world'],
                       help='Reference frame for scene encoding')

    args = parser.parse_args()

    # Build config
    cfg = BimanualIPConfig(
        enable_coordinate_edges=not args.no_coordinate_edges,
        enable_bimanual_edges=not args.no_bimanual_edges,
        enable_sync_edges=not args.no_sync_edges,
        scene_frame=args.scene_frame,
    )

    if args.phase in (0, 1):
        # Phase 1: reuse single-arm occupancy network pre-training
        train_occupancy_network(
            args.shapenet_root,
            args.encoder_ckpt,
            device=args.device,
        )

    if args.phase in (0, 2):
        train_bimanual_model(
            args.shapenet_root,
            args.encoder_ckpt,
            args.save_dir,
            cfg=cfg,
            device=args.device,
            resume_from=args.resume,
        )


if __name__ == '__main__':
    main()
