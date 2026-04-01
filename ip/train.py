"""
Training script for Instant Policy (Appendix E).

Two-phase training:
  Phase 1: Pre-train geometry encoder as occupancy network on ShapeNet.
  Phase 2: Train the full graph diffusion model on pseudo-demonstrations.

Hyperparameters (Appendix E):
  - AdamW optimiser, lr=1e-5, weight_decay=1e-4
  - 2.5M optimisation steps + 50K cooldown
  - float16 mixed precision
  - Single GPU (RTX 3080-ti, ~5 days)
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from typing import Optional

from .config import IPConfig
from .model import GraphDiffusionPolicy
from .geometry_encoder import OccupancyNetwork
from .dataset import PseudoDemoDataset


# ──────────────────────────────────────────────────────────────────────
# Phase 1: Occupancy Network Pre-training
# ──────────────────────────────────────────────────────────────────────

def train_occupancy_network(
    shapenet_root: str,
    save_path: str,
    num_steps: int = 100_000,
    lr: float = 1e-4,
    batch_size: int = 32,
    device: str = 'cuda',
):
    """
    Pre-train the geometry encoder φ_e as part of an occupancy network.

    Training data: sample objects from ShapeNet, sample query points
    (50% on surface, 50% random), predict occupancy.
    """
    print("=" * 60)
    print("Phase 1: Pre-training Geometry Encoder (Occupancy Network)")
    print("=" * 60)

    cfg = IPConfig()
    model = OccupancyNetwork(cfg).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    # Simple synthetic data loop (replace with real ShapeNet loader)
    from .pseudo_demo import load_shapenet_meshes, sample_scene, render_point_clouds

    mesh_paths = load_shapenet_meshes(shapenet_root, max_objects=50000)
    if not mesh_paths:
        print(f"Warning: No meshes found in {shapenet_root}. "
              "Using random point clouds for demo.")

    model.train()
    for step in range(1, num_steps + 1):
        # Generate training data
        if mesh_paths:
            objects = sample_scene(mesh_paths, num_objects=1)
            pcd = render_point_clouds(objects, None)
        else:
            # Fallback: random shape
            pcd = _generate_random_shape(2048)

        pcd_tensor = torch.from_numpy(pcd[:2048]).float().unsqueeze(0).to(device)

        # Sample query points: 50% near surface, 50% random
        surface_queries = pcd_tensor + torch.randn_like(pcd_tensor) * 0.01
        random_queries = torch.rand(1, 2048, 3, device=device) * 0.4 - 0.2
        queries = torch.cat([surface_queries[:, :1024], random_queries[:, :1024]], dim=1)

        # Ground truth: surface queries → 1, random → probably 0
        gt = torch.cat([
            torch.ones(1, 1024, 1, device=device),
            torch.zeros(1, 1024, 1, device=device),
        ], dim=1)

        with autocast():
            logits = model(pcd_tensor, queries)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, gt)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        if step % 1000 == 0:
            print(f"  Step {step}/{num_steps}  Loss: {loss.item():.4f}")

    # Save encoder weights
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Geometry encoder saved to {save_path}")
    return model.encoder


def _generate_random_shape(n_points: int) -> 'np.ndarray':
    """Generate a random ellipsoid point cloud for testing."""
    import numpy as np
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    a, b, c = np.random.uniform(0.02, 0.08, 3)
    x = a * np.sin(phi) * np.cos(theta)
    y = b * np.sin(phi) * np.sin(theta)
    z = c * np.cos(phi)
    pts = np.stack([x, y, z], axis=1) + np.random.randn(n_points, 3) * 0.002
    return pts.astype(np.float32)


# ──────────────────────────────────────────────────────────────────────
# Phase 2: Full Model Training
# ──────────────────────────────────────────────────────────────────────

def train_model(
    shapenet_root: str,
    encoder_ckpt: str,
    save_dir: str,
    cfg: IPConfig = None,
    device: str = 'cuda',
    resume_from: Optional[str] = None,
):
    """
    Train the full GraphDiffusionPolicy model on pseudo-demonstrations.
    """
    print("=" * 60)
    print("Phase 2: Training Graph Diffusion Policy")
    print("=" * 60)

    cfg = cfg or IPConfig()
    model = GraphDiffusionPolicy(cfg).to(device)

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

    # Optimiser (Appendix E)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(
        trainable_params, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Learning rate schedule with cosine cooldown
    def lr_lambda(step):
        if step < cfg.max_optim_steps - cfg.cooldown_steps:
            return 1.0
        progress = (step - (cfg.max_optim_steps - cfg.cooldown_steps)) / cfg.cooldown_steps
        return max(0.01, 0.5 * (1.0 + __import__('math').cos(progress * __import__('math').pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_fp16)

    # Dataset
    dataset = PseudoDemoDataset(shapenet_root, cfg)
    loader = DataLoader(dataset, batch_size=None)  # IterableDataset yields singles

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
        torch.nn.utils.clip_grad_norm_(trainable_params, cfg.grad_clip)
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
            ckpt_path = os.path.join(save_dir, f'model_step{step}.pt')
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'cfg': cfg,
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, 'model_final.pt')
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
    parser = argparse.ArgumentParser(description='Train Instant Policy')
    parser.add_argument('--shapenet_root', type=str, required=True,
                       help='Path to ShapeNet dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ip',
                       help='Directory to save checkpoints')
    parser.add_argument('--phase', type=int, default=0,
                       choices=[0, 1, 2],
                       help='0=both, 1=encoder only, 2=model only')
    parser.add_argument('--encoder_ckpt', type=str,
                       default='./checkpoints_ip/geo_encoder.pt')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    if args.phase in (0, 1):
        train_occupancy_network(
            args.shapenet_root,
            args.encoder_ckpt,
            device=args.device,
        )

    if args.phase in (0, 2):
        train_model(
            args.shapenet_root,
            args.encoder_ckpt,
            args.save_dir,
            device=args.device,
            resume_from=args.resume,
        )


if __name__ == '__main__':
    main()
