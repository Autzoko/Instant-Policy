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
    Pre-train the geometry encoder φ_e as part of an occupancy network
    (Mescheder et al. 2019, Appendix A).

    For each step:
      1. Load a random ShapeNet mesh.
      2. Sample 2048 surface points as encoder input.
      3. Sample query points: 50% near surface, 50% random in bounding box.
      4. Compute ground truth occupancy via mesh containment test.
      5. Train encoder + decoder with BCE loss.
    """
    import numpy as np

    print("=" * 60)
    print("Phase 1: Pre-training Geometry Encoder (Occupancy Network)")
    print("=" * 60)

    cfg = IPConfig()
    model = OccupancyNetwork(cfg).to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    scaler = GradScaler()

    from .pseudo_demo import load_shapenet_meshes

    mesh_paths = load_shapenet_meshes(shapenet_root, max_objects=50000)
    if not mesh_paths:
        print(f"Warning: No meshes found in {shapenet_root}. "
              "Using random point clouds for demo.")

    print(f"  Loaded {len(mesh_paths)} ShapeNet meshes")

    # Check trimesh availability (needed for proper occupancy)
    try:
        import trimesh
        _has_trimesh = True
        print("  trimesh available: using mesh-based occupancy ground truth")
    except ImportError:
        _has_trimesh = False
        print("  trimesh not available: using approximate occupancy labels")

    model.train()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)

    for step in range(1, num_steps + 1):
        # ── 1. Load a random ShapeNet mesh ──────────────────────
        pcd_np, query_np, gt_np = _sample_occupancy_data(
            mesh_paths, _has_trimesh, cfg.num_pcd_points)

        pcd_tensor = torch.from_numpy(pcd_np).float().unsqueeze(0).to(device)
        queries = torch.from_numpy(query_np).float().unsqueeze(0).to(device)
        gt = torch.from_numpy(gt_np).float().unsqueeze(0).to(device)

        with autocast():
            logits = model(pcd_tensor, queries)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, gt)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        if step % 1000 == 0:
            print(f"  Step {step}/{num_steps}  Loss: {loss.item():.4f}")

        # Save intermediate checkpoint every 25K steps
        if step % 25_000 == 0:
            mid_path = save_path.replace('.pt', f'_step{step}.pt')
            torch.save(model.encoder.state_dict(), mid_path)
            print(f"  Intermediate checkpoint: {mid_path}")

    # Save final encoder weights
    torch.save(model.encoder.state_dict(), save_path)
    print(f"Geometry encoder saved to {save_path}")
    return model.encoder


def _sample_occupancy_data(mesh_paths, has_trimesh, num_pcd_points=2048):
    """
    Sample one training example for occupancy pre-training.

    Returns:
        pcd:     (num_pcd_points, 3)  surface points (encoder input)
        queries: (2048, 3)            query points (50% surface, 50% random)
        gt:      (2048, 1)            occupancy labels
    """
    import numpy as np

    if not mesh_paths or not has_trimesh:
        # Skip to fallback
        pcd = _generate_random_shape(num_pcd_points)
        surface_queries = pcd[:1024] + np.random.randn(1024, 3).astype(np.float32) * 0.01
        random_queries = np.random.uniform(-0.2, 0.2, size=(1024, 3)).astype(np.float32)
        queries = np.concatenate([surface_queries, random_queries], axis=0)
        gt = np.concatenate([
            np.ones((1024, 1), dtype=np.float32),
            np.zeros((1024, 1), dtype=np.float32),
        ], axis=0)
        return pcd, queries, gt

    mesh_path = mesh_paths[np.random.randint(len(mesh_paths))]

    if has_trimesh:
        import trimesh
        try:
            mesh = trimesh.load(mesh_path, force='mesh')
            # Normalise to unit bounding box centered at origin
            mesh.vertices -= mesh.bounding_box.centroid
            scale = mesh.bounding_box.extents.max()
            if scale > 1e-8:
                mesh.vertices /= scale
            mesh.vertices *= 0.1  # scale to ~10cm (typical object size)

            # ── Encoder input: 2048 surface points ──
            pcd = mesh.sample(num_pcd_points).astype(np.float32)

            # ── Query points: 50% near surface, 50% random ──
            # Surface queries: points on surface + small noise
            surface_pts = mesh.sample(1024).astype(np.float32)
            surface_queries = surface_pts + np.random.randn(1024, 3).astype(np.float32) * 0.005

            # Random queries: uniform in bounding box with padding
            bbox_min = pcd.min(axis=0) - 0.05
            bbox_max = pcd.max(axis=0) + 0.05
            random_queries = np.random.uniform(
                bbox_min, bbox_max, size=(1024, 3)
            ).astype(np.float32)

            queries = np.concatenate([surface_queries, random_queries], axis=0)

            # ── Ground truth: mesh containment test ──
            # Mescheder et al. 2019: use mesh to determine inside/outside
            if mesh.is_watertight:
                occupancy = mesh.contains(queries).astype(np.float32)
            else:
                # Fallback for non-watertight meshes:
                # surface queries (with small noise) → likely occupied
                # Use distance-based heuristic
                closest, dist, _ = trimesh.proximity.closest_point(mesh, queries)
                occupancy = (dist < 0.008).astype(np.float32)

            gt = occupancy.reshape(-1, 1)
            return pcd, queries, gt

        except Exception:
            # Mesh load failed — use fallback for this sample
            pcd = _generate_random_shape(num_pcd_points)
            surface_queries = pcd[:1024] + np.random.randn(1024, 3).astype(np.float32) * 0.01
            random_queries = np.random.uniform(-0.2, 0.2, size=(1024, 3)).astype(np.float32)
            queries = np.concatenate([surface_queries, random_queries], axis=0)
            gt = np.concatenate([
                np.ones((1024, 1), dtype=np.float32),
                np.zeros((1024, 1), dtype=np.float32),
            ], axis=0)
            return pcd, queries, gt

    # Should not reach here (early return above handles no-trimesh case)
    raise RuntimeError("Unreachable")


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
