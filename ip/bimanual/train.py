"""
Bimanual Instant Policy training script.

Three-phase training (same structure as single-arm):
  Phase 1: Pre-train geometry encoder (reuse single-arm, unchanged).
  Phase 2: Train bimanual graph diffusion model on pseudo-demonstrations.
  Phase 3: (Future) Language transfer for bimanual tasks.

Supports:
  - Single GPU:  python -m ip.bimanual.train --phase 2 ...
  - Multi-GPU:   torchrun --nproc_per_node=NUM_GPUS -m ip.bimanual.train --phase 2 ...

Hyperparameters follow Appendix E with the same defaults.
"""
import os
import math
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import argparse
from typing import Optional

from .config import BimanualIPConfig
from .model import BimanualGraphDiffusionPolicy
from .dataset import BimanualPseudoDemoDataset
from ..train import train_occupancy_network  # Phase 1 reused directly


# ──────────────────────────────────────────────────────────────────────
# DDP helpers
# ──────────────────────────────────────────────────────────────────────

def _is_distributed():
    return dist.is_available() and dist.is_initialized()


def _local_rank():
    return int(os.environ.get('LOCAL_RANK', 0))


def _world_size():
    return dist.get_world_size() if _is_distributed() else 1


def _is_main():
    return _local_rank() == 0


def _setup_distributed():
    """Initialise DDP if launched via torchrun."""
    if 'LOCAL_RANK' not in os.environ:
        return False
    dist.init_process_group(backend='nccl')
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    return True


def _cleanup_distributed():
    if _is_distributed():
        dist.destroy_process_group()


def _print_main(*args, **kwargs):
    """Print only on rank 0."""
    if _is_main():
        print(*args, **kwargs)


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

    Supports single-GPU and multi-GPU (DDP) training.
    In DDP mode, each GPU generates its own pseudo-demo data independently
    and gradients are all-reduced across GPUs, effectively multiplying
    the batch size by the number of GPUs.
    """
    distributed = _is_distributed()
    local_rank = _local_rank()
    world_size = _world_size()

    if distributed:
        device = f'cuda:{local_rank}'

    _print_main("=" * 60)
    _print_main("Phase 2: Training Bimanual Graph Diffusion Policy")
    if distributed:
        _print_main(f"  DDP mode: {world_size} GPUs")
    else:
        _print_main(f"  Single GPU mode")
    _print_main("=" * 60)

    cfg = cfg or BimanualIPConfig()
    model = BimanualGraphDiffusionPolicy(cfg).to(device)

    # Load pre-trained geometry encoder and freeze it
    if os.path.exists(encoder_ckpt):
        state = torch.load(encoder_ckpt, map_location=device)
        model.geo_encoder.load_state_dict(state)
        _print_main(f"Loaded geometry encoder from {encoder_ckpt}")
    else:
        _print_main(f"Warning: encoder checkpoint not found at {encoder_ckpt}")

    for param in model.geo_encoder.parameters():
        param.requires_grad = False
    _print_main("Geometry encoder frozen.")

    # Wrap with DDP
    if distributed:
        model = DDP(model, device_ids=[local_rank],
                    find_unused_parameters=False)
        raw_model = model.module
    else:
        raw_model = model

    # Count parameters
    total_params = sum(p.numel() for p in raw_model.parameters())
    trainable_params_list = [p for p in raw_model.parameters() if p.requires_grad]
    trainable_count = sum(p.numel() for p in trainable_params_list)
    _print_main(f"Total parameters: {total_params:,}")
    _print_main(f"Trainable parameters: {trainable_count:,}")

    # Optimiser
    optimiser = torch.optim.AdamW(
        trainable_params_list, lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    # Learning rate schedule with cosine cooldown
    def lr_lambda(step):
        if step < cfg.max_optim_steps - cfg.cooldown_steps:
            return 1.0
        progress = (step - (cfg.max_optim_steps - cfg.cooldown_steps)) / cfg.cooldown_steps
        return max(0.01, 0.5 * (1.0 + math.cos(progress * math.pi)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    scaler = GradScaler(enabled=cfg.use_fp16)

    # Dataset — each GPU gets its own independent IterableDataset
    # (pseudo-demos are generated on-the-fly with random seeds)
    _print_main("Initialising dataset...")
    dataset = BimanualPseudoDemoDataset(shapenet_root, cfg)
    loader = DataLoader(dataset, batch_size=None)

    # Resume
    start_step = 0
    if resume_from and os.path.exists(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        raw_model.load_state_dict(ckpt['model'])
        optimiser.load_state_dict(ckpt['optimiser'])
        start_step = ckpt.get('step', 0)
        _print_main(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    if _is_main():
        os.makedirs(save_dir, exist_ok=True)

    _print_main("Starting training loop...")
    import time
    running_loss = 0.0
    log_steps = 0
    step_t0 = time.time()
    for step, batch in enumerate(loader, start=start_step + 1):
        if step > cfg.max_optim_steps:
            break

        # Log first few steps individually so user knows training started
        if step <= start_step + 3:
            _print_main(f"  Step {step}: generating batch + forward pass...")

        with autocast(enabled=cfg.use_fp16):
            loss = model(batch)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        torch.nn.utils.clip_grad_norm_(trainable_params_list, cfg.grad_clip)
        old_scale = scaler.get_scale()
        scaler.step(optimiser)
        scaler.update()
        # Only step scheduler when optimizer actually stepped (avoids warning)
        if old_scale <= scaler.get_scale():
            scheduler.step()

        running_loss += loss.item()
        log_steps += 1

        if step <= start_step + 3:
            _print_main(f"  Step {step}: loss={loss.item():.4f}  "
                        f"({time.time()-step_t0:.1f}s)")
            step_t0 = time.time()

        if step % 100 == 0:
            avg_loss = running_loss / max(log_steps, 1)
            lr = optimiser.param_groups[0]['lr']
            gpu_info = f"  [x{world_size} GPUs]" if distributed else ""
            _print_main(f"  Step {step}/{cfg.max_optim_steps}  "
                        f"Loss: {avg_loss:.5f}  LR: {lr:.2e}{gpu_info}")
            running_loss = 0.0
            log_steps = 0

        if step % 10_000 == 0 and _is_main():
            ckpt_path = os.path.join(save_dir, f'bimanual_step{step}.pt')
            torch.save({
                'step': step,
                'model': raw_model.state_dict(),
                'optimiser': optimiser.state_dict(),
                'cfg': cfg,
            }, ckpt_path)
            print(f"  Checkpoint saved to {ckpt_path}")

    # Final save
    if _is_main():
        final_path = os.path.join(save_dir, 'bimanual_final.pt')
        torch.save({
            'step': step,
            'model': raw_model.state_dict(),
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

    # Setup DDP if launched via torchrun
    distributed = _setup_distributed()

    # Build config
    cfg = BimanualIPConfig(
        enable_coordinate_edges=not args.no_coordinate_edges,
        enable_bimanual_edges=not args.no_bimanual_edges,
        enable_sync_edges=not args.no_sync_edges,
        scene_frame=args.scene_frame,
    )

    if args.phase in (0, 1):
        # Phase 1 only on main process (single-GPU, shared encoder)
        if _is_main():
            train_occupancy_network(
                args.shapenet_root,
                args.encoder_ckpt,
                device=args.device if not distributed else f'cuda:{_local_rank()}',
            )
        if distributed:
            dist.barrier()  # wait for Phase 1 to finish

    if args.phase in (0, 2):
        train_bimanual_model(
            args.shapenet_root,
            args.encoder_ckpt,
            args.save_dir,
            cfg=cfg,
            device=args.device,
            resume_from=args.resume,
        )

    _cleanup_distributed()


if __name__ == '__main__':
    main()
