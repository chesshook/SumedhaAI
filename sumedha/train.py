"""Two-GPU-capable training loop for SUMEDHA AI."""
from __future__ import annotations

import argparse
import math
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from .model import SumedhaNet


class ChessDataset(Dataset):
    def __init__(self, x, move_indices, move_probs, values, indices):
        self.x = torch.from_numpy(np.asarray(x[indices], dtype=np.float32))
        self.moves = torch.from_numpy(np.asarray(move_indices[indices], dtype=np.int64))
        self.probs = torch.from_numpy(np.asarray(move_probs[indices], dtype=np.float32))
        self.values = torch.from_numpy(np.asarray(values[indices], dtype=np.float32))

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.moves[index], self.probs[index], self.values[index]


def log(message: str, rank: int = 0) -> None:
    if rank == 0:
        print(f"[SUMEDHA train] {message}", flush=True)


def save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def loss_for_batch(model, batch, device):
    x, move_indices, move_probs, values = batch
    x = x.to(device, non_blocking=True)
    move_indices = move_indices.to(device, non_blocking=True)
    move_probs = move_probs.to(device, non_blocking=True)
    values = values.to(device, non_blocking=True)

    policy_logits, predicted_values = model(x)
    log_probabilities = F.log_softmax(policy_logits.float(), dim=1)
    chosen_log_probabilities = log_probabilities.gather(1, move_indices)
    policy_loss = -(chosen_log_probabilities * move_probs).sum(dim=1).mean()
    value_loss = F.smooth_l1_loss(predicted_values.float(), values)
    total = policy_loss + 1.5 * value_loss
    return total, policy_loss.detach(), value_loss.detach()


def reduce_metrics(totals, device, distributed):
    metrics = torch.tensor(totals, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    count = max(1.0, metrics[3].item())
    return (
        metrics[0].item() / count,
        metrics[1].item() / count,
        metrics[2].item() / count,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=24, help="Total target epoch, including resumed epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size PER GPU.")
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--blocks", type=int, default=6)
    parser.add_argument("--embedding", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.0007)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20261121)
    args = parser.parse_args()

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for SUMEDHA training. Select GPU T4 x2 in Kaggle.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group(backend="nccl", init_method="env://")

    torch.set_num_threads(1)
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

    data_path = Path(args.data)
    if not data_path.is_file():
        raise FileNotFoundError(data_path)
    with np.load(data_path, allow_pickle=False) as data:
        x = data["x"]
        move_indices = data["move_indices"]
        move_probs = data["move_probs"]
        values = data["values"]
        metadata = str(data["metadata"].item()) if "metadata" in data else "{}"

    count = len(values)
    if count < 32:
        raise RuntimeError(f"Dataset only has {count} samples.")
    if x.shape[1:] != (19, 8, 8):
        raise ValueError(f"Expected [N,19,8,8] board tensors, received {x.shape}")
    if move_indices.ndim != 2 or move_probs.shape != move_indices.shape:
        raise ValueError("Policy indices and probabilities must be matching [N,K] arrays.")

    permutation = np.random.default_rng(args.seed).permutation(count)
    validation_count = max(1, int(count * 0.10))
    validation_indices = permutation[:validation_count]
    training_indices = permutation[validation_count:]
    if len(training_indices) < 2:
        raise RuntimeError("Not enough training positions after validation split.")

    train_data = ChessDataset(x, move_indices, move_probs, values, training_indices)
    valid_data = ChessDataset(x, move_indices, move_probs, values, validation_indices)

    train_sampler = DistributedSampler(
        train_data, num_replicas=world_size, rank=rank, shuffle=True, seed=args.seed
    ) if distributed else None
    valid_sampler = DistributedSampler(
        valid_data, num_replicas=world_size, rank=rank, shuffle=False
    ) if distributed else None

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=train_sampler is None,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=False,
    )
    valid_loader = DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, sampler=valid_sampler,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=args.workers > 0, drop_last=False,
    )

    model_config = {
        "channels": args.channels,
        "blocks": args.blocks,
        "embedding": args.embedding,
    }
    base_model = SumedhaNet(**model_config).to(device)
    model = DDP(base_model, device_ids=[local_rank], output_device=local_rank) if distributed else base_model
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / "latest.pt"
    best_path = output_dir / "best.pt"
    start_epoch = 0
    best_validation = float("inf")
    global_step = 0

    if latest_path.is_file():
        try:
            checkpoint = torch.load(latest_path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(latest_path, map_location=device)
        base_model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        # Extend the cosine schedule when a later Kaggle session requests more epochs.
        # Otherwise a resumed run would inherit the old schedule's near-zero final LR.
        if args.epochs > scheduler.T_max:
            scheduler.T_max = args.epochs
            position = min(scheduler.last_epoch, scheduler.T_max)
            factor = (1.0 + math.cos(math.pi * position / scheduler.T_max)) / 2.0
            for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
                group["lr"] = scheduler.eta_min + (base_lr - scheduler.eta_min) * factor
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_validation = float(checkpoint.get("best_validation", best_validation))
        global_step = int(checkpoint.get("global_step", 0))
        log(f"Resumed from {latest_path} at epoch {start_epoch + 1}.", rank)
    else:
        log(f"Starting from random initialization; {count} labelled positions loaded.", rank)

    log(
        f"Model: channels={args.channels}, blocks={args.blocks}, parameters="
        f"{sum(p.numel() for p in base_model.parameters()):,}", rank
    )
    log(
        f"World size={world_size}; per-GPU batch={args.batch_size}; "
        f"effective batch={args.batch_size * world_size}; train={len(train_data)}; "
        f"validation={len(valid_data)}", rank
    )

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_totals = [0.0, 0.0, 0.0, 0.0]

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                total_loss, policy_loss, value_loss = loss_for_batch(model, batch, device)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            batch_count = batch[0].shape[0]
            train_totals[0] += total_loss.detach().item() * batch_count
            train_totals[1] += policy_loss.item() * batch_count
            train_totals[2] += value_loss.item() * batch_count
            train_totals[3] += batch_count
            global_step += 1

        model.eval()
        valid_totals = [0.0, 0.0, 0.0, 0.0]
        with torch.no_grad():
            for batch in valid_loader:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    total_loss, policy_loss, value_loss = loss_for_batch(model, batch, device)
                batch_count = batch[0].shape[0]
                valid_totals[0] += total_loss.item() * batch_count
                valid_totals[1] += policy_loss.item() * batch_count
                valid_totals[2] += value_loss.item() * batch_count
                valid_totals[3] += batch_count

        train_metrics = reduce_metrics(train_totals, device, distributed)
        valid_metrics = reduce_metrics(valid_totals, device, distributed)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        if distributed:
            dist.barrier()

        improved = valid_metrics[0] < best_validation
        if improved:
            best_validation = valid_metrics[0]

        if rank == 0:
            state = {
                "project": "SUMEDHA AI",
                "schema_version": 1,
                "epoch": epoch,
                "global_step": global_step,
                "model_config": model_config,
                "model": base_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_validation": best_validation,
                "train_metrics": train_metrics,
                "validation_metrics": valid_metrics,
                "dataset_metadata": metadata,
            }
            save_checkpoint(latest_path, state)
            if improved or not best_path.exists():
                save_checkpoint(best_path, state)

            log(
                f"epoch {epoch + 1:03d}/{args.epochs} | "
                f"train loss {train_metrics[0]:.4f} (policy {train_metrics[1]:.4f}, "
                f"value {train_metrics[2]:.4f}) | "
                f"val loss {valid_metrics[0]:.4f} (policy {valid_metrics[1]:.4f}, "
                f"value {valid_metrics[2]:.4f}) | lr {lr:.2e} | "
                f"steps {global_step} | {'BEST' if improved else 'saved'}"
            )
        if distributed:
            dist.barrier()

    if rank == 0:
        log(f"Training complete. Latest: {latest_path}; best: {best_path}", rank)
        log("To continue training, increase SUMEDHA_EPOCHS and rerun the Kaggle cell.", rank)

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
