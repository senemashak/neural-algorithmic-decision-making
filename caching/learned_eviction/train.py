"""Train the CacheEvictionTransformer on Belady labels.

Usage (from caching-experiments/):
    python3 -m learned_eviction.train --data-dir data/run_20260422 --device cuda
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import EvictionDataset
from .model import CacheEvictionTransformer


def build_loaders(data_dir: Path, k: int, w: int, batch_size: int, num_workers: int):
    """Default to LRU+LFU+ARC trace files in data_dir. Trace-level 80/10/10 split."""
    from .dataset import default_split

    trace_files = sorted(data_dir.glob("*_traces.npy"))
    assert trace_files, f"no *_traces.npy under {data_dir}"

    train_ds, val_ds, test_ds = default_split(
        trace_files,
        cache_size=k,
        context_window=w,
        val_frac=0.1,
        test_frac=0.1,
        seed=0,
        cache_root=data_dir.parent / "belady_cache",
    )
    print(f"train {len(train_ds):,} | val {len(val_ds):,} | test {len(test_ds):,}")

    split_info = {
        "files": [p.name for p in trace_files],
        "train": train_ds.row_indices_per_file,
        "val": val_ds.row_indices_per_file,
        "test": test_ds.row_indices_per_file,
    }

    common = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    loaders = (
        DataLoader(train_ds, shuffle=True, drop_last=True, **common),
        DataLoader(val_ds, shuffle=False, **common),
        DataLoader(test_ds, shuffle=False, **common),
    )
    return loaders, split_info


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    for batch in loader:
        cache = batch["cache"].to(device)
        seq = batch["seq"].to(device)
        label = batch["label"].to(device)
        logits = model(cache, seq)
        loss_sum += loss_fn(logits, label).item()
        correct += (logits.argmax(-1) == label).sum().item()
        total += label.numel()
    return loss_sum / max(total, 1), correct / max(total, 1)


def train(args):
    device = torch.device(args.device)
    data_dir = Path(args.data_dir)

    (train_loader, val_loader, _test_loader), split_info = build_loaders(
        data_dir, k=args.k, w=args.context_window,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    log_path = Path(args.log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "split.json", "w") as f:
        json.dump({"data_dir": str(data_dir), **split_info}, f)
    print(f"split saved to {log_path / 'split.json'}")

    model = CacheEvictionTransformer(
        vocab_size=args.vocab_size,
        cache_size=args.k,
        context_window=args.context_window,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {n_params:,} params")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float("inf")
    step = 0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            cache = batch["cache"].to(device, non_blocking=True)
            seq = batch["seq"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            logits = model(cache, seq)
            loss = loss_fn(logits, label)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
            if step % args.log_every == 0:
                acc = (logits.argmax(-1) == label).float().mean().item()
                print(f"epoch {epoch} step {step}  loss {loss.item():.4f}  acc {acc:.3f}")

        val_loss, val_acc = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"[epoch {epoch}] val_loss {val_loss:.4f}  val_acc {val_acc:.3f}  ({dt:.1f}s)")
        with open(log_path / "train.log", "a") as f:
            f.write(json.dumps({"epoch": epoch, "val_loss": val_loss,
                                "val_acc": val_acc, "dt_s": dt}) + "\n")

        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "args": vars(args)},
                       log_path / "best.pt")
            print(f"  ↳ saved new best to {log_path / 'best.pt'}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=str, default="data/run_20260422")
    p.add_argument("--log-dir", type=str, default="learned_eviction/runs/default")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--context-window", type=int, default=1024)
    p.add_argument("--vocab-size", type=int, default=513)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
