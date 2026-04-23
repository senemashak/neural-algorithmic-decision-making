"""Plot train + validation loss/accuracy curves from a training run.

Usage (from caching/):
    python3 -m learned_eviction.plot_curves --run-dir learned_eviction/runs/default
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _smooth(xs: list[float], window: int) -> tuple[list[int], list[float]]:
    if len(xs) < window:
        return [], []
    arr = np.asarray(xs, dtype=np.float64)
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    # Align smoothed value to the window midpoint.
    offset = window // 2
    indices = list(range(offset, offset + len(smoothed)))
    return indices, smoothed.tolist()


def plot_run(run_dir: Path, window: int = 20):
    train = _load_jsonl(run_dir / "train_loss.jsonl")
    val = _load_jsonl(run_dir / "val_loss.jsonl")
    assert train, "no training logs found"
    assert val, "no validation logs found"

    train_steps = [d["step"] for d in train]
    train_loss = [d["loss"] for d in train]
    train_acc = [d["acc"] for d in train]
    val_steps = [d["step"] for d in val]
    val_loss = [d["val_loss"] for d in val]
    val_acc = [d["val_acc"] for d in val]

    sm_idx, sm_loss = _smooth(train_loss, window)
    sm_train_steps = [train_steps[i] for i in sm_idx] if sm_idx else []
    sm_idx_a, sm_acc = _smooth(train_acc, window)
    sm_train_steps_a = [train_steps[i] for i in sm_idx_a] if sm_idx_a else []

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax_loss.plot(train_steps, train_loss, alpha=0.25, color="tab:blue", label="train (per log step)")
    if sm_loss:
        ax_loss.plot(sm_train_steps, sm_loss, color="tab:blue", linewidth=2,
                     label=f"train (moving avg, {window} pts)")
    ax_loss.plot(val_steps, val_loss, "o-", color="tab:orange", markersize=7,
                 linewidth=2, label="val (per epoch)")
    ax_loss.axhline(np.log(32), linestyle="--", color="grey", alpha=0.6,
                    label="chance (log 32)")
    ax_loss.set_xlabel("training step")
    ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss")
    ax_loss.legend()
    ax_loss.grid(True, alpha=0.3)

    ax_acc.plot(train_steps, train_acc, alpha=0.25, color="tab:blue", label="train (per log step)")
    if sm_acc:
        ax_acc.plot(sm_train_steps_a, sm_acc, color="tab:blue", linewidth=2,
                    label=f"train (moving avg, {window} pts)")
    ax_acc.plot(val_steps, val_acc, "o-", color="tab:orange", markersize=7,
                linewidth=2, label="val (per epoch)")
    ax_acc.axhline(1.0 / 32, linestyle="--", color="grey", alpha=0.6, label="chance (1/32)")
    ax_acc.set_xlabel("training step")
    ax_acc.set_ylabel("slot-selection accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend()
    ax_acc.grid(True, alpha=0.3)
    ax_acc.set_ylim(0, max(0.6, max(train_acc + val_acc) * 1.05))

    plt.suptitle(f"Training run: {run_dir.name}", fontsize=12)
    plt.tight_layout()

    out_png = run_dir / "training_curves.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"saved {out_png}")

    # Also print summary table of per-epoch val numbers.
    print("\nper-epoch validation:")
    print(f"{'epoch':>5}  {'step':>6}  {'val_loss':>10}  {'val_acc':>9}  {'dt_s':>8}")
    for d in val:
        print(f"{d['epoch']:>5}  {d['step']:>6}  {d['val_loss']:>10.4f}  "
              f"{d['val_acc']:>9.4f}  {d['dt_s']:>8.1f}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=str, default="learned_eviction/runs/default")
    p.add_argument("--window", type=int, default=20, help="moving-average window (log steps)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    plot_run(Path(args.run_dir), window=args.window)
