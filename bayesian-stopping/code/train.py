"""
Train one model in the sweep.

CLI:
    --dataset_id    1, 2, or 3
    --supervision   cv | act
    --model_id      e.g. D1_cv (used as filename prefix)
    --output_dir    where to write <model_id>_*.{pt, csv, npz}
    --gpu_id        which CUDA device (default 0)
    --n_steps       default 500_000 for cv, 200_000 for act
    --warmup_steps  default n_steps // 5
    --batch_size    default 64
    --lr            default 1e-4
    --val_every     default 5_000
    --log_every     default 1_000
    --seed_val      default 42
    --seed_test     default 43
    --seed_train    default 0
    --smoke         200-step smoke test (overrides n_steps & schedule)

Loss:
    cv  -> MSE on positions t=1..n-1, target y_cv from oracle labels.
    act -> BCE-with-logits on positions t=1..n-1, target y_act = 1[X_t >= C_hat_t].
At policy time:
    cv  -> accept iff X_t >= predicted C_hat[t]
    act -> accept iff predicted logit[t] > 0  (sigmoid > 0.5)

Outputs:
    <output_dir>/<model_id>_best_model.pt        (lowest val_loss checkpoint)
    <output_dir>/<model_id>_log.csv              (1000-step periodic log)
    <output_dir>/<model_id>_attention_snapshot.npz  (post-training snapshot)
"""

import argparse
import collections
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import config
from dataset import DATASETS, build_val_test, stream_batches
from model import GPTStopper
from oracle import posterior_path_batch


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def make_lr_lambda(total_steps: int, warmup_steps: int):
    def lr_lambda(epoch: int) -> float:
        step = epoch + 1
        if step <= warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return lr_lambda


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, bundle, device, supervision, sigma2, batch_size=4096):
    model.eval()
    X = bundle.X_val
    y_cv = bundle.y_cv_val
    y_act = bundle.y_act_val
    N, n = X.shape

    total_loss_num = 0.0
    total_count = 0
    payoffs = np.empty(N)
    agree_correct = 0
    agree_total = 0

    for i in range(0, N, batch_size):
        s = slice(i, i + batch_size)
        Xb = torch.as_tensor(X[s], device=device, dtype=torch.float32)
        Cb = model(Xb)                                            # (B, n)
        ycv_b = torch.as_tensor(y_cv[s], device=device, dtype=torch.float32)
        yact_b = torch.as_tensor(y_act[s], device=device, dtype=torch.float32)

        if supervision == "cv":
            diff = Cb[:, :n - 1] - ycv_b
            # Scale-invariant val loss: divide raw MSE by sigma^2 so the
            # value is comparable across regimes (matches the train-side
            # 1/sigma^2 factor).
            total_loss_num += (diff ** 2).sum().item() / sigma2
            accept = (Xb[:, :n - 1] >= Cb[:, :n - 1])
        else:  # act
            logits = Cb[:, :n - 1]
            total_loss_num += F.binary_cross_entropy_with_logits(
                logits, yact_b, reduction="sum"
            ).item()
            accept = logits > 0.0
        total_count += yact_b.numel()

        any_acc = accept.any(dim=1)
        first_idx = accept.float().argmax(dim=1)
        last = torch.full_like(first_idx, n - 1)
        stop = torch.where(any_acc, first_idx, last)
        payoffs[s] = Xb.gather(1, stop.unsqueeze(1)).squeeze(1).cpu().numpy()

        agree_correct += (accept.float() == yact_b).sum().item()
        agree_total += yact_b.numel()

    val_loss = total_loss_num / total_count
    val_payoff = float(payoffs.mean())
    val_agree = agree_correct / agree_total
    model.train()
    return val_loss, val_payoff, val_agree


# ---------------------------------------------------------------------------
# Attention snapshot
# ---------------------------------------------------------------------------

def attention_snapshot(model, bundle, cfg, device, npz_path: Path,
                       supervision: str, n_seq: int = 256):
    model.eval()
    X = bundle.X_val[:n_seq]
    y_cv = bundle.y_cv_val[:n_seq]
    y_act = bundle.y_act_val[:n_seq]
    n = X.shape[1]
    with torch.no_grad():
        Xb = torch.as_tensor(X, device=device, dtype=torch.float32)
        out, attn = model(Xb, return_attn=True)
    attn_np = attn.cpu().numpy().astype(np.float32)
    out_np = out.cpu().numpy().astype(np.float32)
    mu_path, _ = posterior_path_batch(X, cfg.mu_0, cfg.tau0_2, cfg.sigma2)

    eps = 1e-12
    H = -np.where(attn_np > 0, attn_np * np.log(attn_np + eps), 0.0).sum(axis=-1)
    H_per_lh = H[:, :, :, 1:].mean(axis=(0, 3))                   # (L, M)

    save_kwargs = dict(
        attn=attn_np,
        X=X.astype(np.float32),
        mu_t=mu_path.astype(np.float32),
        oracle_C_hat_t=y_cv.astype(np.float32),
        oracle_y_act=y_act.astype(np.float32),
        entropy_table=H_per_lh.astype(np.float32),
        supervision=np.array(supervision),
    )
    if supervision == "cv":
        save_kwargs["model_C_hat_t"] = out_np[:, :n - 1].astype(np.float32)
    else:
        save_kwargs["model_act_logit_t"] = out_np[:, :n - 1].astype(np.float32)
    np.savez(npz_path, **save_kwargs)
    model.train()


# ---------------------------------------------------------------------------
# Path verification
# ---------------------------------------------------------------------------

def ensure_writable(path: Path) -> Path:
    abs_path = path.resolve()
    print(f"output dir: {abs_path}", flush=True)
    if not abs_path.exists():
        try:
            abs_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            sys.exit(f"FATAL: cannot create {abs_path}: {e}")
    test = abs_path / ".write_test"
    try:
        test.touch()
        test.unlink()
    except Exception as e:
        sys.exit(f"FATAL: not writable: {abs_path}: {e}")
    print(f"  -> verified writable", flush=True)
    return abs_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_id", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--supervision", required=True, choices=["cv", "act"])
    p.add_argument("--model_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpu_id", type=int, default=0)
    p.add_argument("--n_steps", type=int, default=None)
    p.add_argument("--warmup_steps", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--val_every", type=int, default=5_000)
    p.add_argument("--log_every", type=int, default=1_000)
    p.add_argument("--seed_val", type=int, default=config.SEED_VAL)
    p.add_argument("--seed_test", type=int, default=config.SEED_TEST)
    p.add_argument("--seed_train", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()

    # Default schedule per supervision (cv polishes longer than act).
    if args.n_steps is None:
        args.n_steps = config.N_STEPS_ACT if args.supervision == "act" else config.N_STEPS_CV
    if args.warmup_steps is None:
        args.warmup_steps = int(args.n_steps * config.WARMUP_FRAC)

    if args.smoke:
        args.n_steps, args.warmup_steps = 200, 50
        args.val_every, args.log_every = 100, 50

    out = ensure_writable(Path(args.output_dir))
    BEST_PATH = out / f"{args.model_id}_best_model.pt"
    CSV_PATH = out / f"{args.model_id}_log.csv"
    NPZ_PATH = out / f"{args.model_id}_attention_snapshot.npz"

    if not torch.cuda.is_available():
        sys.exit("FATAL: cuda not available")
    if args.gpu_id >= torch.cuda.device_count():
        sys.exit(f"FATAL: --gpu_id={args.gpu_id} >= "
                 f"{torch.cuda.device_count()} available GPUs")
    device = torch.device(f"cuda:{args.gpu_id}")
    print(f"[{args.model_id}] device={device}  smoke={args.smoke}")
    print(f"[{args.model_id}] argv={sys.argv}")

    cfg = DATASETS[args.dataset_id]
    print(f"[{args.model_id}] dataset={cfg.name} (rho={cfg.rho})  "
          f"supervision={args.supervision}")
    print(f"[{args.model_id}] schedule: n_steps={args.n_steps}, "
          f"warmup={args.warmup_steps}, batch={args.batch_size}, lr={args.lr}")

    print(f"[{args.model_id}] building val/test splits...")
    bundle = build_val_test(cfg, seed_val=args.seed_val, seed_test=args.seed_test)

    torch.set_float32_matmul_precision("high")
    model = GPTStopper(
        n=cfg.n,
        d_emb=config.D_EMB,
        n_layers=config.N_LAYERS,
        n_heads=config.N_HEADS,
        supervision=args.supervision,
        sigma=cfg.sigma,                        # baked-in input/output scale buffers
    ).to(device)
    print(f"[{args.model_id}] params: {model.num_params():,}")
    train_model = torch.compile(model, mode="reduce-overhead")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lr_lambda=make_lr_lambda(args.n_steps, args.warmup_steps),
    )

    rng = np.random.default_rng(args.seed_train)
    it = stream_batches(cfg, args.batch_size, rng, bundle.C_hat, bundle.grids)

    smoothed = collections.deque(maxlen=1000)
    history = {k: [] for k in ("step", "smoothed_train_loss", "val_loss",
                                "val_payoff", "val_agreement_oracle",
                                "throughput_steps_per_sec", "wall_clock_sec")}
    best_val_loss = float("inf")
    best_val_step = -1

    print(f"[{args.model_id}] starting training")
    t_start = time.perf_counter()

    for step in range(1, args.n_steps + 1):
        X_np, y_cv_np, y_act_np = next(it)
        X_t = torch.as_tensor(X_np, device=device, dtype=torch.float32)
        if args.supervision == "cv":
            y_t = torch.as_tensor(y_cv_np, device=device, dtype=torch.float32)
        else:
            y_t = torch.as_tensor(y_act_np, device=device, dtype=torch.float32)

        out_pred = train_model(X_t)
        if args.supervision == "cv":
            # Both `out_pred` and `y_t` are at raw (sigma-scale) units, so MSE
            # scales as sigma^2. Divide by sigma^2 to keep gradient magnitudes
            # regime-invariant; the val-loss reporting also uses this scaled
            # value (so it is comparable across regimes as well).
            loss = F.mse_loss(out_pred[:, :cfg.n - 1], y_t) / cfg.sigma2
        else:
            loss = F.binary_cross_entropy_with_logits(
                out_pred[:, :cfg.n - 1], y_t,
            )

        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        smoothed.append(loss.item())

        if step % args.log_every == 0:
            elapsed = time.perf_counter() - t_start
            thr = step / elapsed
            sm = float(np.mean(smoothed))
            if step % args.val_every == 0:
                v_loss, v_payoff, v_agree = evaluate(
                    model, bundle, device, args.supervision, cfg.sigma2,
                )
                if v_loss < best_val_loss:
                    best_val_loss, best_val_step = v_loss, step
                    torch.save({
                        "step": step,
                        "state_dict": model.state_dict(),
                        "val_loss": v_loss,
                        "supervision": args.supervision,
                        "model_id": args.model_id,
                        "train_dataset": args.dataset_id,
                    }, BEST_PATH)
                print(f"[{args.model_id}] step {step:>7d}  "
                      f"train={sm:.3e}  val={v_loss:.3e}  "
                      f"payoff={v_payoff:.4f}  agree(BO)={v_agree:.4f}  "
                      f"thr={thr:.0f}s/s  t={elapsed:.0f}s",
                      flush=True)
            else:
                v_loss = v_payoff = v_agree = float("nan")
                print(f"[{args.model_id}] step {step:>7d}  "
                      f"train={sm:.3e}  thr={thr:.0f}s/s  t={elapsed:.0f}s",
                      flush=True)
            history["step"].append(step)
            history["smoothed_train_loss"].append(sm)
            history["val_loss"].append(v_loss)
            history["val_payoff"].append(v_payoff)
            history["val_agreement_oracle"].append(v_agree)
            history["throughput_steps_per_sec"].append(thr)
            history["wall_clock_sec"].append(elapsed)

    # ---- write log.csv ----
    cols = ["step", "smoothed_train_loss", "val_loss", "val_payoff",
            "val_agreement_oracle", "throughput_steps_per_sec",
            "wall_clock_sec"]
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(history["step"])):
            w.writerow([history[c][i] for c in cols])
    print(f"[{args.model_id}] wrote CSV: {CSV_PATH}")

    # ---- attention snapshot from best checkpoint ----
    print(f"[{args.model_id}] loading best ckpt (step {best_val_step}, "
          f"val_loss={best_val_loss:.3e})")
    ckpt = torch.load(BEST_PATH, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["state_dict"])
    print(f"[{args.model_id}] writing attention snapshot...")
    attention_snapshot(model, bundle, cfg, device, NPZ_PATH, args.supervision)

    elapsed = time.perf_counter() - t_start
    print(f"[{args.model_id}] DONE  total_wall_clock={elapsed:.1f}s "
          f"({elapsed/60:.2f} min)  best_val_loss={best_val_loss:.3e}  "
          f"best_step={best_val_step}")


if __name__ == "__main__":
    main()
