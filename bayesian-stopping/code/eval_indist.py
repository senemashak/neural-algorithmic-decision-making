"""
In-distribution evaluation: load <file_prefix>_best_model.pt from --model_dir
and evaluate on the train-dataset's val and test splits.

CLI:
    --model_id        canonical id (e.g. D2_cv); used in JSON metadata.
    --file_prefix     filename prefix; defaults to model_id. Pass explicitly
                      if checkpoint files use a non-canonical naming.
    --model_dir       directory containing <file_prefix>_best_model.pt.
    --train_dataset   1, 2, or 3 (the model's training distribution).
    --eval_dataset    optional override for OOD evaluation in this script
                      (defaults to train_dataset).
    --supervision     optional; auto-detected from checkpoint metadata.
    --n_test          default 10000.
    --output_dir      target dir for indist_eval.{json,csv}.

Outputs:
    <output_dir>/indist_eval.json  — split-stratified payoffs + agreements
    <output_dir>/indist_eval.csv   — per-baseline (val_*, test_*) table
"""

import argparse
import csv
import json
from pathlib import Path

import torch

from dataset import DATASETS, build_val_test
from eval_common import (
    ensure_writable,
    evaluate_on_dataset,
    load_model,
)


def write_indist_csv(metrics: dict, csv_path: Path):
    """Per-baseline table with both val and test columns."""
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "baseline",
            "val_payoff", "val_payoff_se", "val_stop_mean", "val_agreement",
            "test_payoff", "test_payoff_se", "test_stop_mean", "test_agreement",
        ])
        v = metrics["val"]
        t = metrics["test"]
        for name in ("bayes_optimal", "plug_in", "prior_only",
                     "myopic", "secretary"):
            w.writerow([
                name,
                f"{v['baseline_payoffs'][name]:.6f}",
                f"{v['baseline_payoff_ses'][name]:.6f}",
                f"{v['baseline_stop_means'][name]:.2f}",
                f"{v['agreements'][name]:.6f}",
                f"{t['baseline_payoffs'][name]:.6f}",
                f"{t['baseline_payoff_ses'][name]:.6f}",
                f"{t['baseline_stop_means'][name]:.2f}",
                f"{t['agreements'][name]:.6f}",
            ])
        w.writerow([
            "offline",
            f"{v['baseline_payoffs']['offline']:.6f}",
            f"{v['baseline_payoff_ses']['offline']:.6f}",
            f"{v['baseline_stop_means']['offline']:.2f}",
            f"{v['stoptime_matches']['offline_stoptime_match']:.6f}",
            f"{t['baseline_payoffs']['offline']:.6f}",
            f"{t['baseline_payoff_ses']['offline']:.6f}",
            f"{t['baseline_stop_means']['offline']:.2f}",
            f"{t['stoptime_matches']['offline_stoptime_match']:.6f}",
        ])


def print_indist_summary(metrics: dict, header: str):
    print(f"\n=== {header} ===")
    for split in ("val", "test"):
        m = metrics[split]
        print(f"  [{split}]  model payoff: {m['model_payoff']:.4f}  "
              f"(SE {m['model_payoff_se']:.4f})  "
              f"E[tau]={m['model_stop_mean']:.2f}  "
              f"agree(BO)={m['agreements']['bayes_optimal']:.4f}")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--file_prefix", default=None)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--train_dataset", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--eval_dataset", type=int, default=None, choices=[1, 2, 3])
    p.add_argument("--supervision", default=None, choices=["cv", "act"])
    p.add_argument("--n_test", type=int, default=10_000)
    p.add_argument("--output_dir", required=True)
    args = p.parse_args()

    file_prefix = args.file_prefix or args.model_id
    if args.eval_dataset is None:
        args.eval_dataset = args.train_dataset

    out = ensure_writable(Path(args.output_dir))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_cfg = DATASETS[args.train_dataset]
    eval_cfg = DATASETS[args.eval_dataset]
    print(f"[{args.model_id}] train: {train_cfg.name}  eval: {eval_cfg.name}")

    ckpt_path = Path(args.model_dir) / f"{file_prefix}_best_model.pt"
    print(f"[{args.model_id}] loading: {ckpt_path}")
    model = load_model(str(ckpt_path), eval_cfg.n, device,
                       supervision=args.supervision)

    print(f"[{args.model_id}] building val/test (seeds 42/43)...")
    bundle = build_val_test(eval_cfg, seed_val=42, seed_test=43,
                            N_val=10_000, N_test=args.n_test)

    metrics_val = evaluate_on_dataset(model, eval_cfg, bundle.X_val,
                                       bundle.y_act_val, device)
    metrics_test = evaluate_on_dataset(model, eval_cfg, bundle.X_test,
                                        bundle.y_act_test, device)
    metrics = {
        "val": metrics_val,
        "test": metrics_test,
        "model_id":      args.model_id,
        "file_prefix":   file_prefix,
        "checkpoint":    str(ckpt_path.resolve()),
        "train_dataset": train_cfg.name,
        "train_rho":     train_cfg.rho,
        "eval_dataset":  eval_cfg.name,
        "eval_rho":      eval_cfg.rho,
        "supervision":   model.supervision,
        "n_val":         10_000,
        "n_test":        args.n_test,
    }

    json_path = out / "indist_eval.json"
    csv_path  = out / "indist_eval.csv"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    write_indist_csv(metrics, csv_path)

    print_indist_summary(
        metrics,
        header=f"In-distribution: {args.model_id} on {eval_cfg.name}",
    )
    print(f"wrote: {json_path}")
    print(f"wrote: {csv_path}")


if __name__ == "__main__":
    main()
