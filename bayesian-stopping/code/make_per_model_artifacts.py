"""
Regenerate per-model artifacts (train_val_curves.png, report.md,
summary_table.csv) from the existing log.csv, train.log, and JSON eval files.

CLI:
    --model_id      canonical id (e.g. D2_cv).
    --file_prefix   filename prefix; defaults to model_id.
    --model_dir     dir with <file_prefix>_log.csv and <file_prefix>_train.log.
    --per_model_dir output dir; should already contain indist_eval.json,
                    ood_eval_D{j}.json (for the two non-train datasets),
                    and attention_summary.json.
    --train_dataset 1, 2, or 3.
    --supervision   cv | act.

Inputs read:
    {model_dir}/{file_prefix}_log.csv
    {model_dir}/{file_prefix}_train.log
    {per_model_dir}/indist_eval.json
    {per_model_dir}/ood_eval_D{j}.json   for j in {1,2,3}\\{train_dataset}
    {per_model_dir}/attention_summary.json

Outputs written:
    {per_model_dir}/train_val_curves.png
    {per_model_dir}/report.md
    {per_model_dir}/summary_table.csv
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from eval_common import ensure_writable


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def read_log_csv(path: Path) -> dict:
    cols = {}
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        for col in header:
            cols[col] = []
        for row in r:
            for col, val in zip(header, row):
                try:
                    cols[col].append(float(val))
                except ValueError:
                    cols[col].append(float("nan"))
    return {k: np.array(v) for k, v in cols.items()}


def parse_train_log(path: Path) -> dict:
    text = path.read_text()
    out: dict = {"val_baselines": {}}
    m = re.search(r"params:\s*([\d,]+)", text)
    if m: out["n_params"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Total wall-clock:\s*([\d.]+)\s*s", text)
    if m: out["wall_clock_sec"] = float(m.group(1))
    m = re.search(r"total_wall_clock=([\d.]+)s", text)
    if m: out["wall_clock_sec"] = float(m.group(1))
    m = re.search(r"Average throughput:\s*([\d.]+) steps/sec", text)
    if m: out["avg_throughput"] = float(m.group(1))
    m = re.search(r"best_step=(\d+)", text)
    if m: out["best_step"] = int(m.group(1))
    m = re.search(r"Best checkpoint at step:\s*(\d+)", text)
    if m: out["best_step"] = int(m.group(1))

    # Val baselines, if logged at startup (calibration-style).
    block = re.search(
        r"computing baseline payoffs on val\.\.\.\n(.*?)(?:\n\nstarting training|\nparams:)",
        text, re.S,
    )
    if block:
        for line in block.group(1).splitlines():
            m = re.match(r"\s+(\w+)\s+([\d.\-eE]+)", line)
            if m: out["val_baselines"][m.group(1)] = float(m.group(2))

    m = re.search(r"Step to reach 1\.10x final val loss:\s*(\d+)", text)
    if m: out["step_1p10"] = int(m.group(1))
    m = re.search(r"Step to reach 1\.01x final val loss:\s*(\d+)", text)
    if m: out["step_1p01"] = int(m.group(1))
    return out


def load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# train_val_curves.png  (split-axis loss + payoff)
# ---------------------------------------------------------------------------

def plot_train_val_curves(log: dict, val_baselines: dict, supervision: str,
                           out_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    steps = log["step"]
    train_loss = log["smoothed_train_loss"]
    val_mask = ~np.isnan(log["val_loss"])
    val_step = steps[val_mask]
    val_loss = log["val_loss"][val_mask]
    val_payoff = log["val_payoff"][val_mask]

    ax1.plot(steps, train_loss, "b-", linewidth=1.2, alpha=0.85,
             label="train (smoothed)")
    ax1.scatter(val_step, val_loss, c="red", marker="o", s=22, zorder=5,
                label="val")
    ax1.set_yscale("log")
    ax1.set_ylabel("MSE" if supervision == "cv" else "BCE")
    ax1.set_title(f"Train and val loss ({supervision} supervision)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_step, val_payoff, "g-o", markersize=4,
             label="model val payoff")
    if "bayes_optimal" in val_baselines:
        ax2.axhline(val_baselines["bayes_optimal"], color="black",
                    linestyle="--",
                    label=f"Bayes-optimal = {val_baselines['bayes_optimal']:.4f}")
    if "plug_in" in val_baselines:
        ax2.axhline(val_baselines["plug_in"], color="tab:blue",
                    linestyle="--",
                    label=f"plug-in = {val_baselines['plug_in']:.4f}")
    ax2.set_xlabel("step")
    ax2.set_ylabel("E[payoff]")
    ax2.set_title("Val payoff vs step")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# report.md
# ---------------------------------------------------------------------------

def make_report_md(model_id, train_ds_name, supervision, train_log_meta, log,
                    indist, ood_jsons, attn) -> str:
    val_mask = ~np.isnan(log["val_loss"])
    finite_loss = log["val_loss"][val_mask]
    if finite_loss.size == 0:
        final_val_loss = float("nan")
        best_step = -1
        val_payoff_indist = float("nan")
    else:
        final_val_loss = float(finite_loss.min())
        best_step_idx = int(np.nanargmin(log["val_loss"]))
        best_step = int(log["step"][best_step_idx])
        val_payoff_indist = float(log["val_payoff"][best_step_idx])

    test_payoff_indist = indist["test"]["model_payoff"]
    val_payoff_indist_eval = indist["val"]["model_payoff"]
    bo_test = indist["test"]["baseline_payoffs"]["bayes_optimal"]
    plug_test = indist["test"]["baseline_payoffs"]["plug_in"]
    offline_test = indist["test"]["baseline_payoffs"]["offline"]

    ent = np.array(attn["per_lh_entropy"])
    L, M = ent.shape
    high = np.unravel_index(int(ent.argmax()), ent.shape)
    n_uniform = len(attn["uniform_candidates"])

    lines = []
    P = lines.append
    P(f"# Model report — `{model_id}`")
    P("")
    P(f"- **Model**: `{model_id}`")
    P(f"- **Train dataset**: {train_ds_name}")
    P(f"- **Supervision**: `{supervision}`")
    P(f"- **Eval seeds**: val=42, test=43; N=10,000 each")
    P("")
    P("## Training")
    P("")
    P(f"- **Param count**: {train_log_meta.get('n_params', float('nan')):,}")
    if "wall_clock_sec" in train_log_meta:
        P(f"- **Wall-clock**: {train_log_meta['wall_clock_sec']:.1f} s "
          f"({train_log_meta['wall_clock_sec']/60:.2f} min)")
    if "avg_throughput" in train_log_meta:
        P(f"- **Throughput**: {train_log_meta['avg_throughput']:.1f} steps/sec")
    P(f"- **Best checkpoint at step**: {best_step}")
    P(f"- **Final (= best) val loss** "
      f"({'MSE' if supervision == 'cv' else 'BCE'}): {final_val_loss:.3e}")
    P("")
    P(f"## In-distribution payoff ({train_ds_name})")
    P("")
    P(f"- **Val payoff (eval script, best ckpt)**: {val_payoff_indist_eval:.4f}  "
      f"(SE {indist['val']['model_payoff_se']:.4f})")
    P(f"- **Val payoff (training-time, best step)**: {val_payoff_indist:.4f}")
    P(f"- **Test payoff**: {test_payoff_indist:.4f}  "
      f"(SE {indist['test']['model_payoff_se']:.4f})")
    P(f"- **Gap to Bayes-optimal (test)**: "
      f"{test_payoff_indist - bo_test:+.4f}  (BO test = {bo_test:.4f})")
    P(f"- **Gap to plug-in (test)**: "
      f"{test_payoff_indist - plug_test:+.4f}  (plug-in test = {plug_test:.4f})")
    P("")
    P("**Normalized (test):**")
    P(f"- $R(\\pi_\\theta) / R(\\pi^\\star) = {test_payoff_indist / bo_test:.6f}$  "
      f"(1.0 = matches oracle)")
    P(f"- $R(\\pi_\\theta) / R(\\pi^{{\\mathrm{{offline}}}}) = "
      f"{test_payoff_indist / offline_test:.6f}$  (1.0 = matches hindsight)")
    P("")
    P(f"_Hindsight reference (offline payoff on {train_ds_name} test): "
      f"{offline_test:.4f}_ — not online; reported for scale only.")
    P("")
    P(f"## Per-baseline agreement (in-dist {train_ds_name}, N=10k each split)")
    P("")
    P("| baseline       | val agree | test agree | val payoff | test payoff |")
    P("|---             |---:       |---:        |---:        |---:         |")
    for name in ("bayes_optimal", "plug_in", "prior_only", "myopic", "secretary"):
        P(f"| {name:<14} "
          f"| {indist['val']['agreements'][name]:.4f} "
          f"| {indist['test']['agreements'][name]:.4f} "
          f"| {indist['val']['baseline_payoffs'][name]:.4f} "
          f"| {indist['test']['baseline_payoffs'][name]:.4f} |")
    P(f"| offline        "
      f"| {indist['val']['stoptime_matches']['offline_stoptime_match']:.4f}* "
      f"| {indist['test']['stoptime_matches']['offline_stoptime_match']:.4f}* "
      f"| {indist['val']['baseline_payoffs']['offline']:.4f} "
      f"| {indist['test']['baseline_payoffs']['offline']:.4f} |")
    P("")
    P("\\*offline shows stop-time match, not per-step agreement (offline isn't online).")
    P("")
    P("## OOD payoffs")
    P("")
    if ood_jsons:
        P("| eval | test payoff | test gap to BO | test R/R\\* | test R/R_off | "
          "test agree(BO) | test agree(plug) |")
        P("|---   |---:         |---:            |---:         |---:           "
          "|---:            |---:              |")
        for ds, j in sorted(ood_jsons.items()):
            t = j["test"]
            mp_t = t["model_payoff"]
            bp_t = t["baseline_payoffs"]["bayes_optimal"]
            off_t = t["baseline_payoffs"]["offline"]
            P(f"| {ds}  | {mp_t:.4f}     "
              f"| {mp_t - bp_t:+.4f}       "
              f"| {mp_t / bp_t:.4f}      "
              f"| {mp_t / off_t:.4f}       "
              f"| {t['agreements']['bayes_optimal']:.4f}        "
              f"| {t['agreements']['plug_in']:.4f}          |")
    else:
        P("_No OOD evaluations on file._")
    P("")
    P("## Attention")
    P("")
    P(f"- **Highest-entropy head**: L{int(high[0])} H{int(high[1])}, "
      f"mean entropy = {ent[high]:.3f}")
    P(f"- **Reference uniform** (mean log(t+1) over t=1..n-1): "
      f"{attn['reference_mean_log_t_plus_1']:.3f}")
    P(f"- **Uniform-candidate threshold** (0.9 × ref): "
      f"{attn['uniform_threshold']:.3f}")
    P(f"- **Uniform-candidate heads**: {n_uniform} / {L * M}")
    P(f"- **Min head entropy**: {attn['summary_stats']['min_entropy']:.3f}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# summary_table.csv
# ---------------------------------------------------------------------------

def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.12g}"
    return str(x)


def build_summary_columns(train_dataset: int):
    """User's spec: `<other1>` and `<other2>` are the two non-train datasets,
    in numeric order. Returns the column list with placeholders resolved."""
    others = sorted(set([1, 2, 3]) - {train_dataset})
    o1, o2 = others
    cols = [
        "model_id", "train_dataset", "supervision",
        "n_params", "n_train_steps", "wall_clock_sec",
        "final_val_loss",
        "val_payoff_indist", f"val_payoff_ood_D{o1}", f"val_payoff_ood_D{o2}",
        "test_payoff_indist", f"test_payoff_ood_D{o1}", f"test_payoff_ood_D{o2}",
        "gap_to_BO_val_indist",
        f"gap_to_BO_val_ood_D{o1}", f"gap_to_BO_val_ood_D{o2}",
        "gap_to_BO_test_indist",
        f"gap_to_BO_test_ood_D{o1}", f"gap_to_BO_test_ood_D{o2}",
        # Normalized payoffs: R(pi)/R(pi^*) and R(pi)/R(pi^offline)
        "val_payoff_indist_norm_BO", "test_payoff_indist_norm_BO",
        "val_payoff_indist_norm_offline", "test_payoff_indist_norm_offline",
        f"val_payoff_ood_D{o1}_norm_BO", f"test_payoff_ood_D{o1}_norm_BO",
        f"val_payoff_ood_D{o1}_norm_offline", f"test_payoff_ood_D{o1}_norm_offline",
        f"val_payoff_ood_D{o2}_norm_BO", f"test_payoff_ood_D{o2}_norm_BO",
        f"val_payoff_ood_D{o2}_norm_offline", f"test_payoff_ood_D{o2}_norm_offline",
        "agree_BO_val_indist", "agree_plugin_val_indist", "agree_priorOnly_val_indist",
        f"agree_BO_val_ood_D{o1}", f"agree_plugin_val_ood_D{o1}",
        f"agree_priorOnly_val_ood_D{o1}",
        f"agree_BO_val_ood_D{o2}", f"agree_plugin_val_ood_D{o2}",
        f"agree_priorOnly_val_ood_D{o2}",
        "agree_BO_test_indist", "agree_plugin_test_indist", "agree_priorOnly_test_indist",
        f"agree_BO_test_ood_D{o1}", f"agree_plugin_test_ood_D{o1}",
        f"agree_priorOnly_test_ood_D{o1}",
        f"agree_BO_test_ood_D{o2}", f"agree_plugin_test_ood_D{o2}",
        f"agree_priorOnly_test_ood_D{o2}",
        "n_uniform_heads", "max_head_entropy",
    ]
    return cols, o1, o2


def make_summary_row(model_id, train_dataset, supervision,
                      train_log_meta, log, indist, ood_jsons, attn) -> dict:
    cols, o1, o2 = build_summary_columns(train_dataset)
    train_ds_name = f"D_{train_dataset}"

    val_mask = ~np.isnan(log["val_loss"])
    if val_mask.any():
        final_val_loss = float(log["val_loss"][val_mask].min())
        n_train_steps = int(log["step"][-1])
    else:
        final_val_loss = float("nan")
        n_train_steps = 0

    row = {c: float("nan") for c in cols}
    row["model_id"]       = model_id
    row["train_dataset"]  = train_ds_name
    row["supervision"]    = supervision
    row["n_params"]       = train_log_meta.get("n_params", float("nan"))
    row["n_train_steps"]  = n_train_steps
    row["wall_clock_sec"] = train_log_meta.get("wall_clock_sec", float("nan"))
    row["final_val_loss"] = final_val_loss

    # In-dist (= eval on train_dataset's val/test)
    iv = indist["val"]; it = indist["test"]
    row["val_payoff_indist"]  = iv["model_payoff"]
    row["test_payoff_indist"] = it["model_payoff"]
    row["gap_to_BO_val_indist"]  = iv["model_payoff"] - iv["baseline_payoffs"]["bayes_optimal"]
    row["gap_to_BO_test_indist"] = it["model_payoff"] - it["baseline_payoffs"]["bayes_optimal"]
    row["val_payoff_indist_norm_BO"]       = iv["model_payoff"] / iv["baseline_payoffs"]["bayes_optimal"]
    row["test_payoff_indist_norm_BO"]      = it["model_payoff"] / it["baseline_payoffs"]["bayes_optimal"]
    row["val_payoff_indist_norm_offline"]  = iv["model_payoff"] / iv["baseline_payoffs"]["offline"]
    row["test_payoff_indist_norm_offline"] = it["model_payoff"] / it["baseline_payoffs"]["offline"]
    row["agree_BO_val_indist"]        = iv["agreements"]["bayes_optimal"]
    row["agree_plugin_val_indist"]    = iv["agreements"]["plug_in"]
    row["agree_priorOnly_val_indist"] = iv["agreements"]["prior_only"]
    row["agree_BO_test_indist"]        = it["agreements"]["bayes_optimal"]
    row["agree_plugin_test_indist"]    = it["agreements"]["plug_in"]
    row["agree_priorOnly_test_indist"] = it["agreements"]["prior_only"]

    for o, ds_id in (("o1", o1), ("o2", o2)):
        ds_name = f"D_{ds_id}"
        if ds_name not in ood_jsons:
            continue
        ojv = ood_jsons[ds_name]["val"]
        ojt = ood_jsons[ds_name]["test"]
        suffix = f"D{ds_id}"
        row[f"val_payoff_ood_{suffix}"]  = ojv["model_payoff"]
        row[f"test_payoff_ood_{suffix}"] = ojt["model_payoff"]
        row[f"gap_to_BO_val_ood_{suffix}"]  = (
            ojv["model_payoff"] - ojv["baseline_payoffs"]["bayes_optimal"]
        )
        row[f"gap_to_BO_test_ood_{suffix}"] = (
            ojt["model_payoff"] - ojt["baseline_payoffs"]["bayes_optimal"]
        )
        row[f"val_payoff_ood_{suffix}_norm_BO"]       = ojv["model_payoff"] / ojv["baseline_payoffs"]["bayes_optimal"]
        row[f"test_payoff_ood_{suffix}_norm_BO"]      = ojt["model_payoff"] / ojt["baseline_payoffs"]["bayes_optimal"]
        row[f"val_payoff_ood_{suffix}_norm_offline"]  = ojv["model_payoff"] / ojv["baseline_payoffs"]["offline"]
        row[f"test_payoff_ood_{suffix}_norm_offline"] = ojt["model_payoff"] / ojt["baseline_payoffs"]["offline"]
        row[f"agree_BO_val_ood_{suffix}"]        = ojv["agreements"]["bayes_optimal"]
        row[f"agree_plugin_val_ood_{suffix}"]    = ojv["agreements"]["plug_in"]
        row[f"agree_priorOnly_val_ood_{suffix}"] = ojv["agreements"]["prior_only"]
        row[f"agree_BO_test_ood_{suffix}"]        = ojt["agreements"]["bayes_optimal"]
        row[f"agree_plugin_test_ood_{suffix}"]    = ojt["agreements"]["plug_in"]
        row[f"agree_priorOnly_test_ood_{suffix}"] = ojt["agreements"]["prior_only"]

    row["n_uniform_heads"]  = len(attn["uniform_candidates"])
    row["max_head_entropy"] = float(np.array(attn["per_lh_entropy"]).max())
    return row, cols


def write_summary_csv(row: dict, cols: list, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerow([fmt(row[c]) for c in cols])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--file_prefix", default=None)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--per_model_dir", required=True)
    p.add_argument("--train_dataset", type=int, required=True, choices=[1, 2, 3])
    p.add_argument("--supervision", required=True, choices=["cv", "act"])
    args = p.parse_args()

    file_prefix = args.file_prefix or args.model_id
    model_dir = Path(args.model_dir).resolve()
    pmd = ensure_writable(Path(args.per_model_dir))

    log_csv = model_dir / f"{file_prefix}_log.csv"
    train_log = model_dir / f"{file_prefix}_train.log"
    if not log_csv.exists():
        sys.exit(f"FATAL: missing {log_csv}")
    if not train_log.exists():
        sys.exit(f"FATAL: missing {train_log}")

    print(f"reading log.csv: {log_csv}")
    log = read_log_csv(log_csv)
    print(f"reading train.log: {train_log}")
    train_meta = parse_train_log(train_log)

    indist_p = pmd / "indist_eval.json"
    if not indist_p.exists():
        sys.exit(f"FATAL: missing {indist_p}")
    indist = load_json(indist_p)

    ood_jsons = {}
    for ds_id in sorted(set([1, 2, 3]) - {args.train_dataset}):
        p_ood = pmd / f"ood_eval_D{ds_id}.json"
        if p_ood.exists():
            ood_jsons[f"D_{ds_id}"] = load_json(p_ood)
        else:
            print(f"  warning: missing {p_ood}")

    attn_p = pmd / "attention_summary.json"
    if not attn_p.exists():
        sys.exit(f"FATAL: missing {attn_p}")
    attn = load_json(attn_p)

    train_ds_name = f"D_{args.train_dataset}"

    # train_val_curves.png
    print("writing train_val_curves.png...")
    plot_train_val_curves(log, train_meta.get("val_baselines", {}),
                          args.supervision, pmd / "train_val_curves.png")

    # report.md (replaces calibration_report.md)
    print("writing report.md...")
    md = make_report_md(args.model_id, train_ds_name, args.supervision,
                        train_meta, log, indist, ood_jsons, attn)
    (pmd / "report.md").write_text(md)

    # summary_table.csv
    print("writing summary_table.csv...")
    row, cols = make_summary_row(args.model_id, args.train_dataset,
                                  args.supervision, train_meta, log,
                                  indist, ood_jsons, attn)
    write_summary_csv(row, cols, pmd / "summary_table.csv")

    print("\nfiles written:")
    for name in ("train_val_curves.png", "report.md", "summary_table.csv"):
        print(f"  {pmd / name}")


if __name__ == "__main__":
    main()
