#!/bin/bash
# Sweep runner: train all 6 models, then run the per-model analysis pipeline,
# then build cross-model artifacts. SWEEP_ROOT is read from config.py.
#
# Five GPUs / six models: D1_cv, D3_cv, D1_act, D2_act, D3_act run in parallel
# (one per GPU, GPUs 0..4); D2_cv runs sequentially after the first cv job
# frees a GPU.
#
# Path layout under $SWEEP_ROOT (= config.SWEEP_ROOT):
#   D1_cv/  D2_cv/  D3_cv/  D1_act/  D2_act/  D3_act/
#       <model_id>_best_model.pt
#       <model_id>_log.csv
#       <model_id>_train.log
#       <model_id>_attention_snapshot.npz
#       experiments/<model_id>/   # per-model report bundle
#   experiments/
#       sweep_summary.csv
#       run_sweep.log
#       threshold_trajectories.png
#       threshold_trajectories_zoomed.png
#       figures/
#
# Deadlock fix vs v1: the previous run_sweep.sh used `exec > >(tee ...)`
# followed by a bare `wait`, which hangs because `wait` cannot reap the tee
# subshell that became a child via the exec redirect. v2 instead uses
# `wait $pid1 $pid2 ...` with explicit PIDs and tees from the caller side
# (`./run_sweep.sh 2>&1 | tee log` or via nohup), avoiding the issue.

set -e

CODE_DIR=/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/code
SWEEP_ROOT=$(cd "$CODE_DIR" && python3 -c 'import config; print(config.SWEEP_ROOT)')
mkdir -p "$SWEEP_ROOT/experiments"

echo "=== run_sweep.sh ==="
echo "  sweep_root: $SWEEP_ROOT"
echo "  code_dir:   $CODE_DIR"
echo "  start:      $(date -Is)"
echo

cd "$CODE_DIR"

# ---------------------------------------------------------------------------
# Phase 1 — launch 5 training jobs in parallel
# ---------------------------------------------------------------------------

echo "=== Phase 1: launching 5 training jobs in parallel ==="
echo

declare -A JOB_TRAIN JOB_SUP JOB_GPU
JOB_TRAIN[D1_cv]=1;  JOB_SUP[D1_cv]=cv;   JOB_GPU[D1_cv]=0
JOB_TRAIN[D3_cv]=3;  JOB_SUP[D3_cv]=cv;   JOB_GPU[D3_cv]=1
JOB_TRAIN[D1_act]=1; JOB_SUP[D1_act]=act; JOB_GPU[D1_act]=2
JOB_TRAIN[D2_act]=2; JOB_SUP[D2_act]=act; JOB_GPU[D2_act]=3
JOB_TRAIN[D3_act]=3; JOB_SUP[D3_act]=act; JOB_GPU[D3_act]=4

declare -a PIDS=()
for mid in D1_cv D3_cv D1_act D2_act D3_act; do
  ds=${JOB_TRAIN[$mid]}
  sup=${JOB_SUP[$mid]}
  gpu=${JOB_GPU[$mid]}
  out_dir="$SWEEP_ROOT/$mid"
  mkdir -p "$out_dir"
  log="$out_dir/${mid}_train.log"
  echo "  launching $mid: ds=D_$ds, sup=$sup, gpu=$gpu, log=$log"
  ( python3 -u train.py \
      --dataset_id "$ds" --supervision "$sup" \
      --model_id "$mid" --output_dir "$out_dir" \
      --gpu_id "$gpu" \
  ) > "$log" 2>&1 &
  PIDS+=($!)
done

echo
echo "  waiting for ${#PIDS[@]} parallel jobs to finish (PIDs: ${PIDS[*]})"
echo "  budget: ~2h per cv at n=256, ~1h per act"

# Wait on each PID explicitly (avoids the v1 tee+wait deadlock).
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    echo "  FATAL: job pid=$pid exited non-zero"
    exit 1
  fi
done
echo "  all parallel jobs finished at $(date -Is)"
echo

# ---------------------------------------------------------------------------
# Phase 1b — D2_cv sequentially (slot freed up by an earlier cv finishing)
# ---------------------------------------------------------------------------

echo "=== Phase 1b: D2_cv on GPU 0 (sequential) ==="
mid=D2_cv
out_dir="$SWEEP_ROOT/$mid"
mkdir -p "$out_dir"
log="$out_dir/${mid}_train.log"
echo "  launching $mid: ds=D_2, sup=cv, gpu=0, log=$log"
python3 -u train.py \
  --dataset_id 2 --supervision cv \
  --model_id "$mid" --output_dir "$out_dir" \
  --gpu_id 0 > "$log" 2>&1
echo "  D2_cv finished at $(date -Is)"
echo

# ---------------------------------------------------------------------------
# Phase 2 — per-model analysis (eval indist/ood, attention, report)
# ---------------------------------------------------------------------------

echo "=== Phase 2: per-model analysis on all 6 models ==="
echo

# (model_id, file_prefix, model_dir, train_dataset, supervision)
ANALYZE() {
  local mid="$1" prefix="$2" mdir="$3" ds="$4" sup="$5"
  local pmd="$mdir/experiments/$mid"
  echo "--- $mid (file_prefix=$prefix, train=D_$ds, sup=$sup) ---"
  mkdir -p "$pmd"

  python3 eval_indist.py \
    --model_id "$mid" --file_prefix "$prefix" \
    --model_dir "$mdir" --train_dataset "$ds" \
    --output_dir "$pmd"

  for j in 1 2 3; do
    if [ "$j" != "$ds" ]; then
      python3 eval_ood.py \
        --model_id "$mid" --file_prefix "$prefix" \
        --model_dir "$mdir" --train_dataset "$ds" --eval_dataset "$j" \
        --output_dir "$pmd"
    fi
  done

  python3 attention_analysis.py \
    --model_id "$mid" --file_prefix "$prefix" \
    --model_dir "$mdir" --output_dir "$pmd"

  python3 make_per_model_artifacts.py \
    --model_id "$mid" --file_prefix "$prefix" \
    --model_dir "$mdir" --per_model_dir "$pmd" \
    --train_dataset "$ds" --supervision "$sup"
}

for mid in D1_cv D2_cv D3_cv D1_act D2_act D3_act; do
  ds=${JOB_TRAIN[$mid]:-2}     # D2_cv is not in JOB_TRAIN; default to 2
  sup=${JOB_SUP[$mid]:-cv}     # D2_cv is not in JOB_SUP; default to cv
  ANALYZE "$mid" "$mid" "$SWEEP_ROOT/$mid" "$ds" "$sup"
done

echo

# ---------------------------------------------------------------------------
# Phase 3 — cross-model threshold trajectories (cv-supervised models only)
# ---------------------------------------------------------------------------

echo "=== Phase 3: cross-model threshold_traj (cv-only auto-filtered) ==="
MODELS_ARG="D1_cv:$SWEEP_ROOT/D1_cv/D1_cv_best_model.pt:1,\
D2_cv:$SWEEP_ROOT/D2_cv/D2_cv_best_model.pt:2,\
D3_cv:$SWEEP_ROOT/D3_cv/D3_cv_best_model.pt:3"

python3 threshold_traj.py \
  --models "$MODELS_ARG" \
  --output_dir "$SWEEP_ROOT/experiments"

# threshold_traj_v2: same three models, distinct colours, hard-zoomed band.
python3 threshold_traj_v2.py \
  --ckpt_d1 "$SWEEP_ROOT/D1_cv/D1_cv_best_model.pt" \
  --ckpt_d2 "$SWEEP_ROOT/D2_cv/D2_cv_best_model.pt" \
  --ckpt_d3 "$SWEEP_ROOT/D3_cv/D3_cv_best_model.pt" \
  --output_dir "$SWEEP_ROOT/experiments/figures"

# Cross-model figures (heatmap, payoff bars, agreement matrix, attention).
python3 visualize_results.py
python3 visualize_attention_data_dependence.py
echo

# ---------------------------------------------------------------------------
# Phase 4 — concat per-model summary_table.csv -> sweep_summary.csv
# ---------------------------------------------------------------------------

echo "=== Phase 4: concat summary_table.csv -> sweep_summary.csv ==="
SUMMARY_OUT="$SWEEP_ROOT/experiments/sweep_summary.csv"
python3 - <<EOF
import csv
from pathlib import Path
import config

inputs = [
    config.SWEEP_ROOT / mid / "experiments" / mid / "summary_table.csv"
    for mid in ("D1_cv", "D2_cv", "D3_cv", "D1_act", "D2_act", "D3_act")
]
rows, all_cols = [], []
for p in inputs:
    if not p.exists():
        print(f"  WARN: missing {p}; skipping"); continue
    with open(p) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
            for c in r.fieldnames:
                if c not in all_cols:
                    all_cols.append(c)

with open("$SUMMARY_OUT", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=all_cols, restval="", extrasaction="ignore")
    w.writeheader()
    for row in rows:
        w.writerow(row)
print(f"  wrote {len(rows)} rows x {len(all_cols)} cols -> $SUMMARY_OUT")
EOF
echo

# ---------------------------------------------------------------------------
# Final inventory
# ---------------------------------------------------------------------------

echo "=== Final inventory ==="
echo "--- $SWEEP_ROOT ---"
find "$SWEEP_ROOT" -maxdepth 3 \( -type f -o -type d \) | sort
echo
echo "=== Done at $(date -Is) ==="
