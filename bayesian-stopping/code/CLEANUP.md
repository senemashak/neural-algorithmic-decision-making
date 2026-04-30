# v2 Cleanup Log

Cleanup pass performed prior to launching the v2 sweep. The goal was
surface-debt cleanup, not algorithmic change. Algorithms (oracle solver,
posterior recursion, training loop) are unchanged.

## Removed files

- `finish_sweep.sh` — v1 manual-fallback runner that existed because
  `run_sweep.sh` deadlocked on `wait` after `exec > >(tee ...)`. The
  deadlock is fixed in v2's `run_sweep.sh` (explicit-PID wait), so this
  script has no remaining purpose.
- `run_experiments.sh` — original single-model validation pipeline from
  the pre-sweep "calibration" run on D2_cv. Superseded by `run_sweep.sh`'s
  Phase 2 per-model analysis loop, which runs the same four scripts on
  each of the six models.

## New file

- `config.py` — central source of truth for v2 constants: problem
  parameters (MU_0, TAU0, SIGMA_VALUES, RHO_VALUES, N), ADP solver
  resolution (K, J), training hyperparameters (BATCH_SIZE, LR,
  WARMUP_FRAC, N_STEPS_CV, N_STEPS_ACT), eval splits (N_VAL, N_TEST,
  SEED_VAL, SEED_TEST), architecture (D_EMB, N_LAYERS, N_HEADS),
  diagnostics (ATTENTION_SNAPSHOT_N_SEQS), and SWEEP_ROOT path. Every
  other module imports from here.

## Spec changes (v1 -> v2)

| parameter | v1 | v2 |
|---|---|---|
| TAU0 | varied per regime: $\sqrt{10}, 1, 1/\sqrt{10}$ | fixed: $10$ |
| SIGMA | fixed: $1$ | varied per regime: $\{1, 10, 100\}$ |
| RHO = $\sigma^2/\tau_0^2$ | $\{0.1, 1, 10\}$ | $\{0.01, 1, 100\}$ |
| N | $64$ | $256$ |
| K (ADP grid) | $256$ | $1024$ |
| J (Gauss-Hermite nodes) | $32$ | $64$ |
| N_STEPS_CV | $500{,}000$ | $200{,}000$ (n=256 ~4x slower per step) |
| N_STEPS_ACT | $200{,}000$ | $100{,}000$ (proportional reduction) |

Rationale for the sigma-vs-tau_0 swap is in the `config.py` module
docstring: tau_0 is not identifiable from a single in-context sequence
while sigma is, so varying sigma sharpens the OOD diagnostic by giving
the algorithm hypothesis a falsifiable in-context-detection signature.

## Refactors

- `dataset.py`: `DatasetConfig` defaults (`tau0_2`, `mu_0`, `n`) now
  come from `config.py`. `DATASETS` dict is built from
  `config.SIGMA_VALUES`. `build_val_test` defaults `seed_val`,
  `seed_test`, `N_val`, `N_test`, `K`, `J` to config values.
- `oracle.py`: `solve_adp(K=None, J=None)` resolves to
  `config.K`/`config.J` if not provided. New module-level `import config`.
- `model.py`: `GPTStopper` constructor takes `sigma` (default 1.0); two
  non-learnable persistent buffers `input_scale = 1/sigma` and
  `output_scale = sigma` (cv) / `1.0` (act) are baked in. Forward pass
  scales inputs and outputs internally so callers always see raw $X$
  in / raw $\widehat{C}$ or logit out, while the transformer operates
  at unit scale regardless of training regime. The buffers serialize
  with the checkpoint and load via `load_state_dict`.
- `train.py`: `import config`; defaults pulled from `config`; passes
  `cfg.sigma` into `GPTStopper(sigma=...)`.
- `eval_common.py`: `import config`; `load_model` uses `config.D_EMB /
  N_LAYERS / N_HEADS` defaults and `sigma=1.0` for construction (overwritten
  by the saved buffers in `load_state_dict`).
- `threshold_traj.py` / `threshold_traj_v2.py`: `n=64` hardcode replaced
  with `config.N`.
- `visualize_results.py`: `SWEEP = config.SWEEP_ROOT`.
- `visualize_attention_data_dependence.py`: `SWEEP = config.SWEEP_ROOT`;
  `T_HI` derived as `config.N - 2`; caption strings now interpolate the
  actual t-range instead of hardcoded `[3, 62]`.
- `run_sweep.sh`: now reads `SWEEP_ROOT` from `config.py` via a Python
  one-liner; uses explicit-PID `wait` to avoid the v1 tee+wait deadlock;
  invokes `threshold_traj_v2.py`, `visualize_results.py`, and
  `visualize_attention_data_dependence.py` in Phase 3 (which v1 didn't);
  D2_cv now runs sequentially after the parallel block (matches the
  user's GPU schedule for v2).

## Calibration references

A grep for the string "calibration" finds three remaining hits, all in
historical comments — none in active code paths:
- `make_per_model_artifacts.py:78` — comment marker for the val-baseline
  block ("calibration-style").
- `make_per_model_artifacts.py:427` — comment about replacing
  `calibration_report.md`.
- `attention_analysis.py:2` — module docstring describing what attention
  analysis was originally written for.

These are kept per the earlier user instruction to leave historical
calibration references in comments.

## TODO/FIXME/XXX/DEPRECATED

A grep for these markers across `code/*.py *.sh` returns zero hits.

## What was NOT changed

- Algorithmic content of `oracle.py`, `dataset.py` (sampling, labeling),
  `train.py` (training loop), `eval_indist.py`, `eval_ood.py`,
  `attention_analysis.py`, `make_per_model_artifacts.py`. Just imports,
  defaults, and (for `model.py`) the input/output scaling buffers.
- `research-notes.tex` — Phase 6 will update this after evaluations.
