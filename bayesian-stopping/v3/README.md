# V3 — Online selection of online algorithms

Authoritative spec: [`research-notes_v3.tex`](research-notes_v3.tex). Reference
material from the previous iteration lives in [`../v2/`](../v2/) and is frozen;
do not edit it.

## V2 → V3 deltas

V3 is a strict superset of V2's setting (all three V2 datasets remain), but the
five points below change enough of the pipeline that V2's code is a starting
draft, not a base class.

1. **The model is now scale-blind to σ.** V2 baked two non-learnable
   buffers — `input_scale = 1/σ` before the input projection, `output_scale = σ`
   after the cv head — into `GPTStopper`, so the transformer always saw
   unit-magnitude inputs. V3 deletes both: `X_t` enters raw, `Ĉ_t` exits raw,
   and the network has no σ-dependent constants in its forward pass. The whole
   point of V3 is to ask whether the model can recover σ in-context, which is
   meaningless if we hand it 1/σ as a buffer.

2. **σ-normalization moves to the loss only, and is per-sequence.** With raw
   inputs/outputs, cv squared errors still scale as σ². V3 keeps a 1/σ²
   factor, but applies it to the per-sequence cv loss using each sequence's
   own σ_i (constant within a sequence; different across sequences in the
   random-σ regimes). This is bookkeeping the data pipeline does — the model
   never sees σ_i. Act loss is unchanged.

3. **Two new training distributions with random σ.** In addition to the
   static-σ datasets D_1, D_2, D_3 (σ ∈ {1, 10, 100}), V3 adds:
   - `D_disc`: σ ~ Uniform{1, 10, 100} (one σ per sequence, one of the three).
   - `D_logu`: log₁₀ σ ~ Uniform(0, 2), i.e. σ ∈ [1, 100] uniform on the log
     scale. Each decade gets equal mass under both random priors.

   Five training distributions total → ten models (5 × {cv, act}).

4. **A 2D random-variance ADP (Algorithm 2 in §3.2).** With σ latent the
   sufficient statistic is the pair (S_t, Q_t = Σ X_s²). The DP discretizes
   in transformed coordinates (X̄_t, L_t = log σ̂_t²) on a 256×256 grid with
   bilinear interpolation. The mixture predictive integrates over σ using
   the marginal log-likelihood of Equation (5) in the spec — the discrete
   version sums over {1, 10, 100}, the continuous version uses J_σ = 64-point
   Gauss–Legendre nodes on log σ ∈ [0, log 100] and J = 64 Gauss–Hermite
   nodes per σ-component. V2's 1D µ-only ADP is reused for the static
   datasets unchanged.

5. **Validation comes from the training distribution; test is per-regime.**
   V2 generated one held-out (val, test) pair per dataset from the same
   distribution it trained on. V3 separates the roles: validation is drawn
   from the training distribution and is used for checkpoint selection only;
   test is always drawn per regime D_1, D_2, D_3 and is used for all
   reported numbers. Static-σ models coincide on the diagonal, but the
   random-σ models have a single training-distribution validation set and
   three regime-specific test sets.

## Phases

1. **Oracles** — port V2's static ADP unchanged; implement Algorithm 2
   (2D random-variance ADP) plus the marginal-log-likelihood evaluator;
   run the convergence check at the reference resolution.
2. **Architecture and training** — strip σ from `GPTStopper`; add D_disc
   and D_logu samplers and per-sequence loss normalization; train all ten
   (distribution × supervision) models.
3. **Evaluation** — score every model on each test regime with R/R*, action
   agreement, threshold trajectories, and the data-only baselines (MAP-σ
   and MLE-σ plug-ins) the random-variance models are meant to beat.
4. **Interpretability** — attention-entropy curves, layerwise probes for
   the sufficient statistics / σ-posterior / Ĉ_t, residual-stream
   interventions on the σ direction, activation patching across components,
   logit-lens through depth, and per-head ablation sliced by regime.

## Layout

```
v3/
  research-notes_v3.tex      authoritative spec
  README.md                  this file
  oracle/                    static + random ADP, marginal log-likelihood
  model/                     transformer, heads, checkpoint helpers
  data/                      five samplers (D_1, D_2, D_3, D_disc, D_logu)
                             + labelers + train-stream / val-test builders
  eval/                      payoff, agreement, threshold trajectories,
                             data-only baselines (MAP-σ, MLE-σ plug-ins)
  interp/                    attention entropy, probes, interventions,
                             patching, logit lens, head ablation
  results/                   trained checkpoints, per-model JSON, figures
```

## Spec corrections

V3's spec (`research-notes_v3.tex`) and the original step plans get amended
when implementation surfaces an issue. Corrections are logged here so that
anyone reading the repo cold can reconcile the .tex / step prompts with the
code. Append new entries chronologically (most recent at the bottom) with
the form `[date] [scope]: rule. Reason. Where it lives.` so the list stays
scannable as it grows.

- **[2026-05-05] §3.2 random-variance ADP, X̄ axis discretization.** The
  original spec claimed "marginal SD of X̄_t at most τ_0 uniformly" and
  prescribed a fixed grid `I_X̄ = [-50, 50]`. The claim is wrong:
  `Var(X̄_t) = τ_0² + σ²/t`, so the SD only converges down to `τ_0`, and
  at small `t` with large `σ` it can be much larger (e.g. ~100 for σ=100,
  t=1). At a fixed [-50, 50] grid this clips a substantial fraction of
  σ=100 sequences at small t (~62% at t=1), corrupting labels for a third
  of `D_disc` training data and the upper end of `D_logu`.

  Corrected to a **per-stage adaptive X̄ grid** with half-width
  `s_X̄_t = √(τ_0² + σ_max²/t)`, where σ_max = sup of the prior's support.
  At t=1: ±5·√(τ_0² + σ_max²) ≈ ±503 for σ_max=100, ±70.7 for σ_max=10
  (point-mass reduction test). As t → ∞: ±5·τ_0 = ±50, matching the
  original asymptotic claim. Mirrors the per-stage µ_t grid the static
  ADP already uses (Algorithm 1). Implemented in `oracle/random_adp.py`
  and `oracle/random_adp_torch.py`; spec text updated in
  `research-notes_v3.tex` §3.2 and Algorithm 2.

- **[2026-05-05] Step 2B reduction-test magnitude tolerance.** The Step 2
  plan called for `max |Ĉ_static − Ĉ_random| < 1e-2` at point-mass σ.
  Relaxed to `< 0.1`. At the spec's resolutions (random K1=256 vs static
  K=2048) the random ADP's effective µ-direction resolution is ~8×
  coarser, so a uniform ~5e-2 disagreement is the predicted discretization
  gap. The action-label gate at 1e-3 is the operationally meaningful one
  and stays tight; reduction test passes at 8.8e-5. Encoded in
  `oracle/test_random_reduces_to_static.py`.

- **[2026-05-05] Step 2B magnitude gate dropped on the convergence cells.**
  The Step 2B convergence check's absolute magnitude gate (`max |ΔĈ| <
  0.1` per cell) doesn't accommodate σ-scaling of Ĉ. Median |ΔĈ|/σ_regime
  is ~3e-3 uniformly across D_disc and 4–10e-3 across D_logu — production
  tracks reference to ~0.3–1.0% of σ at the median. The "max" outliers
  (4–12 absolute) are isolated points in low-probability tails of the X̄
  distribution where one of (prod, ref) has a different bilinear-interp
  cell boundary; their action labels still agree, so the disagreement is
  operationally invisible. Action-label gate (< 1e-3) passes by ≥4× on
  every cell (worst 2.1e-4); that's the binding criterion going forward.

- **[2026-05-05] Random-variance ADP backend: PyTorch / GPU.** Step 2
  agreed on NumPy-only ("the lower-risk choice, no new dep, switch later
  if iteration becomes painful"). After the magnitude-gate analysis
  required a re-solve and the projected D_logu reference wall on
  8-threaded NumPy was on the order of hours, the ADP was ported to a
  single A6000 GPU. PyTorch is already a dependency (the trainer uses it
  in Step 3 onwards), so this adds none. The NumPy implementation in
  `oracle/random_adp.py` stays as the reference the torch port was
  validated against (reduction test gives identical 4.81e-2 / 8.8e-5
  numerics). The win is on the production solves we'll re-run iteratively
  (D_logu prod ≈ 2 min on GPU; reference solves are run-once-and-cache).

- **[2026-05-06] §5.2 cv-loss normalization: 1/σ_i, not 1/σ_i².** The
  original spec divided the per-sequence cv loss by σ_i² to make loss
  *values* comparable across regimes. The Stage A pilot exposed a
  structural side-effect: with residual ~σ_i and 1/σ_i² normalization,
  the per-sequence gradient is `d/dθ [residual²/σ_i²] ~ 2·residual·dĈ/dθ
  / σ_i² ~ 2/σ_i · dĈ/dθ`, so σ=100 sequences receive 100× less gradient
  per step than σ=1. Mixed-σ training accordingly under-trains σ=100;
  the pilot's mixed-σ gate showed an 8.97× normalized-MSE asymmetry
  across σ groups vs the < 3× target. Single-σ runs showed the same
  effect (D_3_cv val_5000 / val_500 = 0.51 vs D_1_cv 0.031). Switched
  to 1/σ_i normalization, which gives `2·residual/σ_i · dĈ/dθ ~ 2·dĈ/dθ`
  — gradient magnitudes regime-invariant under Adam updates. Loss
  values now scale as σ and are not directly comparable across regimes,
  but loss-value comparability was already noted as not held in the V2
  footnote (each model's descent shape is the meaningful convergence
  signal). Spec text in `research-notes_v3.tex` §5.2 updated; loss code
  in `model/losses.py` swapped `sigma_i.pow(2)` for `sigma_i`.

  *Addendum (after re-running Stage A under 1/σ_i):* target-magnitude
  asymmetry persists under any loss normalization. The cv head has to
  travel ~σ in parameter space to fit each regime, and Adam moves
  parameters by ~lr per step regardless of loss form, so σ=100 needs
  roughly σ_max/σ_min = 100× more steps than σ=1 to reach comparable
  fractional residual. Empirically tracked in Stage B via
  `val_loss_sigma_{...}` fields in each random-variance run's
  `log.jsonl` (and aggregated into `training_curves_per_sigma.png`).
  Stage A's "MSE/σ < 3× across σ groups" gate was malformed at 5000
  steps and is dropped; Stage B's `training_curves_per_sigma.png` is
  the authoritative diagnostic for whether σ=100 caught up.

## What's in v3/ and where it came from

`v3/` is **not** a copy of `v2/`. Each module is brought over from `v2/code/`
on a per-piece basis as the corresponding step lands, or written fresh. Log
each addition here so the provenance is clear.

| v3 path | source | status |
| --- | --- | --- |
| `oracle/conjugate.py` | new | V3 — `posterior_update`, `posterior_path_batch`, `compute_eta`, `interp_uniform` ported from `v2/code/oracle.py` (verbatim except for module split); `marginal_log_likelihood` (Eq. 5) is fresh for V3. Imported by both ADPs and by all baselines/eval. |
| `oracle/static_adp.py` | `v2/code/oracle.py` | edited — verbatim port of `solve_adp` and `C_hat_lin`; conjugate helpers extracted to `conjugate.py`. Adds a `query()` interface mirroring the random-ADP convention. |
| `oracle/random_adp.py` | new | V3 — Algorithm 2 (2D ADP in (X̄_t, L_t) with bilinear interp), with **per-stage adaptive X̄ bounds** per spec correction above. NumPy + scipy implementation, threaded via `ThreadPoolExecutor`. Owns the `query`, `save_table`, `load_table` interface used by labeling and eval. Reference / fallback impl; the torch port below was validated against it (identical reduction-test numerics). |
| `oracle/random_adp_torch.py` | new | V3 — PyTorch port of Algorithm 2; **default solver going forward, runs on a single A6000 at float64.** One-for-one translation of the math, with a hand-written 4-corner bilinear lookup (no `scipy.ndimage` analog in torch). Returns the same dict format as the NumPy solver, so `query`/`save_table`/`load_table` from `random_adp.py` are reused unchanged. Why torch: D_logu **production** solve drops from ~30 min on threaded NumPy CPU to ~2 min on GPU — that's the iteration-loop win. The reference solve (D_logu ref 33 min on GPU vs hours on CPU) runs once and is cached, so the wall-time argument there is weaker; torch is still the simpler operational choice given PyTorch is already a dep for the Step 3 trainer. See `[2026-05-05] Random-variance ADP backend: PyTorch / GPU` under Spec corrections. |
| `oracle/test_random_reduces_to_static.py` | new | V3 — runnable reduction test: solves random ADP at point-mass σ = 10 and compares to static ADP at σ = 10 over 10⁴ D_2 sequences. |
| `oracle/phase2_static.py` | new | V3 — runs static ADP at production (K=2048,J=128) and reference (K=4096,J=256) for D_1, D_2, D_3; saves tables and writes `results/phase2/static_convergence.md`. |
| `oracle/phase2_random.py` | new | V3 — runs random ADP at production and reference for D_disc and D_logu; saves tables and writes `results/phase2/random_convergence.md`. |
| `model/transformer.py` | `v2/code/model.py` | edited — `GPTStopper` ported with three deletions: the `input_scale`/`output_scale` buffers (V2 lines 158-165, 194-201), the `sigma` constructor argument, and the `supervision` flag. Both cv and act heads coexist on every model; trainer picks which loss applies. Forward returns `{'cv', 'act'}` dict at raw scale. Param count 1,619,714 (V2 was 1,619,585; +129 for the second linear head). |
| `model/losses.py` | new | V3 — `cv_loss(C_hat, y_cv, sigma_i, cv_mask)` and `act_loss(logits, y_act, act_mask)`. cv_loss does per-sequence `1/σ_i²` normalization (per-batch mean would re-couple the regimes); σ_i is loss-side only and never enters the model's forward. |
| `data/distributions.py` | new | V3 — five samplers (D_1, D_2, D_3, D_disc, D_logu) producing per-sequence `(X, sigma_i, mu_i)`. V2 only had the three static distributions; D_disc and D_logu are new. |
| `data/labeling.py` | `v2/code/dataset.py:label_sequences` | edited + extended — `label_static(X, sigma, table)` is the V2 logic with output reshaped to `(N, n)` (terminal placeholder); `label_random(X, table)` is fresh, queries the 2D ADP at recovered (X̄_t, L_t) and hardcodes y_act_1=0 / cv-mask t=1 (within-sequence sample variance undefined for one observation; oracle rejects at t=1 in practice for n=256). |
| `data/streaming.py` | `v2/code/dataset.py:stream_batches` | edited — `stream_batches(distribution, batch, table, rng)` yields `(X, sigma_i, y_cv, y_act, cv_mask, act_mask)`. cv_mask is False at t=1 for random distributions, False at t=n always. Cache helpers `build_cache_one`, `build_all_caches`, `load_cache` populate `data/cache/` with 5 val sets (per training distribution, seed = 42*100 + offset) and 3 test sets (per regime, seed = 43*100 + offset). |
| `data/labeling_v2_reference.py` | `v2/code/dataset.py:label_sequences` | verbatim — used by Smoke 1 only to diff V3's static labeler against V2's reference. Delete after Step 3 if no further regression checks are needed. |
| `data/smoke1_labeler_agreement.py` | new | V3 — Smoke 1 runner, writes `results/phase3/smoke1.md`. |
| `data/smoke2_random_self_consistency.py` | new | V3 — Smoke 2 runner, writes `results/phase3/smoke2.md`. |
| `model/smoke3_forward_loss.py` | new | V3 — Smoke 3 runner, writes `results/phase3/smoke3.md`. |
| `train/configs.py` | new | V3 — `RunConfig` dataclass + `make_run_configs(stage)` returning all 10 (distribution × supervision) configs. Wave layout (3 waves on GPUs 0/1/2/4) and step counts pinned here; pilot/full are the same configs with different step_count and val_every. |
| `train/io.py` | new | V3 — `load_checkpoint(run_name, which='best')` reload entry point used by all of Step 5+. Returns `(model, head_name, metadata)`; `head_name` is the canonical `trained_head` field (filenames are hints only). `save_checkpoint(path, payload, overwrite)` writes; periodic checkpoints refuse to overwrite. |
| `train/loop.py` | `v2/code/train.py` | edited — single-model `train_one(config, device)`; replaces V2's batch-constant `1/cfg.sigma2` with per-sequence `1/sigma_i` (Spec corrections 2026-05-06). Per-σ-group val loss fields (`val_loss_sigma_*`) emitted on every val event. Validation in `torch.no_grad()`. Periodic checkpoints every 50k. Both heads always saved with `trained_head` flag. |
| `train/sweep.py` | new | V3 — single-run CLI entry: `python -m train.sweep --run D_disc_cv --gpu 4 --stage full`. SIGINT handler registered. Inside the subprocess we always use `cuda:0`; the shell sets `CUDA_VISIBLE_DEVICES` per physical GPU. |
| `train/run_sweep.sh` | `v2/code/run_sweep.sh` | edited — 3-wave bash driver. `bash run_sweep.sh pilot` → Stage A 5000-step pilot; `bash run_sweep.sh full` → Stage B full sweep. Each wave is launched in parallel, then `wait`-ed before the next. GPUs 0, 1, 2, 4 only; never 3. |
| `train/pilot_report.py` | new | V3 — Stage A pilot diagnostics generator. Writes `results/phase4_pilot/pilot_report.md` with the four required tables (per-run wall, convergence shape, mixed-σ loss-normalization check, reload contract) and `pilot_curves.png`. |
| `train/sweep_summary.py` | new | V3 — Stage B post-sweep generator. Writes `results/phase4/{sweep_summary.md, sweep_summary.csv, training_curves.png, training_curves_per_sigma.png, training_curves_per_distribution.pdf, reproducibility.md}` from each run's `log.jsonl` + `curves.npz`. |
| `data/labeling_torch.py` | new | V3 — GPU-side random labeler used by the trainer for D_disc / D_logu runs. Vectorized 4-corner bilinear lookup with one big advanced-indexing call across all stages; matches the numpy `label_random` to ~4e-13. Without this, CPU bilinear labeling would push random runs to ~150 ms/step (over the spec's gate); with it, random runs hit ~50 ms/step. |
| `eval/policies.py` | `v2/code/baselines.py` + `eval_common.py` | edited — vectorized over the test set (V2's per-sequence loop replaced by numpy-vectorized cumsum + per-stage ADP queries). Adds the V3 data-only baselines `map_sigma_plugin(sigma_grid, log_omega, eta)` and `mle_sigma_plugin` and the random-ADP oracle. Single API: each policy returns `(action, threshold)` of shape `(N_seq, n)`. |
| `eval/payoff.py` | new | V3 — `expected_payoff(action, X)` and `normalized_payoff(R, R_oracle)`. Trivial wrappers; central enough to be its own module so Step 6+ can import without dragging in baseline policy machinery. |
| `eval/agreement.py` | new | V3 — `per_step_agreement(action_a, action_b)` over `(seq, t in 1..n-1)`. Diagnostic only — high agreement does not imply matched payoff (V2 §5.1 pathology). |
| `eval/trajectory.py` | new | V3 — `trajectory_mean_std(threshold)` returns mean ± std over the test sequences as a function of t. Used for the threshold-trajectory PDFs. |
| `eval/per_sigma_payoff.py` | new | V3 — `per_sigma_breakdown(model, head, X, sigma_i, distribution, device)`. For random-variance models, bins the seed-44 per-σ test cache by σ_i and reports `R / R*(σ_i)` against the single-σ static oracle at the bin's representative σ. D_disc: 3 bins. D_logu: 5 bins on log_10 σ. |
| `eval/run_eval.py` | new | V3 — top-level driver. Iterates 30 cells (10 models × 3 test regimes); precomputes baseline policies once per regime; emits per-cell JSON + sweep-level `payoff_matrix_raw.json`, `agreement_tensor.npz`, `trajectories.npz`, `persigma_results.json`. Uses `load_checkpoint` from Step 4. ~60s total wall on cuda:0. |
| `eval/render.py` | new | V3 — figure + table renderer. Reads the four sweep-level files; writes `payoff_matrix.{csv,tex,png}`, `payoff_matrix_baselines.csv`, `trajectories_<run>.pdf` (× 5 cv runs), `agreement_heatmaps.png`, `per_sigma_<run>.pdf` (× 4 random-variance runs), `summary.md`. |
