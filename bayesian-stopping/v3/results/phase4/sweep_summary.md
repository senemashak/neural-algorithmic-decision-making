# Stage B sweep summary

Generated: 2026-05-06T15:20:47

## Per-run

| run | wall (s) | total steps | final train | final val | best val | best step | best path |
|---|---:|---:|---:|---:|---:|---:|---|
| D_1_cv | 11119 | 200000 | 7.4060e-06 | 1.2037e-05 | 1.2037e-05 | 200000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_1_cv/best.pt` |
| D_1_act | 5754 | 100000 | 3.1640e-04 | 2.1353e-04 | 2.1353e-04 | 100000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_1_act/best.pt` |
| D_2_cv | 11786 | 200000 | 4.6260e-06 | 5.8059e-06 | 5.8059e-06 | 200000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_2_cv/best.pt` |
| D_2_act | 5358 | 100000 | 3.8944e-04 | 2.4102e-04 | 2.4102e-04 | 100000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_2_act/best.pt` |
| D_3_cv | 10705 | 200000 | 1.1075e-03 | 1.2466e-03 | 1.2466e-03 | 200000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_3_cv/best.pt` |
| D_3_act | 5406 | 100000 | 4.2287e-04 | 5.8145e-04 | 5.8145e-04 | 100000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_3_act/best.pt` |
| D_disc_cv | 15144 | 300000 | 8.5087e-04 | 7.6761e-04 | 7.6759e-04 | 297000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_disc_cv/best.pt` |
| D_disc_act | 7468 | 150000 | 9.3722e-04 | 6.9891e-04 | 6.9891e-04 | 150000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_disc_act/best.pt` |
| D_logu_cv | 16814 | 300000 | 5.5318e-04 | 6.6484e-04 | 6.6484e-04 | 300000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_logu_cv/best.pt` |
| D_logu_act | 11236 | 150000 | 1.2666e-03 | 7.4423e-04 | 7.4423e-04 | 150000 | `/home/senemi/neural-algorithmic-decision-making/bayesian-stopping/v3/checkpoints/D_logu_act/best.pt` |

## Final per-σ-group val loss (random-variance runs)

val_loss is the loss-unit value (per-sequence MSE / σ for cv;
per-sequence BCE for act). For D_logu, σ-bins are log_10 σ ∈
[0, 2/3) low, [2/3, 4/3) mid, [4/3, 2] high.

| run | val_loss_sigma_1 | val_loss_sigma_10 | val_loss_sigma_100 | val_loss_sigma_high | val_loss_sigma_low | val_loss_sigma_mid |
|---|---|---|---|---|---|---|
| D_disc_cv | 2.4947e-04 | 5.1233e-04 | 1.5821e-03 | — | — | — |
| D_disc_act | 3.4967e-04 | 5.8054e-04 | 1.1906e-03 | — | — | — |
| D_logu_cv | — | — | — | 8.5851e-04 | 7.6397e-04 | 3.5625e-04 |
| D_logu_act | — | — | — | 1.0200e-03 | 6.1990e-04 | 5.8441e-04 |
