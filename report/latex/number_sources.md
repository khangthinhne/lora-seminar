# Number sources

Every number in the prose of `results_discussion.tex` is listed here with the
file and row/column it comes from. All experimental numbers are **Team
experiment (toy, synthetic)**. No number is taken from a paper.

## Setup numbers

| Number in prose | Value | Source |
|---|---|---|
| Layer size for training | $d=k=256$ | `configs/base.yaml` (`data.d`, `data.k`); `results/raw/exp02_rank_sweep.csv` columns `d`, `k` |
| Split sizes | 1024 / 256 / 256 | `configs/base.yaml` (`data.n_train`, `n_val`, `n_test`) |
| $W_0$ standard deviation | 0.0625 | `configs/base.yaml` (`data.w0_std`) |
| Observation noise | 0 | `configs/base.yaml` (`data.obs_noise`) |
| True rank | $r^{*}=8$ | `configs/exp02_rank_sweep.yaml`; `results/raw/exp02_rank_sweep.csv` column `target_cfg` |
| Target noise | 0.01 | `configs/exp02_rank_sweep.yaml`; `results/raw/exp02_rank_sweep.csv` column `target_cfg` |
| Scale ratio | 0.5 | `configs/exp02_rank_sweep.yaml`; `results/raw/exp02_rank_sweep.csv` column `target_cfg` |
| Gamma values | 0.5, 1, 2 | `configs/exp03_fullrank_target.yaml`; `results/tables/fullrank_target.csv` column `gamma` |
| Epochs | 50 | `configs/base.yaml` (`train.epochs`); `results/raw/*.csv` column `epochs` |
| Batch size | 128 | `configs/base.yaml` (`train.batch_size`) |
| Weight decay | 0 | `configs/base.yaml` (`train.weight_decay`) |
| Learning-rate grid | 0.001, 0.01, 0.1 | `configs/base.yaml` (`train.lr_grid`) |
| Seeds | 0, 1, 2, 3, 4 (five seeds) | `configs/base.yaml` (`seeds`); `results/raw/*.csv` column `seed` |
| LoRA ranks | 1, 2, 4, 8, 16 | `configs/exp02_rank_sweep.yaml` (`methods` grid) |
| LoRA alpha | 16 | `configs/exp02_rank_sweep.yaml` |
| LoRA init std | 0.01 | `configs/exp02_rank_sweep.yaml` |
| $B=0$ at init | 0 | `AGENT.md` Section 5.1; implementation in `src/lora_lab/methods/lora.py` |
| exp05 ranks | 4, 8 | `configs/exp05_alpha_scaling.yaml` |
| exp05 alphas | 4, 8, 16 | `configs/exp05_alpha_scaling.yaml` |
| exp05 fixed learning rate | 0.03 | `configs/exp05_alpha_scaling.yaml` (`extra.untuned_lr`) |
| exp06 block size / bits | 64 / 4 | `configs/exp06_qlora_sim.yaml` (`quantizers`); `results/raw/exp06_qlora_sim.csv` column `extra` |
| Analytical layer size | $1024\times1024$ | `configs/exp01_param_table.yaml` (`data.d`, `data.k`) |
| Environment: Windows 11 | n/a | `docs/ai_usage_log.md` line 10 |
| Environment: Python 3.10.19 | n/a | `docs/ai_usage_log.md` line 11 |
| Environment: PyTorch 2.2.2+cu121 | n/a | `docs/ai_usage_log.md` line 11 |
| Environment: Intel Core i7-13700H, 14 cores, 20 threads | n/a | User-confirmed; detected from `Win32_Processor` on 2026-11-08 (not stored in the repo) |
| Environment: 32 GB RAM | n/a | User-confirmed; detected from `Win32_ComputerSystem` on 2026-11-08 (not stored in the repo) |
| CPU only, no GPU results | n/a | User-confirmed; `docs/ai_usage_log.md` lines 45-47 |
| Git commit | `6f47a47` | `results/raw/*.csv` column `git_commit` |

## Parameter and memory numbers (Table 1, exp01)

Source: `results/tables/param_table.csv` and
`results/tables/memory_estimate.csv`.

| Number in prose | Value | Row |
|---|---|---|
| Full fine-tuning trainable parameters | 1,048,576 | `param_table.csv`, row `Full fine-tuning`, column `trainable_params` |
| LoRA $r=4$ parameters | 8,192 | `param_table.csv`, row `LoRA r=4, alpha=16` |
| LoRA $r=8$ parameters | 16,384 | `param_table.csv`, row `LoRA r=8, alpha=16` |
| LoRA $r=16$ parameters | 32,768 | `param_table.csv`, row `LoRA r=16, alpha=16` |
| Percentages | 0.781, 1.562, 3.125 | `param_table.csv`, column `percent_of_full` (displayed with three decimals) |
| Reduction factors | 128x, 64x, 32x | `param_table.csv`, column `reduction_factor` |
| 256x256 full FT parameters | 65,536 | `results/tables/rank_sweep.csv`, row `Full fine-tuning`, column `trainable_params` |
| 256x256 LoRA $r=8$ parameters | 4,096 | `results/tables/rank_sweep.csv`, row `LoRA r=8, alpha=16` |
| Full FT weights / grads / Adam / total | 4 / 4 / 8 / 16 MiB | `memory_estimate.csv`, row `Full fine-tuning`; columns `weight_bytes`, `gradient_bytes`, `optimizer_bytes`, `total_bytes` (converted to MiB) |
| LoRA $r=4$ total | 4.094 MiB | `memory_estimate.csv`, row `LoRA r=4, alpha=16`, column `total_bytes` |
| LoRA $r=8$ total | 4.188 MiB | `memory_estimate.csv`, row `LoRA r=8, alpha=16` |
| LoRA $r=16$ total | 4.375 MiB | `memory_estimate.csv`, row `LoRA r=16, alpha=16` |
| Full FT grads + Adam | 12 MiB | 4 MiB + 8 MiB from `memory_estimate.csv`, row `Full fine-tuning` |
| LoRA $r=4$ grads + Adam | 0.094 MiB | 0.031 + 0.062 MiB from `memory_estimate.csv`, row `LoRA r=4, alpha=16` |
| 4-bit base storage | 36,864 B | `results/raw/exp06_qlora_sim.csv`, `extra.weight_storage_bytes`, rows with `quantizer=absmax_block4` |
| Float32 base storage | 262,144 B | `results/raw/exp06_qlora_sim.csv`, `extra.weight_storage_bytes`, rows with `quantizer=none` |
| Storage factor | about 7.1x | 262,144 / 36,864 |

## exp02 rank sweep (Table 2, Figure 1)

Source: `results/tables/rank_sweep.csv` (mean and sample std over five seeds).

| Number in prose | Value | Row |
|---|---|---|
| No fine-tuning test MSE | 2.49e-1 | row `No fine-tuning (delta=0)`, column `test_mse_mean` |
| Full fine-tuning test MSE | 6.15e-6 | row `Full fine-tuning` |
| LoRA $r=1$ test MSE | 2.06e-1 | row `LoRA r=1, alpha=16` |
| LoRA $r=2$ test MSE | 1.65e-1 | row `LoRA r=2, alpha=16` |
| LoRA $r=4$ test MSE | 1.01e-1 | row `LoRA r=4, alpha=16` |
| LoRA $r=8$ test MSE | 3.44e-6 | row `LoRA r=8, alpha=16` |
| LoRA $r=16$ test MSE | 3.84e-6 | row `LoRA r=16, alpha=16` |
| LoRA $r=1$ relative delta error | 9.09e-1 | row `LoRA r=1, alpha=16`, column `rel_delta_err_mean` |
| LoRA $r=8$ relative delta error | 3.72e-3 | row `LoRA r=8, alpha=16` |
| Full fine-tuning relative delta error | 4.90e-3 | row `Full fine-tuning` |
| LoRA $r=8$ vs full FT MSE comparison | 3.44e-6 vs 6.15e-6 | rows as above |
| LoRA $r=8$ vs full FT relative error comparison | 3.72e-3 vs 4.90e-3 | rows as above |

## exp03 full-rank target (Figure 2)

Source: `results/tables/fullrank_target.csv`.

| Number in prose | Value | Row |
|---|---|---|
| gamma=0.5 full FT test MSE | 9.65e-7 | `gamma=0.5`, `Full fine-tuning` |
| gamma=0.5 best LoRA test MSE | 1.15e-1 | `gamma=0.5`, `LoRA r=16, alpha=16` |
| gamma=0.5 gap | +1.15e-1 | `report/results_summary.md` D6 table / difference of the two rows |
| gamma=1 full FT test MSE | 9.73e-5 | `gamma=1`, `Full fine-tuning` |
| gamma=1 best LoRA test MSE | 9.38e-3 | `gamma=1`, `LoRA r=16, alpha=16` |
| gamma=1 gap | +9.29e-3 | `report/results_summary.md` D6 table / difference of the two rows |
| gamma=2 full FT test MSE | 4.06e-4 | `gamma=2`, `Full fine-tuning` |
| gamma=2 best LoRA test MSE | 4.18e-5 | `gamma=2`, `LoRA r=16, alpha=16` |
| gamma=2 gap | -3.64e-4 | `report/results_summary.md` D6 table / difference of the two rows |

## exp04 merge check (Table 3)

Source: `results/tables/merge_check.csv`, column `max_abs_err`.

| Number in prose | Value | Row |
|---|---|---|
| Full FT merge error | 5.25e-6 | `Full fine-tuning` |
| LoRA $r=4$ merge error | 4.77e-6 | `LoRA r=4, alpha=16` |
| LoRA $r=8$ merge error | 4.29e-6 | `LoRA r=8, alpha=16` |
| LoRA $r=16$ merge error | 4.53e-6 | `LoRA r=16, alpha=16` |
| Threshold | 1e-5 | `merge_check.csv`, column `tolerance`; `configs/exp04_merge_check.yaml` |

## exp05 alpha scaling (Figure 3)

Source: `results/tables/alpha_scaling.csv`.

| Number in prose | Value | Row |
|---|---|---|
| $r=8$, tuned, alpha=4 | 3.02e-6 | `LoRA r=8, alpha=4 [other] (tuned lr)` |
| $r=8$, tuned, alpha=8 | 3.10e-6 | `LoRA r=8, alpha=8 [fixed_alpha_8+alpha_eq_r] (tuned lr)` |
| $r=8$, tuned, alpha=16 | 3.44e-6 | `LoRA r=8, alpha=16 [alpha_eq_2r] (tuned lr)` |
| $r=8$, untuned, alpha=4 | 6.31e-6 | `LoRA r=8, alpha=4 [other] (untuned lr)` |
| $r=8$, untuned, alpha=8 | 2.31e-5 | `LoRA r=8, alpha=8 [fixed_alpha_8+alpha_eq_r] (untuned lr)` |
| $r=8$, untuned, alpha=16 | 4.14e-4 | `LoRA r=8, alpha=16 [alpha_eq_2r] (untuned lr)` |
| $r=4$, all settings near | 1.0e-1 | rows `LoRA r=4, ...`, column `test_mse_mean` (range 1.01e-1 to 1.15e-1) |

## exp06 simulated 4-bit base and Table 3 comparison

Source: `results/tables/qlora_sim.csv` and
`results/raw/exp06_qlora_sim.csv` (`extra.quant_error`,
`extra.weight_storage_bytes`).

| Number in prose | Value | Source |
|---|---|---|
| Mean quantization error | 9.92e-2 | `exp06_qlora_sim.csv`, `extra.quant_error`, `absmax_block4` rows |
| LoRA $r=8$ on 4-bit base test MSE | 9.32e-3 | `qlora_sim.csv`, row `LoRA r=8, alpha=16`, `quantizer=absmax_block4` |
| LoRA $r=8$ on 4-bit base relative error | 5.26e-2 | same row, column `rel_delta_err_mean` |
| LoRA $r=8$ float32 test MSE | 3.44e-6 | `qlora_sim.csv`, row `LoRA r=8, alpha=16`, `quantizer=none` |
| LoRA $r=8$ float32 relative error | 3.72e-3 | same row |
| Full FT on 4-bit base test MSE | 6.70e-6 | `qlora_sim.csv`, row `Full fine-tuning`, `quantizer=absmax_block4` |
| Full FT on 4-bit base relative error | 1.98e-1 | same row, column `rel_delta_err_mean` |
| Full FT float32 relative error | 4.90e-3 | `qlora_sim.csv`, row `Full fine-tuning`, `quantizer=none` |
| 4-bit storage 36 KiB vs 256 KiB | 36,864 B vs 262,144 B | `exp06_qlora_sim.csv`, `extra.weight_storage_bytes` (printed in Table 3 as KiB) |
| 6.25% of full count at $r=8$ | 4,096 / 65,536 | `rank_sweep.csv`, rows `LoRA r=8` and `Full fine-tuning` |
| 16x fewer at $r=8$ in 256x256 | 65,536 / 4,096 | same rows |
| 64x fewer at $r=8$ in 1024x1024 | 1,048,576 / 16,384 | `param_table.csv`, rows `Full fine-tuning` and `LoRA r=8` |

## Practical trade-offs numbers

| Number in prose | Value | Source |
|---|---|---|
| LoRA training time range | 0.70--0.90 s | `rank_sweep.csv`, column `train_time_s_mean`, rows `LoRA r=4/8/16` |
| Full FT training time | 1.37 s | `rank_sweep.csv`, row `Full fine-tuning`, column `train_time_s_mean` |
| Merge error below | 1e-5 | `merge_check.csv`, column `tolerance` |

## Paper statements

The only paper-level statements are the definitions of LoRA and QLoRA in
Section 4.4, cited as `hu2022lora` and `dettmers2023qlora`. No quantitative
paper result is used. The bib entries contain only title, authors, venue, year,
and URL.
