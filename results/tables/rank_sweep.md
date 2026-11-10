# D5 - Rank sweep versus full fine-tuning (mean +/- std over seeds)

| Configuration | Trainable params | Test MSE (mean +/- std) | Rel. delta error (mean +/- std) | Train time s (mean +/- std) | Seeds |
|---|---|---|---|---|---|
| No fine-tuning (delta=0) | 65,536 | 2.495e-01 +/- 1.349e-02 | 1.000e+00 +/- 0.000e+00 | 0.0000 +/- 0.0000 | 5 |
| Full fine-tuning | 65,536 | 6.154e-06 +/- 2.689e-06 | 4.900e-03 +/- 1.061e-03 | 1.3682 +/- 0.7844 | 5 |
| LoRA r=1, alpha=16 | 512 | 2.058e-01 +/- 1.248e-02 | 9.086e-01 +/- 5.690e-03 | 0.6980 +/- 0.0645 | 5 |
| LoRA r=2, alpha=16 | 1,024 | 1.652e-01 +/- 1.362e-02 | 8.174e-01 +/- 9.573e-03 | 0.6387 +/- 0.0811 | 5 |
| LoRA r=4, alpha=16 | 2,048 | 1.012e-01 +/- 7.065e-03 | 6.410e-01 +/- 5.417e-03 | 0.7021 +/- 0.0353 | 5 |
| LoRA r=8, alpha=16 | 4,096 | 3.439e-06 +/- 2.146e-07 | 3.716e-03 +/- 1.146e-04 | 0.7691 +/- 0.0962 | 5 |
| LoRA r=16, alpha=16 | 8,192 | 3.839e-06 +/- 2.285e-07 | 3.920e-03 +/- 1.164e-04 | 0.8964 +/- 0.0659 | 5 |

_Source: Team experiment (toy, synthetic). Mean +/- sample std over seeds. Learning rates are selected on the validation split and metrics are reported on the test split. CPU toy timings are not representative of LLMs._
