# exp05 - Alpha scaling with tuned and untuned learning rates

| Configuration | Trainable params | Test MSE (mean +/- std) | Rel. delta error (mean +/- std) | Train time s (mean +/- std) | Seeds |
|---|---|---|---|---|---|
| LoRA r=4, alpha=4 [alpha_eq_r] (tuned lr) | 2,048 | 1.019e-01 +/- 9.407e-03 | 6.404e-01 +/- 1.128e-02 | 0.8106 +/- 0.0417 | 5 |
| LoRA r=4, alpha=4 [alpha_eq_r] (untuned lr) | 2,048 | 1.033e-01 +/- 6.861e-03 | 6.476e-01 +/- 5.285e-03 | 0.8295 +/- 0.0839 | 5 |
| LoRA r=4, alpha=8 [fixed_alpha_8+alpha_eq_2r] (tuned lr) | 2,048 | 1.012e-01 +/- 8.063e-03 | 6.396e-01 +/- 8.565e-03 | 0.7957 +/- 0.0990 | 5 |
| LoRA r=4, alpha=8 [fixed_alpha_8+alpha_eq_2r] (untuned lr) | 2,048 | 1.071e-01 +/- 7.259e-03 | 6.590e-01 +/- 6.364e-03 | 0.9190 +/- 0.1503 | 5 |
| LoRA r=4, alpha=16 [other] (tuned lr) | 2,048 | 1.012e-01 +/- 7.065e-03 | 6.410e-01 +/- 5.417e-03 | 0.7981 +/- 0.0931 | 5 |
| LoRA r=4, alpha=16 [other] (untuned lr) | 2,048 | 1.148e-01 +/- 9.392e-03 | 6.824e-01 +/- 6.997e-03 | 0.8038 +/- 0.0528 | 5 |
| LoRA r=8, alpha=4 [other] (tuned lr) | 4,096 | 3.021e-06 +/- 1.691e-07 | 3.481e-03 +/- 9.536e-05 | 0.8268 +/- 0.1077 | 5 |
| LoRA r=8, alpha=4 [other] (untuned lr) | 4,096 | 6.308e-06 +/- 1.575e-06 | 5.001e-03 +/- 5.978e-04 | 0.7826 +/- 0.0564 | 5 |
| LoRA r=8, alpha=8 [fixed_alpha_8+alpha_eq_r] (tuned lr) | 4,096 | 3.098e-06 +/- 1.713e-07 | 3.524e-03 +/- 9.698e-05 | 0.7734 +/- 0.0747 | 5 |
| LoRA r=8, alpha=8 [fixed_alpha_8+alpha_eq_r] (untuned lr) | 4,096 | 2.312e-05 +/- 1.188e-05 | 9.449e-03 +/- 2.541e-03 | 0.7890 +/- 0.0582 | 5 |
| LoRA r=8, alpha=16 [alpha_eq_2r] (tuned lr) | 4,096 | 3.439e-06 +/- 2.146e-07 | 3.716e-03 +/- 1.146e-04 | 0.8410 +/- 0.0755 | 5 |
| LoRA r=8, alpha=16 [alpha_eq_2r] (untuned lr) | 4,096 | 4.141e-04 +/- 3.126e-04 | 3.900e-02 +/- 1.454e-02 | 0.9076 +/- 0.0632 | 5 |

_Source: Team experiment (toy, synthetic). Mean +/- sample std over seeds. Learning rates are selected on the validation split and metrics are reported on the test split. CPU toy timings are not representative of LLMs._
