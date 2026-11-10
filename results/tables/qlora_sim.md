# exp06 - Simulated QLoRA (simple absmax 4-bit, NOT NF4)

| Configuration | Quantizer | Trainable params | Test MSE (mean +/- std) | Rel. delta error (mean +/- std) | Train time s (mean +/- std) | Seeds |
|---|---|---|---|---|---|---|
| No fine-tuning (delta=0) | none | 65,536 | 2.495e-01 +/- 1.349e-02 | 1.000e+00 +/- 0.000e+00 | 0.0000 +/- 0.0000 | 5 |
| No fine-tuning (delta=0) | absmax_block4 | 65,536 | 2.592e-01 +/- 1.421e-02 | 1.000e+00 +/- 0.000e+00 | 0.0000 +/- 0.0000 | 5 |
| Full fine-tuning | none | 65,536 | 6.154e-06 +/- 2.689e-06 | 4.900e-03 +/- 1.061e-03 | 1.5412 +/- 0.7442 | 5 |
| Full fine-tuning | absmax_block4 | 65,536 | 6.703e-06 +/- 2.888e-06 | 1.985e-01 +/- 1.024e-03 | 1.0128 +/- 0.0814 | 5 |
| LoRA r=4, alpha=16 | none | 2,048 | 1.012e-01 +/- 7.065e-03 | 6.410e-01 +/- 5.417e-03 | 0.9440 +/- 0.1275 | 5 |
| LoRA r=4, alpha=16 | absmax_block4 | 2,048 | 1.107e-01 +/- 7.517e-03 | 6.427e-01 +/- 7.108e-03 | 0.7459 +/- 0.0676 | 5 |
| LoRA r=8, alpha=16 | none | 4,096 | 3.439e-06 +/- 2.146e-07 | 3.716e-03 +/- 1.146e-04 | 0.9655 +/- 0.2592 | 5 |
| LoRA r=8, alpha=16 | absmax_block4 | 4,096 | 9.321e-03 +/- 1.343e-04 | 5.263e-02 +/- 7.161e-04 | 0.7172 +/- 0.0522 | 5 |

_Source: Team experiment (toy, synthetic). Mean +/- sample std over seeds. Learning rates are selected on the validation split and metrics are reported on the test split. CPU toy timings are not representative of LLMs._
