# D1 - Trainable parameter table (1024 x 1024 layer)

| Configuration | Trainable parameters | % of full FT | Reduction factor | Full FT parameters |
|---|---|---|---|---|
| Full fine-tuning | 1,048,576 | 100.0000% | 1.00x | 1,048,576 |
| LoRA r=4, alpha=16 | 8,192 | 0.7812% | 128.00x | 1,048,576 |
| LoRA r=8, alpha=16 | 16,384 | 1.5625% | 64.00x | 1,048,576 |
| LoRA r=16, alpha=16 | 32,768 | 3.1250% | 32.00x | 1,048,576 |

_Source: Team experiment (toy, synthetic). Analytical counts, no training. Trainable parameter count is not GPU memory; see D2._
