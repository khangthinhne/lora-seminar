# D7 - Numerical merge check (float32)

| Configuration | Trainable parameters | Max abs error | Threshold | All seeds pass | Seeds |
|---|---|---|---|---|---|
| Full fine-tuning | 65,536 | 5.245e-06 | 1.0e-05 | yes | 5 |
| LoRA r=4, alpha=16 | 2,048 | 4.768e-06 | 1.0e-05 | yes | 5 |
| LoRA r=8, alpha=16 | 4,096 | 4.292e-06 | 1.0e-05 | yes | 5 |
| LoRA r=16, alpha=16 | 8,192 | 4.530e-06 | 1.0e-05 | yes | 5 |

_Source: Team experiment (toy, synthetic). Compares ``layer(x)`` with ``layer.merge()(x)`` on a test batch. The experiment raises an error if any seed exceeds the configured threshold._
