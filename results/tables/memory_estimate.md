# D2 - Analytical memory estimate for one layer (float32, Adam)

| Configuration | Frozen weights | Gradients | Adam state (2 slots) | Total |
|---|---|---|---|---|
| Full fine-tuning | 4,194,304 B (4.000 MiB) | 4,194,304 B (4.000 MiB) | 8,388,608 B (8.000 MiB) | 16,777,216 B (16.000 MiB) |
| LoRA r=4, alpha=16 | 4,194,304 B (4.000 MiB) | 32,768 B (0.031 MiB) | 65,536 B (0.062 MiB) | 4,292,608 B (4.094 MiB) |
| LoRA r=8, alpha=16 | 4,194,304 B (4.000 MiB) | 65,536 B (0.062 MiB) | 131,072 B (0.125 MiB) | 4,390,912 B (4.188 MiB) |
| LoRA r=16, alpha=16 | 4,194,304 B (4.000 MiB) | 131,072 B (0.125 MiB) | 262,144 B (0.250 MiB) | 4,587,520 B (4.375 MiB) |

_Source: Team experiment (toy, synthetic). Analytical estimate only, **not measured GPU memory**. Activations are excluded, which is why parameter count != GPU memory. Assumes float32 weights/gradients and two float32 Adam moments per trainable parameter._
