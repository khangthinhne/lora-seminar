"""Generate LaTeX tables for the report from generated CSV files.

This module only reads ``results/tables/*.csv`` and ``results/raw/*.csv``.
It must not import torch or rerun training. Every number in every table is
derived from a CSV row, so the tables can be regenerated with::

    python -m analysis.make_latex_tables
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "results" / "tables"
OUT_DIR = ROOT / "report" / "latex" / "tables"


def read_table(name: str) -> pd.DataFrame:
    """Read one generated table CSV."""
    return pd.read_csv(TABLE_DIR / f"{name}.csv")


def read_raw(name: str) -> pd.DataFrame:
    """Read one raw experiment CSV and decode JSON columns."""
    df = pd.read_csv(RAW_DIR / f"{name}.csv")
    df["hparams"] = df["method_hparams"].apply(json.loads)
    df["extra"] = df["extra"].apply(json.loads)
    return df


def make_params_memory_table() -> Path:
    """Create the parameter-count plus analytical-memory table."""
    params = read_table("param_table")
    memory = read_table("memory_estimate")
    df = params.merge(memory, on=["configuration", "method"], how="inner")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                _label(row["configuration"]),
                _int(row["trainable_params"]),
                f"{row['percent_of_full']:.3f}\\%",
                f"${row['reduction_factor']:g}\\times$",
                f"{row['weight_bytes'] / (1024**2):.3f}",
                f"{row['gradient_bytes'] / (1024**2):.3f}",
                f"{row['optimizer_bytes'] / (1024**2):.3f}",
                f"{row['total_bytes'] / (1024**2):.3f}",
            ]
        )

    return _write_table(
        "params_memory",
        column_spec="lrrrrrrr",
        header=[
            "Configuration",
            "Trainable params",
            "\\% of full",
            "Reduction",
            "Weights (MiB)",
            "Grads (MiB)",
            "Adam (MiB)",
            "Total (MiB)",
        ],
        rows=rows,
        caption=(
            "Trainable parameters and analytical memory estimate for one "
            "$1024\\times1024$ layer (float32, Adam). The memory columns count "
            "frozen weights, gradients, and two Adam moments per trainable "
            "parameter; activations are excluded, so these are not measured GPU "
            "memory. Source: Team experiment (toy, synthetic), exp01."
        ),
        label="tab:params_memory",
        resize=True,
    )


def make_rank_sweep_table() -> Path:
    """Create the exp02 mean +/- std rank-sweep table."""
    df = read_table("rank_sweep")

    rows = []
    for _, row in df.iterrows():
        rows.append(
            [
                _label(row["configuration"]),
                _int(row["trainable_params"]),
                _pm_sci(row["test_mse_mean"], row["test_mse_std"]),
                _pm_sci(row["rel_delta_err_mean"], row["rel_delta_err_std"]),
                _pm_fixed(row["train_time_s_mean"], row["train_time_s_std"]),
            ]
        )

    return _write_table(
        "rank_sweep",
        column_spec="lrrrr",
        header=[
            "Configuration",
            "Trainable params",
            "Test MSE",
            "Rel. $\\Delta$ error",
            "Train time (s)",
        ],
        rows=rows,
        caption=(
            "Rank sweep against full fine-tuning on the rank-8 synthetic target "
            "(mean $\\pm$ sample std over five seeds). Learning rates are "
            "selected on the validation split and metrics are reported on the "
            "test split. CPU toy timings are not representative of LLM training. "
            "Source: Team experiment (toy, synthetic), exp02."
        ),
        label="tab:rank_sweep",
        resize=True,
    )


def make_comparison_table() -> Path:
    """Create the full-FT / LoRA / simulated-4-bit comparison table."""
    rank_sweep = read_table("rank_sweep").set_index("configuration")
    qlora = read_table("qlora_sim")
    merge = read_table("merge_check").set_index("configuration")
    exp02 = read_raw("exp02_rank_sweep")
    exp06 = read_raw("exp06_qlora_sim")

    base_storage = _mean_extra(exp02, "weight_storage_bytes")
    base_storage_4bit = _mean_extra(
        exp06[exp06["quantizer"] == "absmax_block4"], "weight_storage_bytes"
    )
    quant_error_4bit = _mean_extra(exp06[exp06["quantizer"] == "absmax_block4"], "quant_error")

    def rank_row(configuration: str, label: str) -> list[str]:
        row = rank_sweep.loc[configuration]
        merge_err = merge.loc[configuration, "max_abs_err"]
        return [
            label,
            _int(row["trainable_params"]),
            _pm_sci(row["test_mse_mean"], row["test_mse_std"]),
            _pm_sci(row["rel_delta_err_mean"], row["rel_delta_err_std"]),
            f"{base_storage / 1024:.0f}",
            "0",
            _pm_fixed(row["train_time_s_mean"], row["train_time_s_std"]),
            _sci(merge_err),
        ]

    def qlora_row(configuration: str, quantizer: str, label: str) -> list[str]:
        row = qlora[
            (qlora["configuration"] == configuration) & (qlora["quantizer"] == quantizer)
        ].iloc[0]
        return [
            label,
            _int(row["trainable_params"]),
            _pm_sci(row["test_mse_mean"], row["test_mse_std"]),
            _pm_sci(row["rel_delta_err_mean"], row["rel_delta_err_std"]),
            f"{base_storage_4bit / 1024:.0f}",
            _sci(quant_error_4bit),
            _pm_fixed(row["train_time_s_mean"], row["train_time_s_std"]),
            "n/a",
        ]

    rows = [
        rank_row("Full fine-tuning", "Full FT (float32)"),
        rank_row("LoRA r=4, alpha=16", "LoRA $r=4$ (float32)"),
        rank_row("LoRA r=8, alpha=16", "LoRA $r=8$ (float32)"),
        rank_row("LoRA r=16, alpha=16", "LoRA $r=16$ (float32)"),
        qlora_row("Full fine-tuning", "absmax_block4", "Full FT + 4-bit base (sim.)"),
        qlora_row("LoRA r=8, alpha=16", "absmax_block4", "LoRA $r=8$ + 4-bit base (sim.)"),
    ]

    return _write_table(
        "comparison",
        column_spec="lrrrrrrr",
        header=[
            "Configuration",
            "Trainable params",
            "Test MSE",
            "Rel. $\\Delta$ error",
            "Base (KiB)",
            "Quant. error",
            "Train time (s)",
            "Merge err.",
        ],
        rows=rows,
        caption=(
            "Comparison of full fine-tuning, LoRA, and simulated 4-bit-base "
            "variants on the synthetic target (mean $\\pm$ sample std over five "
            "seeds). ``Base'' is the stored size of one $256\\times256$ frozen "
            "base and excludes adapters, gradients, optimizer state, and "
            "activations. Merge error is the maximum over seeds (threshold "
            "$10^{-5}$). The 4-bit rows use simple absmax block quantization, "
            "\\emph{not} NF4. Source: Team experiment (toy, synthetic), "
            "exp02/exp04/exp06."
        ),
        label="tab:comparison",
        resize=True,
    )


def make_all() -> list[Path]:
    """Generate every LaTeX table."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return [
        make_params_memory_table(),
        make_rank_sweep_table(),
        make_comparison_table(),
    ]


def _write_table(
    name: str,
    *,
    column_spec: str,
    header: list[str],
    rows: list[list[str]],
    caption: str,
    label: str,
    resize: bool = False,
) -> Path:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
    ]
    if resize:
        lines.append("\\resizebox{\\linewidth}{!}{%")
    lines.append(f"\\begin{{tabular}}{{{column_spec}}}")
    lines.append("\\toprule")
    lines.append(" & ".join(header) + " \\\\")
    lines.append("\\midrule")
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    if resize:
        lines.append("}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")
    path = OUT_DIR / f"{name}.tex"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _mean_extra(df: pd.DataFrame, key: str) -> float:
    values = [float(extra.get(key, 0.0)) for extra in df["extra"]]
    return float(np.mean(values))


def _label(configuration: str) -> str:
    label = re.sub(r"r=(\d+)", r"$r=\1$", configuration)
    label = re.sub(r"alpha=(\d+)", r"$\\alpha=\1$", label)
    label = label.replace("delta=0", "$\\Delta=0$")
    return label


def _int(value: Any) -> str:
    return f"{int(value):,}".replace(",", "{,}")


def _sci(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    numeric = float(value)
    if numeric == 0.0:
        return "$0$"
    if numeric == 1.0:
        return "$1.00$"
    mantissa, exponent = f"{numeric:.2e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def _pm_sci(mean: float, std: float) -> str:
    return f"{_sci(mean)} $\\pm$ {_sci(std)}"


def _pm_fixed(mean: float, std: float) -> str:
    return f"${float(mean):.2f} \\pm {float(std):.2f}$"


if __name__ == "__main__":
    for generated_path in make_all():
        print(f"wrote {generated_path.relative_to(ROOT)}")
