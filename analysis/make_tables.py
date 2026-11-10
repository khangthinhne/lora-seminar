"""Build Markdown/CSV tables from generated raw CSV files.

This module only reads ``results/raw/*.csv``. It must not import torch or rerun
training. Missing raw files are skipped so that partial pipelines still work.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

RAW_DIR = Path("results/raw")
TABLE_DIR = Path("results/tables")
SOURCE_NOTE = "Source: Team experiment (toy, synthetic)."


def read_raw(experiment: str) -> pd.DataFrame | None:
    """Return the raw CSV for ``experiment`` or ``None`` when it does not exist."""
    path = RAW_DIR / f"{experiment}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def configuration_label(method: str, method_hparams: dict[str, Any]) -> str:
    """Human-readable configuration label used in every table."""
    if method == "full_ft" and method_hparams.get("trained") is False:
        return "No fine-tuning (delta=0)"
    if method == "full_ft":
        return "Full fine-tuning"
    if method == "lora":
        r = method_hparams.get("r")
        alpha = method_hparams.get("alpha")
        label = f"LoRA r={r}, alpha={_format_number(alpha)}"
        if method_hparams.get("alpha_scheme"):
            label += f" [{method_hparams['alpha_scheme']}]"
        if method_hparams.get("lr_selection"):
            label += f" ({method_hparams['lr_selection']} lr)"
        return label
    return f"{method} {json.dumps(method_hparams, sort_keys=True)}"


def make_param_table() -> Path | None:
    """Create D1: trainable parameter table for a 1024x1024 layer."""
    df = read_raw("exp01_param_table")
    if df is None:
        return None

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        hparams = json.loads(row["method_hparams"])
        trainable = int(row["trainable_params"])
        total = int(row["total_params"])
        rows.append(
            {
                "configuration": configuration_label(row["method"], hparams),
                "method": row["method"],
                "r": hparams.get("r", ""),
                "alpha": hparams.get("alpha", ""),
                "trainable_params": trainable,
                "total_params": total,
                "percent_of_full": 100.0 * trainable / total,
                "reduction_factor": total / trainable if trainable else float("nan"),
            }
        )
    rows.sort(key=_param_row_order)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "param_table.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md_rows = [
        [
            item["configuration"],
            _format_int(item["trainable_params"]),
            f"{item['percent_of_full']:.4f}%",
            f"{item['reduction_factor']:.2f}x",
            _format_int(item["total_params"]),
        ]
        for item in rows
    ]
    md_path = TABLE_DIR / "param_table.md"
    _write_markdown(
        md_path,
        title="D1 - Trainable parameter table (1024 x 1024 layer)",
        headers=[
            "Configuration",
            "Trainable parameters",
            "% of full FT",
            "Reduction factor",
            "Full FT parameters",
        ],
        rows=md_rows,
        note=(
            f"{SOURCE_NOTE} Analytical counts, no training. "
            "Trainable parameter count is not GPU memory; see D2."
        ),
    )
    return md_path


def make_memory_table() -> Path | None:
    """Create D2: analytical weights/gradients/Adam-state estimate."""
    df = read_raw("exp01_param_table")
    if df is None:
        return None

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        hparams = json.loads(row["method_hparams"])
        extra = json.loads(row["extra"])
        rows.append(
            {
                "configuration": configuration_label(row["method"], hparams),
                "method": row["method"],
                "r": hparams.get("r", ""),
                "alpha": hparams.get("alpha", ""),
                "weight_bytes": int(extra["memory_weight_bytes"]),
                "gradient_bytes": int(extra["memory_gradient_bytes"]),
                "optimizer_bytes": int(extra["memory_optimizer_bytes"]),
                "total_bytes": int(extra["memory_total_bytes"]),
            }
        )
    rows.sort(key=_param_row_order)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "memory_estimate.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    md_rows = [
        [
            item["configuration"],
            _format_bytes(item["weight_bytes"]),
            _format_bytes(item["gradient_bytes"]),
            _format_bytes(item["optimizer_bytes"]),
            _format_bytes(item["total_bytes"]),
        ]
        for item in rows
    ]
    md_path = TABLE_DIR / "memory_estimate.md"
    _write_markdown(
        md_path,
        title="D2 - Analytical memory estimate for one layer (float32, Adam)",
        headers=[
            "Configuration",
            "Frozen weights",
            "Gradients",
            "Adam state (2 slots)",
            "Total",
        ],
        rows=md_rows,
        note=(
            f"{SOURCE_NOTE} Analytical estimate only, **not measured GPU memory**. "
            "Activations are excluded, which is why parameter count != GPU memory. "
            "Assumes float32 weights/gradients and two float32 Adam moments per "
            "trainable parameter."
        ),
    )
    return md_path


def make_rank_sweep_table() -> Path | None:
    """Create D5: mean +/- std table for the main rank sweep."""
    return _make_sweep_table(
        experiment="exp02_rank_sweep",
        basename="rank_sweep",
        title="D5 - Rank sweep versus full fine-tuning (mean +/- std over seeds)",
        include_gamma=False,
        include_quantizer=False,
    )


def make_fullrank_table() -> Path | None:
    """Create D6: mean +/- std table for the full-rank limitation experiment."""
    return _make_sweep_table(
        experiment="exp03_fullrank_target",
        basename="fullrank_target",
        title="D6 - Full-rank decaying-spectrum target (mean +/- std over seeds)",
        include_gamma=True,
        include_quantizer=False,
    )


def make_alpha_table() -> Path | None:
    """Create the optional exp05 alpha-scaling table."""
    return _make_sweep_table(
        experiment="exp05_alpha_scaling",
        basename="alpha_scaling",
        title="exp05 - Alpha scaling with tuned and untuned learning rates",
        include_gamma=False,
        include_quantizer=False,
    )


def make_qlora_table() -> Path | None:
    """Create the optional exp06 simulated-QLoRA table."""
    return _make_sweep_table(
        experiment="exp06_qlora_sim",
        basename="qlora_sim",
        title="exp06 - Simulated QLoRA (simple absmax 4-bit, NOT NF4)",
        include_gamma=False,
        include_quantizer=True,
    )


def make_merge_table() -> Path | None:
    """Create D7: numerical merge-error table."""
    df = read_raw("exp04_merge_check")
    if df is None:
        return None
    groups = group_runs(df)
    rows: list[dict[str, Any]] = []
    for items in groups.values():
        first = items[0]
        errors = [item["merge_max_abs_err"] for item in items]
        tolerances = [item["merge_tolerance"] for item in items]
        rows.append(
            {
                "configuration": configuration_label(first["method"], first["hparams"]),
                "method": first["method"],
                "hparams": first["hparams"],
                "max_abs_err": max(errors),
                "mean_abs_err": float(np.mean(errors)),
                "tolerance": tolerances[0],
                "all_ok": all(item["merge_ok"] for item in items),
                "n_seeds": len(items),
                "trainable_params": first["trainable_params"],
            }
        )
    rows.sort(key=_sweep_row_order)

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TABLE_DIR / "merge_check.csv"
    pd.DataFrame(
        [{key: value for key, value in row.items() if key not in {"hparams"}} for row in rows]
    ).to_csv(csv_path, index=False)

    md_rows = [
        [
            row["configuration"],
            _format_int(row["trainable_params"]),
            f"{row['max_abs_err']:.3e}",
            f"{row['tolerance']:.1e}",
            "yes" if row["all_ok"] else "NO",
            str(row["n_seeds"]),
        ]
        for row in rows
    ]
    md_path = TABLE_DIR / "merge_check.md"
    _write_markdown(
        md_path,
        title="D7 - Numerical merge check (float32)",
        headers=[
            "Configuration",
            "Trainable parameters",
            "Max abs error",
            "Threshold",
            "All seeds pass",
            "Seeds",
        ],
        rows=md_rows,
        note=(
            f"{SOURCE_NOTE} Compares ``layer(x)`` with ``layer.merge()(x)`` on a "
            "test batch. The experiment raises an error if any seed exceeds the "
            "configured threshold."
        ),
    )
    return md_path


def make_all_tables() -> list[Path]:
    """Generate every table whose raw CSV currently exists."""
    builders = (
        make_param_table,
        make_memory_table,
        make_rank_sweep_table,
        make_fullrank_table,
        make_merge_table,
        make_alpha_table,
        make_qlora_table,
    )
    generated: list[Path] = []
    for builder in builders:
        path = builder()
        if path is not None:
            generated.append(path)
    return generated


def summarize(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, sample std)`` ignoring NaNs; std is 0 for a single value."""
    clean = [float(value) for value in values if value is not None and not math.isnan(value)]
    if not clean:
        return float("nan"), float("nan")
    mean = float(np.mean(clean))
    std = float(np.std(clean, ddof=1)) if len(clean) > 1 else 0.0
    return mean, std


def group_runs(df: pd.DataFrame) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    """Group raw rows by (method, hparams, target cfg, quantizer)."""
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for _, row in df.iterrows():
        hparams = json.loads(row["method_hparams"])
        target_cfg = json.loads(row["target_cfg"])
        extra = json.loads(row["extra"])
        key = (
            row["method"],
            json.dumps(hparams, sort_keys=True),
            json.dumps(target_cfg, sort_keys=True),
            row["quantizer"],
        )
        groups.setdefault(key, []).append(
            {
                "method": row["method"],
                "hparams": hparams,
                "target": row["target"],
                "target_cfg": target_cfg,
                "extra": extra,
                "quantizer": row["quantizer"],
                "seed": int(row["seed"]),
                "trainable_params": int(row["trainable_params"]),
                "total_params": int(row["total_params"]),
                "lr": _to_float(row["lr"]),
                "test_mse": _to_float(row["test_mse"]),
                "rel_delta_err": _to_float(row["rel_delta_err"]),
                "train_time_s": _to_float(row["train_time_s"]),
                "merge_max_abs_err": _to_float(extra.get("merge_max_abs_err")),
                "merge_tolerance": _to_float(extra.get("merge_tolerance")),
                "merge_ok": bool(extra.get("merge_ok", False)),
                "quant_error": _to_float(extra.get("quant_error")),
            }
        )
    return groups


def _make_sweep_table(
    *,
    experiment: str,
    basename: str,
    title: str,
    include_gamma: bool,
    include_quantizer: bool,
) -> Path | None:
    df = read_raw(experiment)
    if df is None:
        return None

    rows: list[dict[str, Any]] = []
    for items in group_runs(df).values():
        first = items[0]
        test_mean, test_std = summarize([item["test_mse"] for item in items])
        rel_mean, rel_std = summarize([item["rel_delta_err"] for item in items])
        time_mean, time_std = summarize([item["train_time_s"] for item in items])
        rows.append(
            {
                "configuration": configuration_label(first["method"], first["hparams"]),
                "method": first["method"],
                "hparams": first["hparams"],
                "gamma": first["target_cfg"].get("gamma", ""),
                "target": first["target"],
                "quantizer": first["quantizer"],
                "n_seeds": len(items),
                "trainable_params": first["trainable_params"],
                "total_params": first["total_params"],
                "test_mse_mean": test_mean,
                "test_mse_std": test_std,
                "rel_delta_err_mean": rel_mean,
                "rel_delta_err_std": rel_std,
                "train_time_s_mean": time_mean,
                "train_time_s_std": time_std,
            }
        )
    rows.sort(key=lambda item: (_to_float(item["gamma"], missing=-1.0), _sweep_row_order(item)))

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    for row in rows:
        csv_item = {key: value for key, value in row.items() if key != "hparams"}
        csv_item["hparams"] = json.dumps(row["hparams"], sort_keys=True)
        csv_rows.append(csv_item)
    pd.DataFrame(csv_rows).to_csv(TABLE_DIR / f"{basename}.csv", index=False)

    headers = ["Configuration"]
    if include_gamma:
        headers.append("Gamma")
    if include_quantizer:
        headers.append("Quantizer")
    headers.extend(
        [
            "Trainable params",
            "Test MSE (mean +/- std)",
            "Rel. delta error (mean +/- std)",
            "Train time s (mean +/- std)",
            "Seeds",
        ]
    )
    md_rows = []
    for row in rows:
        md_row = [row["configuration"]]
        if include_gamma:
            md_row.append(_format_number(row["gamma"]))
        if include_quantizer:
            md_row.append(row["quantizer"])
        md_row.extend(
            [
                _format_int(row["trainable_params"]),
                _format_mean_std(row["test_mse_mean"], row["test_mse_std"]),
                _format_mean_std(row["rel_delta_err_mean"], row["rel_delta_err_std"]),
                _format_mean_std(row["train_time_s_mean"], row["train_time_s_std"], "f"),
                str(row["n_seeds"]),
            ]
        )
        md_rows.append(md_row)

    md_path = TABLE_DIR / f"{basename}.md"
    _write_markdown(
        md_path,
        title=title,
        headers=headers,
        rows=md_rows,
        note=(
            f"{SOURCE_NOTE} Mean +/- sample std over seeds. Learning rates are "
            "selected on the validation split and metrics are reported on the "
            "test split. CPU toy timings are not representative of LLMs."
        ),
    )
    return md_path


def _write_markdown(
    path: Path,
    *,
    title: str,
    headers: list[str],
    rows: list[list[str]],
    note: str,
) -> None:
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(["", f"_{note}_", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _param_row_order(item: dict[str, Any]) -> tuple[int, float, float]:
    method = item["method"]
    order = 0 if method == "full_ft" else 1
    rank = item.get("r")
    rank = float(rank) if isinstance(rank, (int, float)) else -1.0
    alpha = item.get("alpha")
    alpha = float(alpha) if isinstance(alpha, (int, float)) else -1.0
    return (order, rank, alpha)


def _sweep_row_order(item: dict[str, Any]) -> tuple[int, float, float, str, str]:
    method = item["method"]
    hparams = item["hparams"]
    if method == "full_ft" and hparams.get("trained") is False:
        group = 0
    elif method == "full_ft":
        group = 1
    elif method == "lora":
        group = 2
    else:
        group = 3
    rank = hparams.get("r")
    rank = float(rank) if isinstance(rank, (int, float)) else -1.0
    alpha = hparams.get("alpha")
    alpha = float(alpha) if isinstance(alpha, (int, float)) else -1.0
    return (
        group,
        rank,
        alpha,
        str(hparams.get("alpha_scheme", "")),
        str(hparams.get("lr_selection", "")),
    )


def _to_float(value: Any, missing: float = float("nan")) -> float:
    if value is None:
        return missing
    try:
        if isinstance(value, str) and value == "":
            return missing
        return float(value)
    except (TypeError, ValueError):
        return missing


def _format_int(value: int) -> str:
    return f"{int(value):,}"


def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{float(value):g}"
    return str(value)


def _format_mean_std(mean: float, std: float, kind: str = "e") -> str:
    if math.isnan(mean):
        return "n/a"
    if kind == "f":
        return f"{mean:.4f} +/- {std:.4f}"
    return f"{mean:.3e} +/- {std:.3e}"


def _format_bytes(value: int) -> str:
    mib = value / (1024 * 1024)
    return f"{value:,} B ({mib:.3f} MiB)"


if __name__ == "__main__":
    for generated_path in make_all_tables():
        print(f"wrote {generated_path}")
