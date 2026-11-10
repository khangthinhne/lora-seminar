"""Build figures from generated raw CSV files.

Figures are labelled ``Team experiment (toy, synthetic)``. This module only
reads CSV files; it does not import torch or rerun training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.make_tables import configuration_label, group_runs, read_raw, summarize  # noqa: E402

FIGURE_DIR = Path("results/figures")
SOURCE = "Source: Team experiment (toy, synthetic)."
FLOOR = 1e-12
COLORS = ["#1b6ca8", "#d1495b", "#2a9d8f", "#6a4c93", "#e07a5f", "#3d405b"]


def make_rank_sweep_figures() -> list[Path]:
    """Create the three main exp02 figures."""
    df = read_raw("exp02_rank_sweep")
    if df is None:
        return []
    groups = group_runs(df)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    lora = _lora_curves(groups)
    full = _baseline(groups, trained=True)
    no_ft = _baseline(groups, trained=False)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if lora:
        ranks = [point[0] for point in lora]
        means = [_floor(point[1]) for point in lora]
        stds = [point[2] for point in lora]
        ax.errorbar(
            ranks,
            means,
            yerr=stds,
            marker="o",
            color=COLORS[0],
            capsize=3,
            label="LoRA (validation-selected lr)",
        )
        ax.set_xticks(ranks)
        ax.set_xticklabels([f"r={rank}" for rank in ranks])
        ax.set_xscale("log", base=2)
    if full is not None:
        ax.axhline(_floor(full[0]), linestyle="--", color=COLORS[1], label="Full fine-tuning")
    if no_ft is not None:
        ax.axhline(
            _floor(no_ft[0]), linestyle=":", color=COLORS[5], label="No fine-tuning (delta=0)"
        )
    ax.set_yscale("log")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Test MSE (log scale)")
    ax.set_title("exp02: test MSE versus LoRA rank")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    paths.append(_save(fig, "exp02_test_mse_vs_rank.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    if lora:
        ranks = [point[0] for point in lora]
        means = [_floor(point[1]) for point in lora]
        stds = [point[2] for point in lora]
        ax.errorbar(
            ranks,
            means,
            yerr=stds,
            marker="o",
            color=COLORS[0],
            capsize=3,
            label="LoRA",
        )
        ax.set_xticks(ranks)
        ax.set_xticklabels([f"r={rank}" for rank in ranks])
        ax.set_xscale("log", base=2)
    if full is not None:
        ax.axhline(
            max(full[0] - full[2], FLOOR),
            linestyle="--",
            color=COLORS[1],
            label="Full fine-tuning",
        )
    if no_ft is not None:
        ax.axhline(_floor(no_ft[0]), linestyle=":", color=COLORS[5], label="No fine-tuning")
    ax.set_yscale("log")
    ax.set_xlabel("LoRA rank r")
    ax.set_ylabel("Relative delta error (log scale)")
    ax.set_title("exp02: distance to the true update versus rank")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    paths.append(_save(fig, "exp02_rel_delta_error_vs_rank.png"))

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for point in lora:
        ax.scatter(
            [point[3]],
            [_floor(point[1])],
            color=COLORS[0],
            s=40,
            label="LoRA" if point[0] == lora[0][0] else None,
        )
        ax.annotate(f"r={point[0]}", (point[3], _floor(point[1])), fontsize=7)
    if full is not None:
        ax.scatter(
            [full[2]],
            [_floor(full[0])],
            marker="s",
            color=COLORS[1],
            s=55,
            label="Full fine-tuning",
        )
    if no_ft is not None:
        ax.scatter(
            [no_ft[2]],
            [_floor(no_ft[0])],
            marker="^",
            color=COLORS[5],
            s=55,
            label="No fine-tuning",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test MSE (log scale)")
    ax.set_title("exp02: quality versus trainable parameter count")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    paths.append(_save(fig, "exp02_params_vs_test_mse.png"))
    return paths


def make_fullrank_figure() -> list[Path]:
    """Create the exp03 figure: test MSE versus rank for each gamma."""
    df = read_raw("exp03_fullrank_target")
    if df is None:
        return []
    groups = group_runs(df)
    gammas = sorted({item["target_cfg"]["gamma"] for items in groups.values() for item in items})
    if not gammas:
        return []

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(gammas), figsize=(4.6 * len(gammas), 4.0), sharey=True)
    if len(gammas) == 1:
        axes = [axes]
    for axis, gamma in zip(axes, gammas, strict=True):
        subset = {
            key: items
            for key, items in groups.items()
            if items[0]["target_cfg"].get("gamma") == gamma
        }
        lora = _lora_curves(subset)
        full = _baseline(subset, trained=True)
        no_ft = _baseline(subset, trained=False)
        if lora:
            ranks = [point[0] for point in lora]
            axis.errorbar(
                ranks,
                [_floor(point[1]) for point in lora],
                yerr=[point[2] for point in lora],
                marker="o",
                color=COLORS[0],
                capsize=2,
                label="LoRA",
            )
            axis.set_xticks(ranks)
            axis.set_xticklabels([f"{rank}" for rank in ranks])
            axis.set_xscale("log", base=2)
        if full is not None:
            axis.axhline(_floor(full[0]), linestyle="--", color=COLORS[1], label="Full FT")
        if no_ft is not None:
            axis.axhline(_floor(no_ft[0]), linestyle=":", color=COLORS[5], label="No FT")
        axis.set_yscale("log")
        axis.set_title(f"gamma = {gamma:g}")
        axis.set_xlabel("LoRA rank r")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Test MSE (log scale)")
    axes[0].legend(fontsize=7)
    fig.suptitle("exp03: full-rank decaying spectrum, test MSE versus rank")
    return [_save(fig, "exp03_test_mse_vs_rank_by_gamma.png")]


def make_alpha_figure() -> list[Path]:
    """Create the optional exp05 alpha-scaling figure."""
    df = read_raw("exp05_alpha_scaling")
    if df is None:
        return []
    groups = group_runs(df)
    modes = sorted(
        {item["hparams"].get("lr_selection", "") for items in groups.values() for item in items}
    )
    if not modes:
        return []
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(modes), figsize=(5.0 * len(modes), 4.0), sharey=True)
    if len(modes) == 1:
        axes = [axes]
    for axis, mode in zip(axes, modes, strict=True):
        for color_index, rank in enumerate(sorted(_lora_ranks(groups))):
            points = []
            for items in groups.values():
                first = items[0]
                hparams = first["hparams"]
                if first["method"] != "lora" or hparams.get("r") != rank:
                    continue
                if hparams.get("lr_selection", "") != mode:
                    continue
                mean, std = summarize([item["test_mse"] for item in items])
                points.append((float(hparams["alpha"]), mean, std))
            points.sort()
            if points:
                axis.errorbar(
                    [point[0] for point in points],
                    [_floor(point[1]) for point in points],
                    yerr=[point[2] for point in points],
                    marker="o",
                    capsize=2,
                    color=COLORS[color_index % len(COLORS)],
                    label=f"r={rank}",
                )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xlabel("alpha (log scale)")
        axis.set_title(f"{mode} learning rate")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7)
    axes[0].set_ylabel("Test MSE (log scale)")
    fig.suptitle("exp05: alpha scaling with tuned versus untuned learning rates")
    return [_save(fig, "exp05_alpha_scaling.png")]


def make_qlora_figure() -> list[Path]:
    """Create the optional exp06 simulated-QLoRA figure."""
    df = read_raw("exp06_qlora_sim")
    if df is None:
        return []
    groups = group_runs(df)
    quantizers = sorted({items[0]["quantizer"] for items in groups.values()})
    if len(quantizers) < 2:
        return []
    configs: list[tuple[str, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    for items in groups.values():
        first = items[0]
        if first["method"] != "lora" and first["method"] != "full_ft":
            continue
        label = configuration_label(first["method"], first["hparams"])
        key = (label, first["method"])
        if key not in seen:
            seen.add(key)
            configs.append((label, first["hparams"]))
    configs.sort(key=lambda item: item[0])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    width = 0.8 / len(quantizers)
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    x_positions = np.arange(len(configs))
    for quantizer_index, quantizer in enumerate(quantizers):
        means: list[float] = []
        stds: list[float] = []
        for label, _hparams in configs:
            values = [
                item["test_mse"]
                for items in groups.values()
                if items[0]["quantizer"] == quantizer
                and configuration_label(items[0]["method"], items[0]["hparams"]) == label
                for item in items
            ]
            mean, std = summarize(values)
            means.append(_floor(mean))
            stds.append(std)
        ax.bar(
            x_positions + quantizer_index * width,
            means,
            width=width,
            yerr=stds,
            capsize=2,
            color=COLORS[quantizer_index % len(COLORS)],
            label=quantizer,
        )
    ax.set_xticks(x_positions + width * (len(quantizers) - 1) / 2)
    ax.set_xticklabels([label for label, _ in configs], rotation=20, ha="right", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("Test MSE (log scale)")
    ax.set_title("exp06: full-precision base versus simple 4-bit base (NOT NF4)")
    ax.grid(True, axis="y", which="both", alpha=0.25)
    ax.legend(fontsize=8)
    return [_save(fig, "exp06_qlora_sim.png")]


def make_all_figures() -> list[Path]:
    """Generate every figure whose raw CSV currently exists."""
    generated: list[Path] = []
    generated.extend(make_rank_sweep_figures())
    generated.extend(make_fullrank_figure())
    generated.extend(make_alpha_figure())
    generated.extend(make_qlora_figure())
    return generated


def _lora_curves(
    groups: dict[tuple[str, ...], list[dict[str, Any]]],
) -> list[tuple[int, float, float, int]]:
    points: list[tuple[int, float, float, int]] = []
    for items in groups.values():
        first = items[0]
        if first["method"] != "lora":
            continue
        mean, std = summarize([item["test_mse"] for item in items])
        points.append((int(first["hparams"]["r"]), mean, std, int(first["trainable_params"])))
    points.sort(key=lambda point: point[0])
    return points


def _lora_ranks(groups: dict[tuple[str, ...], list[dict[str, Any]]]) -> set[int]:
    return {
        int(items[0]["hparams"]["r"]) for items in groups.values() if items[0]["method"] == "lora"
    }


def _baseline(
    groups: dict[tuple[str, ...], list[dict[str, Any]]],
    *,
    trained: bool,
) -> tuple[float, float, int] | None:
    for items in groups.values():
        first = items[0]
        if first["method"] != "full_ft":
            continue
        if bool(first["hparams"].get("trained", True)) != trained:
            continue
        mean, std = summarize([item["test_mse"] for item in items])
        return mean, std, int(first["trainable_params"])
    return None


def _floor(value: float) -> float:
    return max(float(value), FLOOR)


def _save(fig: plt.Figure, filename: str) -> Path:
    fig.text(0.5, 0.01, SOURCE, ha="center", fontsize=7, style="italic")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = FIGURE_DIR / filename
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for generated_path in make_all_figures():
        print(f"wrote {generated_path}")
