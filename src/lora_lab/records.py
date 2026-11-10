"""Long-format run records and CSV serialization.

One row per run. Existing columns must never be renamed; new experiment-specific
metrics go into the JSON ``extra`` column.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA: tuple[str, ...] = (
    "experiment",
    "method",
    "method_hparams",
    "target",
    "target_cfg",
    "quantizer",
    "d",
    "k",
    "seed",
    "lr",
    "epochs",
    "trainable_params",
    "total_params",
    "trainable_ratio",
    "train_mse",
    "val_mse",
    "test_mse",
    "rel_delta_err",
    "train_time_s",
    "extra",
    "git_commit",
    "timestamp",
)


@dataclass
class RunRecord:
    """A single experimental run.

    ``total_params`` is the full dense parameter count ``d * k``; this makes
    ``trainable_ratio`` the fraction of a full fine-tune. ``extra`` holds
    experiment-specific metrics so the schema does not need to change.
    """

    experiment: str
    method: str
    method_hparams: dict[str, Any]
    target: str
    target_cfg: dict[str, Any]
    quantizer: str
    d: int
    k: int
    seed: int
    lr: float | None
    epochs: int
    trainable_params: int
    total_params: int
    trainable_ratio: float
    train_mse: float | None
    val_mse: float | None
    test_mse: float | None
    rel_delta_err: float | None
    train_time_s: float | None
    extra: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    timestamp: str = ""

    def to_row(self) -> dict[str, str]:
        """Convert the record into a CSV-ready string row."""
        return {
            "experiment": self.experiment,
            "method": self.method,
            "method_hparams": _dump_json(self.method_hparams),
            "target": self.target,
            "target_cfg": _dump_json(self.target_cfg),
            "quantizer": self.quantizer,
            "d": str(self.d),
            "k": str(self.k),
            "seed": str(self.seed),
            "lr": _format_optional_float(self.lr),
            "epochs": str(self.epochs),
            "trainable_params": str(self.trainable_params),
            "total_params": str(self.total_params),
            "trainable_ratio": _format_float(self.trainable_ratio),
            "train_mse": _format_optional_float(self.train_mse),
            "val_mse": _format_optional_float(self.val_mse),
            "test_mse": _format_optional_float(self.test_mse),
            "rel_delta_err": _format_optional_float(self.rel_delta_err),
            "train_time_s": _format_optional_float(self.train_time_s),
            "extra": _dump_json(self.extra),
            "git_commit": self.git_commit,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_row(cls, row: dict[str, str]) -> RunRecord:
        """Build a record from a CSV row (inverse of :meth:`to_row`)."""
        return cls(
            experiment=row["experiment"],
            method=row["method"],
            method_hparams=_load_json(row["method_hparams"]),
            target=row["target"],
            target_cfg=_load_json(row["target_cfg"]),
            quantizer=row["quantizer"],
            d=int(row["d"]),
            k=int(row["k"]),
            seed=int(row["seed"]),
            lr=_parse_optional_float(row["lr"]),
            epochs=int(row["epochs"]),
            trainable_params=int(row["trainable_params"]),
            total_params=int(row["total_params"]),
            trainable_ratio=float(row["trainable_ratio"]),
            train_mse=_parse_optional_float(row["train_mse"]),
            val_mse=_parse_optional_float(row["val_mse"]),
            test_mse=_parse_optional_float(row["test_mse"]),
            rel_delta_err=_parse_optional_float(row["rel_delta_err"]),
            train_time_s=_parse_optional_float(row["train_time_s"]),
            extra=_load_json(row["extra"]),
            git_commit=row["git_commit"],
            timestamp=row["timestamp"],
        )


def write_records(path: str | Path, records: list[RunRecord]) -> Path:
    """Write records to ``path`` using the fixed schema (overwrites the file)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMA)
        writer.writeheader()
        for record in records:
            writer.writerow(record.to_row())
    return destination


def read_records(path: str | Path) -> list[RunRecord]:
    """Read records from ``path``; raises if the header is missing columns."""
    source = Path(path)
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = tuple(reader.fieldnames or ())
        missing = [column for column in SCHEMA if column not in fieldnames]
        if missing:
            raise ValueError(f"{source}: missing columns {missing}")
        return [RunRecord.from_row(row) for row in reader]


def _dump_json(value: dict[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _load_json(value: str) -> dict[str, Any]:
    parsed = json.loads(value) if value else {}
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _format_float(value: float) -> str:
    return repr(float(value))


def _format_optional_float(value: float | None) -> str:
    return "" if value is None else _format_float(value)


def _parse_optional_float(value: str) -> float | None:
    return None if value == "" else float(value)
