"""CLI runner: ``python -m lora_lab.runner --config configs/exp02_rank_sweep.yaml``.

The runner knows only the ``EXPERIMENTS`` registry; it never imports concrete
methods, targets, or quantizers.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

import lora_lab.experiments  # noqa: F401  (populates EXPERIMENTS)
import lora_lab.methods  # noqa: F401  (populates METHODS)
import lora_lab.quant  # noqa: F401  (populates QUANTIZERS)
import lora_lab.targets  # noqa: F401  (populates TARGETS)
from lora_lab.config import load_config
from lora_lab.records import write_records
from lora_lab.registry import EXPERIMENTS

LOGGER = logging.getLogger("lora_lab.runner")
REPO_ROOT = Path(__file__).resolve().parents[2]


def git_commit() -> str:
    """Return the short git commit hash, or ``unknown`` outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def utc_timestamp() -> str:
    """Return a second-resolution UTC ISO timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description="Run one registered lora-lab experiment.")
    parser.add_argument("--config", required=True, help="path to an experiment YAML file")
    parser.add_argument("--raw-dir", default="results/raw", help="directory for raw CSV output")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s %(name)s: %(message)s",
    )
    cfg = load_config(args.config)
    if cfg.name not in EXPERIMENTS:
        available = ", ".join(sorted(EXPERIMENTS))
        raise SystemExit(f"unknown experiment {cfg.name!r}; available: {available}")

    LOGGER.info("running %s from %s", cfg.name, cfg.source)
    experiment = EXPERIMENTS[cfg.name]()
    records = experiment.run(cfg)

    commit = git_commit()
    timestamp = utc_timestamp()
    for record in records:
        record.git_commit = commit
        record.timestamp = timestamp

    raw_dir = Path(args.raw_dir)
    csv_path = write_records(raw_dir / f"{cfg.name}.csv", records)
    config_copy = raw_dir / f"{cfg.name}.config.yaml"
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    config_copy.write_text(
        yaml.safe_dump(dict(cfg.resolved), sort_keys=False),
        encoding="utf-8",
    )
    LOGGER.info("wrote %d records to %s", len(records), csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
