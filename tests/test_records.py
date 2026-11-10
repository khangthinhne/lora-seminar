from __future__ import annotations

from lora_lab.records import SCHEMA, RunRecord, read_records, write_records


def _record(seed: int) -> RunRecord:
    return RunRecord(
        experiment="unit_test",
        method="lora",
        method_hparams={"r": 4, "alpha": 16, "flag": True},
        target="low_rank",
        target_cfg={"true_rank": 8, "nested": {"scale": 0.5}},
        quantizer="none",
        d=16,
        k=8,
        seed=seed,
        lr=0.03,
        epochs=3,
        trainable_params=96,
        total_params=128,
        trainable_ratio=0.75,
        train_mse=0.1,
        val_mse=0.2,
        test_mse=0.3,
        rel_delta_err=0.4,
        train_time_s=0.5,
        extra={"quant_error": 0.0, "merge_ok": True},
        git_commit="abc1234",
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_write_read_round_trip(tmp_path) -> None:
    records = [_record(0), _record(1)]
    path = write_records(tmp_path / "runs.csv", records)
    loaded = read_records(path)
    assert len(loaded) == 2
    assert [record.to_row() for record in loaded] == [record.to_row() for record in records]


def test_schema_header_is_fixed(tmp_path) -> None:
    path = write_records(tmp_path / "runs.csv", [_record(0)])
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert tuple(header.split(",")) == SCHEMA


def test_optional_values_round_trip(tmp_path) -> None:
    record = _record(0)
    record.lr = None
    record.train_mse = None
    record.test_mse = None
    record.rel_delta_err = None
    record.train_time_s = None
    path = write_records(tmp_path / "runs.csv", [record])
    loaded = read_records(path)[0]
    assert loaded.lr is None
    assert loaded.train_mse is None
    assert loaded.test_mse is None
    assert loaded.rel_delta_err is None
    assert loaded.train_time_s is None
