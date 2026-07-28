"""Tests for parsed PSX record validation."""

from data_pipeline.src.validator import validate_records


def _record(**overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "symbol": "OGDC",
        "date": "2026-07-27",
        "ldcp": 220.5,
        "open": 221.0,
        "high": 225.25,
        "low": 219.75,
        "close": 224.0,
        "change": 3.5,
        "change_percent": 1.59,
        "volume": 1234567,
    }
    record.update(overrides)
    return record


def test_accepts_a_valid_record() -> None:
    record = _record()

    valid, rejected = validate_records([record])

    assert valid == [record]
    assert rejected == []


def test_rejects_empty_symbol_inverted_range_and_negative_volume() -> None:
    valid, rejected = validate_records(
        [_record(symbol=" ", high=10.0, low=11.0, volume=-1)]
    )

    assert valid == []
    assert len(rejected) == 1
    assert "symbol must be non-empty" in rejected[0]["reason"]
    assert "high must be greater than or equal to low" in rejected[0]["reason"]
    assert "volume must be non-negative" in rejected[0]["reason"]


def test_rejects_a_missing_required_field() -> None:
    record = _record()
    del record["close"]

    valid, rejected = validate_records([record])

    assert valid == []
    assert rejected[0]["reason"] == "missing required field: close"
