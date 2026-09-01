"""Offline regression tests for the Phase-2 market+macro data contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_pipeline.src.phase2_data_contract import (
    CPI_YOY_SERIES,
    LEAD_AGENT_FEATURE_COLUMNS,
    MACRO_REQUIRED_COLUMNS,
    PHASE2_DATA_CONTRACT_DECISION,
    PHASE2_DATA_CONTRACT_VERSION,
    PHASE2_MACRO_SCHEMA_VERSION,
    POLICY_RATE_SERIES,
    USD_PKR_SERIES,
    Phase2DataContractError,
    align_release_aware_macro,
    build_causal_market_features,
    build_raw_evidence_manifest,
    contract_evidence_hash,
    deterministic_frame_hash,
    fit_train_only_scaler,
    freeze_common_calendar_split,
    load_phase2_data_contract,
    select_train_validation_rows,
    validate_macro_observations,
    validate_phase2_data_contract,
)


def _macro_row(
    series: str,
    reference_date: str,
    value: float,
    release_date: str,
    effective_at: str,
    *,
    vintage: str = "first_release",
    release_at: str | None = None,
) -> dict[str, object]:
    units = {
        POLICY_RATE_SERIES: "percent_per_annum",
        CPI_YOY_SERIES: "percent_year_over_year",
        USD_PKR_SERIES: "pkr_per_usd",
    }
    frequencies = {
        POLICY_RATE_SERIES: "event_driven",
        CPI_YOY_SERIES: "monthly",
        USD_PKR_SERIES: "business_day",
    }
    return {
        "schema_version": PHASE2_MACRO_SCHEMA_VERSION,
        "series": series,
        "reference_date": reference_date,
        "value": value,
        "unit": units[series],
        "native_frequency": frequencies[series],
        "release_date": release_date,
        "release_timestamp": release_at,
        "effective_available_timestamp": effective_at,
        "availability_method": "fixture_official_release",
        "vintage_id": vintage,
        "source": "https://official.example/source",
        "source_version": "fixture_v1",
        "retrieved_at": "2026-09-01T00:00:00Z",
        "provenance_hash": hashlib.sha256(
            f"{series}:{reference_date}:{vintage}".encode()
        ).hexdigest(),
        "point_in_time_safe": True,
    }


def _base_macro() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _macro_row(
                POLICY_RATE_SERIES,
                "2024-01-01",
                10.0,
                "2024-01-01",
                "2024-01-01T00:00:00Z",
                release_at="2024-01-01T00:00:00Z",
            ),
            _macro_row(
                CPI_YOY_SERIES,
                "2023-12-31",
                8.0,
                "2024-01-02",
                "2024-01-02T10:00:00Z",
                release_at="2024-01-02T10:00:00Z",
            ),
            _macro_row(
                USD_PKR_SERIES,
                "2024-01-01",
                280.0,
                "2024-01-02",
                "2024-01-03T00:00:00Z",
                release_at="2024-01-02T12:00:00Z",
            ),
        ]
    )


def _decisions(*timestamps: str) -> pd.DataFrame:
    cutoffs = pd.to_datetime(list(timestamps), utc=True)
    return pd.DataFrame(
        {
            # Session keys are deliberately unique; cutoff timestamps are the
            # condition under test and need not share those synthetic dates.
            "decision_date": pd.date_range("2030-01-01", periods=len(cutoffs)),
            "decision_cutoff": cutoffs,
        }
    )


def _indices(periods: int = 80) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.bdate_range("2024-01-01", periods=periods)
    for position, market_date in enumerate(dates):
        for offset, code in enumerate(("KSE100", "KSE30", "KMI30", "ALLSHR")):
            rows.append(
                {
                    "index_code": code,
                    "date": market_date,
                    "value": 100.0 + offset * 25.0 + position * (1.0 + offset / 10),
                    "volume": 1_000_000 + position * 1_000 + offset,
                    "daily_change": 999_999.0,
                }
            )
    return pd.DataFrame(rows)


def test_release_aware_cpi_join_uses_release_timestamp_not_reference_period() -> None:
    macro = _base_macro()
    result = align_release_aware_macro(
        _decisions("2024-01-02T09:59:59Z", "2024-01-02T10:00:00Z"), macro
    )

    assert pd.isna(result.loc[0, "pbs_cpi_yoy"])
    assert result.loc[1, "pbs_cpi_yoy"] == pytest.approx(8.0)


def test_policy_rate_is_unavailable_until_official_effective_timestamp() -> None:
    macro = _base_macro()
    replacement = _macro_row(
        POLICY_RATE_SERIES,
        "2024-01-05",
        11.0,
        "2024-01-02",
        "2024-01-05T00:00:00Z",
        release_at="2024-01-02T08:00:00Z",
    )
    macro = pd.concat(
        [macro.loc[~macro["series"].eq(POLICY_RATE_SERIES)], pd.DataFrame([replacement])],
        ignore_index=True,
    )

    result = align_release_aware_macro(
        _decisions("2024-01-04T23:59:59Z", "2024-01-05T00:00:00Z"), macro
    )

    assert pd.isna(result.loc[0, "sbp_policy_rate"])
    assert result.loc[1, "sbp_policy_rate"] == pytest.approx(11.0)


def test_fx_reference_value_is_lagged_until_conservative_availability() -> None:
    result = align_release_aware_macro(
        _decisions("2024-01-02T23:59:59Z", "2024-01-03T00:00:00Z"),
        _base_macro(),
    )

    assert pd.isna(result.loc[0, "sbp_usd_pkr_m2m"])
    assert result.loc[1, "sbp_usd_pkr_m2m"] == pytest.approx(280.0)


def test_macro_join_never_interpolates_from_a_future_release() -> None:
    macro = _base_macro()
    macro = pd.concat(
        [
            macro,
            pd.DataFrame(
                [
                    _macro_row(
                        CPI_YOY_SERIES,
                        "2024-01-31",
                        9.5,
                        "2024-02-02",
                        "2024-02-02T10:00:00Z",
                        release_at="2024-02-02T10:00:00Z",
                    )
                ]
            ),
        ],
        ignore_index=True,
    )

    result = align_release_aware_macro(
        _decisions("2024-02-01T00:00:00Z"), macro
    )

    assert result.loc[0, "pbs_cpi_yoy"] == pytest.approx(8.0)


def test_revision_of_older_reference_period_does_not_replace_latest_period() -> None:
    macro = _base_macro()
    additions = [
        _macro_row(
            CPI_YOY_SERIES,
            "2024-01-31",
            9.0,
            "2024-02-02",
            "2024-02-02T10:00:00Z",
            release_at="2024-02-02T10:00:00Z",
        ),
        _macro_row(
            CPI_YOY_SERIES,
            "2023-12-31",
            8.25,
            "2024-03-01",
            "2024-03-01T10:00:00Z",
            release_at="2024-03-01T10:00:00Z",
            vintage="revision_1",
        ),
    ]
    macro = pd.concat([macro, pd.DataFrame(additions)], ignore_index=True)

    result = align_release_aware_macro(
        _decisions("2024-03-02T00:00:00Z"), macro
    )

    assert result.loc[0, "pbs_cpi_yoy"] == pytest.approx(9.0)
    assert result.loc[0, f"{CPI_YOY_SERIES}_reference_date"] == pd.Timestamp(
        "2024-01-31"
    )
    assert result.loc[0, f"{CPI_YOY_SERIES}_available_at"] == pd.Timestamp(
        "2024-02-02T10:00:00Z"
    )


def test_market_features_do_not_use_same_session_close_or_stored_change() -> None:
    source = _indices()
    before = build_causal_market_features(source)
    decision_date = before.loc[60, "decision_date"]
    changed = source.copy(deep=True)
    mask = changed["date"].eq(decision_date) & changed["index_code"].eq("KSE100")
    changed.loc[mask, "value"] *= 10.0
    changed.loc[:, "daily_change"] = -123_456.0
    after = build_causal_market_features(changed)

    np.testing.assert_allclose(
        before.loc[60, list(before.columns[2:])].to_numpy(dtype=float),
        after.loc[60, list(after.columns[2:])].to_numpy(dtype=float),
    )
    assert not np.allclose(
        before.loc[61, "kse100_return_1d_lag1"],
        after.loc[61, "kse100_return_1d_lag1"],
    )


def test_market_return_is_derived_from_levels_across_refresh_boundary() -> None:
    source = _indices(60)
    source.loc[source["date"].eq(source["date"].unique()[30]), "daily_change"] = np.nan

    result = build_causal_market_features(source)

    assert np.isfinite(result.loc[31, "kse100_return_1d_lag1"])


def test_common_calendar_split_is_exact_deterministic_and_nonoverlapping() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    first = freeze_common_calendar_split(dates)
    second = freeze_common_calendar_split(dates)

    assert first == second
    assert first["train"]["row_count"] == 14
    assert first["validation"]["row_count"] == 3
    assert first["test"]["row_count"] == 3
    assert first["train"]["last_date"] < first["validation"]["first_date"]
    assert first["validation"]["last_date"] < first["test"]["first_date"]
    assert first["test_status"] == "SEALED"
    assert first["test_observations_loaded"] is False


def test_train_validation_selector_fails_if_test_observation_is_supplied() -> None:
    dates = pd.bdate_range("2024-01-01", periods=20)
    split = freeze_common_calendar_split(dates)
    rows = pd.DataFrame({"decision_date": dates, "feature": range(20)})

    with pytest.raises(Phase2DataContractError, match="TEST observations"):
        select_train_validation_rows(rows, split)

    non_test = rows.iloc[:17]
    train, validation = select_train_validation_rows(non_test, split)
    assert len(train) == 14
    assert len(validation) == 3
    assert set(train["decision_date"]).isdisjoint(validation["decision_date"])


def test_scaler_is_fit_on_train_only_and_does_not_mutate_sources() -> None:
    feature_columns = ("feature_a", "feature_b")
    train = pd.DataFrame(
        {
            "decision_date": pd.date_range("2024-01-01", periods=3),
            "feature_a": [1.0, 2.0, 3.0],
            "feature_b": [10.0, 20.0, 30.0],
        }
    )
    validation = pd.DataFrame(
        {
            "decision_date": pd.date_range("2024-02-01", periods=2),
            "feature_a": [1_000_000.0, 2_000_000.0],
            "feature_b": [3_000_000.0, 4_000_000.0],
        }
    )
    train_before = train.copy(deep=True)
    validation_before = validation.copy(deep=True)

    result = fit_train_only_scaler(
        train, validation, feature_columns=feature_columns
    )

    np.testing.assert_allclose(result.scaler.mean_, [2.0, 20.0])
    assert result.metadata["fit_partition"] == "TRAIN_ONLY"
    assert result.metadata["validation_fit_rows"] == 0
    assert result.metadata["test_rows_loaded"] is False
    pd.testing.assert_frame_equal(train, train_before)
    pd.testing.assert_frame_equal(validation, validation_before)


def test_missing_required_macro_series_fails_closed() -> None:
    incomplete = _base_macro().loc[
        lambda frame: ~frame["series"].eq(USD_PKR_SERIES)
    ]

    with pytest.raises(Phase2DataContractError, match="Required macro series"):
        validate_macro_observations(incomplete)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    (
        ("effective_available_timestamp", "2023-12-31T00:00:00Z", "availability"),
        ("provenance_hash", "not-a-hash", "provenance"),
        ("point_in_time_safe", False, "Unsafe macro"),
    ),
)
def test_malformed_or_unsafe_macro_source_fails_closed(
    column: str, value: object, message: str
) -> None:
    macro = _base_macro()
    macro.loc[0, column] = value

    with pytest.raises(Phase2DataContractError, match=message):
        validate_macro_observations(macro)


def test_raw_evidence_manifest_is_read_only_and_checksum_verified(tmp_path: Path) -> None:
    source = tmp_path / "official-response.txt"
    source.write_bytes(b"official macro evidence\n")
    before = source.read_bytes()
    before_stat = source.stat()

    manifest = build_raw_evidence_manifest(
        source,
        source_url="https://official.example/macro",
        source_identifier="official_fixture",
        retrieved_at="2026-09-01T00:00:00Z",
        media_type="text/plain",
    )

    assert manifest["sha256"] == hashlib.sha256(before).hexdigest()
    assert source.read_bytes() == before
    assert source.stat().st_mtime_ns == before_stat.st_mtime_ns


def test_deterministic_frame_hash_ignores_row_order_but_detects_values() -> None:
    frame = _base_macro()
    columns = MACRO_REQUIRED_COLUMNS
    first = deterministic_frame_hash(
        frame, columns=columns, sort_by=("series", "reference_date")
    )
    second = deterministic_frame_hash(
        frame.iloc[::-1], columns=columns, sort_by=("series", "reference_date")
    )
    changed = frame.copy()
    changed.loc[0, "value"] += 1

    assert first == second
    assert first != deterministic_frame_hash(
        changed, columns=columns, sort_by=("series", "reference_date")
    )


def test_tracked_contract_is_blocked_without_fabricating_coverage_or_split() -> None:
    payload = load_phase2_data_contract()

    assert payload["artifact_version"] == PHASE2_DATA_CONTRACT_VERSION
    assert payload["decision"] == PHASE2_DATA_CONTRACT_DECISION
    assert payload["canonical_macro_dataset"]["created"] is False
    assert payload["common_calendar_split"]["frozen"] is False
    assert payload["common_calendar_split"]["test_observations_loaded"] is False
    assert payload["normalization"]["fitted"] is False
    assert payload["safety"]["rl_training_performed"] is False
    assert payload["safety"]["test_observations_loaded"] is False


def test_contract_hash_and_safety_fields_fail_closed() -> None:
    payload = deepcopy(load_phase2_data_contract())
    payload["safety"]["test_observations_loaded"] = True
    payload["contract_evidence_hash"] = contract_evidence_hash(payload)

    with pytest.raises(Phase2DataContractError, match="safety"):
        validate_phase2_data_contract(payload)

    payload = deepcopy(load_phase2_data_contract())
    payload["blocking_reason"] = "tampered"
    with pytest.raises(Phase2DataContractError, match="hash verification"):
        validate_phase2_data_contract(payload)


def test_feature_contract_remains_small_interpretable_and_test_free() -> None:
    assert len(LEAD_AGENT_FEATURE_COLUMNS) == 19
    assert len(LEAD_AGENT_FEATURE_COLUMNS) == len(set(LEAD_AGENT_FEATURE_COLUMNS))
    assert not any("test" in name.lower() for name in LEAD_AGENT_FEATURE_COLUMNS)
