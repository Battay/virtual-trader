"""Offline tests for market-mode removal and shrinkage diagnostics."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import data_pipeline.src.clustering_market_mode as market_mode_module
from data_pipeline.src.clustering_market_mode import (
    MarketModeAuditError,
    audit_local_kse100_factor,
    build_equal_weight_market_factor,
    correlation_structure_diagnostics,
    evaluate_ledoit_wolf_shrinkage,
    load_authoritative_current_equity_identity,
    load_train_only_market_values,
    residualize_static_market_factor,
)
from data_pipeline.src.clustering_methodology import (
    build_return_matrix,
    canonical_temporal_date_partitions,
    construct_close_returns,
    training_market_view,
)
from data_pipeline.src.parquet_market_data import load_market_data


MARKET_SCHEMA = pa.schema(
    [
        pa.field("market_date", pa.date32()),
        pa.field("symbol", pa.string()),
        pa.field("ldcp", pa.float64()),
        pa.field("open", pa.float64()),
        pa.field("high", pa.float64()),
        pa.field("low", pa.float64()),
        pa.field("close", pa.float64()),
        pa.field("change", pa.float64()),
        pa.field("change_percent", pa.float64()),
        pa.field("volume", pa.int64()),
    ]
)


def test_local_kse100_alignment_rejects_gap_spanning_returns(tmp_path: Path) -> None:
    dates = pd.date_range("2024-01-01", periods=8)
    index_path = tmp_path / "indices.csv"
    pd.DataFrame(
        {
            "index_code": ["KSE100"] * 5,
            "index_name": ["KSE-100 Index"] * 5,
            "date": [dates[2], dates[3], dates[5], dates[6], dates[7]],
            "value": [100.0, 101.0, 105.0, 106.0, 107.0],
            "source": ["official_psx_timeseries_eod"] * 5,
            "fetched_at": ["2026-08-05T00:00:00Z"] * 5,
        }
    ).to_csv(index_path, index=False)

    evidence = audit_local_kse100_factor(
        index_path, dates, minimum_coverage=0.9
    )

    assert evidence.level_rows_in_train == 5
    assert evidence.return_rows_in_train == 3
    assert evidence.missing_train_return_dates == 4
    assert evidence.first_return_date == "2024-01-04"
    assert list(evidence.returns.index) == [dates[3], dates[6], dates[7]]
    assert not evidence.adequate_for_full_train_static_factor
    assert evidence.source == "official_psx_timeseries_eod"


def test_missing_index_file_fails_closed_without_fabricating_factor(
    tmp_path: Path,
) -> None:
    evidence = audit_local_kse100_factor(
        tmp_path / "missing.csv", pd.date_range("2024-01-01", periods=5)
    )

    assert evidence.reason == "local_index_file_missing"
    assert evidence.returns.empty
    assert evidence.missing_train_return_dates == 4


def test_equal_weight_factor_uses_only_contemporaneous_available_returns() -> None:
    matrix = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03, 0.04],
            "B": [0.03, 0.02, np.nan, 0.08],
            "C": [np.nan, np.nan, 0.09, 0.12],
        },
        index=pd.date_range("2024-01-01", periods=4),
    )
    source = matrix.copy(deep=True)

    result = build_equal_weight_market_factor(
        matrix, ["C", "B", "A"], minimum_constituents=2
    )

    assert result.factor.iloc[0] == pytest.approx(0.02)
    assert np.isnan(result.factor.iloc[1])
    assert result.factor.iloc[2] == pytest.approx(0.06)
    assert result.factor.iloc[3] == pytest.approx(0.08)
    assert result.constituent_counts.tolist() == [2, 1, 2, 3]
    pd.testing.assert_frame_equal(matrix, source)


def test_static_train_regression_recovers_beta_and_preserves_missingness() -> None:
    dates = pd.date_range("2024-01-01", periods=6)
    factor = pd.Series(
        [-0.02, -0.01, 0.0, 0.01, 0.02, 0.03], index=dates
    )
    matrix = pd.DataFrame(
        {
            "A": 0.001 + 1.5 * factor,
            "B": -0.002 + 0.5 * factor,
        },
        index=dates,
    )
    matrix.loc[dates[2], "B"] = np.nan
    source = matrix.copy(deep=True)

    result = residualize_static_market_factor(
        matrix,
        factor,
        ["B", "A"],
        minimum_overlap=4,
        factor_name="fixture_train_factor",
    )
    coefficients = result.coefficients.set_index("symbol")

    assert coefficients.loc["A", "alpha"] == pytest.approx(0.001)
    assert coefficients.loc["A", "beta"] == pytest.approx(1.5)
    assert coefficients.loc["B", "alpha"] == pytest.approx(-0.002)
    assert coefficients.loc["B", "beta"] == pytest.approx(0.5)
    assert result.residuals.loc[dates[2], "B"] is np.nan or pd.isna(
        result.residuals.loc[dates[2], "B"]
    )
    assert result.residuals["A"].abs().max() < 1e-12
    pd.testing.assert_frame_equal(matrix, source)


def test_residual_fit_is_unchanged_when_future_rows_are_kept_out() -> None:
    train_dates = pd.date_range("2024-01-01", periods=6)
    future_dates = pd.date_range("2024-01-07", periods=2)
    train_factor = pd.Series(np.linspace(-0.03, 0.03, 6), index=train_dates)
    train_matrix = pd.DataFrame(
        {"A": 0.002 + 1.2 * train_factor}, index=train_dates
    )
    full_factor = pd.concat(
        [train_factor, pd.Series([5.0, -5.0], index=future_dates)]
    )
    full_matrix = pd.concat(
        [
            train_matrix,
            pd.DataFrame({"A": [100.0, -100.0]}, index=future_dates),
        ]
    )

    train_only = residualize_static_market_factor(
        train_matrix, train_factor, ["A"], minimum_overlap=3
    )
    explicitly_sliced = residualize_static_market_factor(
        full_matrix.loc[train_dates],
        full_factor.loc[train_dates],
        ["A"],
        minimum_overlap=3,
    )

    pd.testing.assert_frame_equal(
        train_only.residuals, explicitly_sliced.residuals
    )
    pd.testing.assert_frame_equal(
        train_only.coefficients, explicitly_sliced.coefficients
    )


def test_insufficient_factor_overlap_does_not_create_residuals() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    matrix = pd.DataFrame({"A": [0.1, 0.2, np.nan, np.nan, np.nan]}, index=dates)
    factor = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)

    result = residualize_static_market_factor(
        matrix, factor, ["A"], minimum_overlap=3
    )

    assert result.residuals["A"].isna().all()
    assert result.coefficients.loc[0, "status"] == "insufficient_factor_overlap"


def test_ledoit_wolf_requires_strict_common_matrix_without_zero_fill() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    incomplete = pd.DataFrame(
        {
            "A": [0.1, np.nan, 0.2, np.nan, 0.3],
            "B": [np.nan, 0.1, np.nan, 0.2, np.nan],
        },
        index=dates,
    )
    source = incomplete.copy(deep=True)

    assessment = evaluate_ledoit_wolf_shrinkage(
        incomplete,
        ["A", "B"],
        representation="fixture",
        minimum_common_dates=2,
    )

    assert not assessment.feasible
    assert assessment.common_date_count == 0
    assert assessment.correlation is None
    pd.testing.assert_frame_equal(incomplete, source)


def test_ledoit_wolf_is_deterministic_when_complete_matrix_is_valid() -> None:
    rng = np.random.default_rng(42)
    complete = pd.DataFrame(
        rng.normal(size=(30, 3)), columns=["A", "B", "C"]
    )

    first = evaluate_ledoit_wolf_shrinkage(
        complete,
        ["C", "A", "B"],
        representation="fixture",
        minimum_common_dates=20,
    )
    second = evaluate_ledoit_wolf_shrinkage(
        complete,
        ["A", "B", "C"],
        representation="fixture",
        minimum_common_dates=20,
    )

    assert first.feasible and second.feasible
    assert first.shrinkage == pytest.approx(second.shrinkage)
    pd.testing.assert_frame_equal(first.correlation, second.correlation)
    assert list(first.correlation.columns) == ["A", "B", "C"]


def test_correlation_structure_diagnostics_detect_market_mode_deterministically(
) -> None:
    correlation = pd.DataFrame(
        [[1.0, 0.8, 0.8], [0.8, 1.0, 0.8], [0.8, 0.8, 1.0]],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    first = correlation_structure_diagnostics("fixture", correlation)
    second = correlation_structure_diagnostics("fixture", correlation.copy())

    assert first == second
    assert first["mean_pairwise_correlation"] == pytest.approx(0.8)
    assert first["largest_eigenvalue"] == pytest.approx(2.6)
    assert first["largest_eigenvalue_share_of_trace"] == pytest.approx(2.6 / 3)


def test_test_partition_never_enters_market_mode_training_returns() -> None:
    dates = pd.date_range("2024-01-01", periods=20)
    partitions = canonical_temporal_date_partitions(dates)
    market = pd.DataFrame(
        {
            "market_date": list(dates) * 2,
            "symbol": ["A"] * 20 + ["B"] * 20,
            "close": np.arange(40, dtype="float64") + 10.0,
        }
    )

    training = training_market_view(market, partitions)
    returns = construct_close_returns(
        training,
        ["A", "B"],
        global_market_dates=partitions.training_dates,
    )

    assert set(returns.returns["market_date"]).isdisjoint(
        partitions.validation_dates
    )
    assert set(returns.returns["market_date"]).isdisjoint(partitions.test_dates)


def test_market_value_loader_predicate_pushes_train_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calendar = pd.date_range("2024-01-01", periods=20)
    training_calendar = calendar[:14]
    training_start = training_calendar[0].date().isoformat()
    training_end = training_calendar[-1].date().isoformat()
    observed: dict[str, object] = {}

    def fake_calendar(
        path: object,
        *,
        start_date: object,
        end_date: object,
    ) -> pd.DatetimeIndex:
        observed.update(
            calendar_path=path,
            calendar_start=start_date,
            calendar_end=end_date,
        )
        return training_calendar

    def fake_market_load(
        path: object,
        *,
        start_date: object,
        end_date: object,
        symbols: object,
    ) -> pd.DataFrame:
        observed.update(
            path=path,
            start_date=start_date,
            end_date=end_date,
            symbols=symbols,
        )
        return pd.DataFrame(
            {
                "market_date": [pd.Timestamp(training_end)],
                "symbol": ["A"],
                "close": [10.0],
            }
        )

    monkeypatch.setattr(market_mode_module, "load_market_calendar", fake_calendar)
    monkeypatch.setattr(market_mode_module, "load_market_data", fake_market_load)

    partitions, market = load_train_only_market_values(
        "fixture.parquet",
        ["B", "A"],
        training_start=training_start,
        training_end=training_end,
    )

    assert partitions.training_dates == tuple(training_calendar)
    assert partitions.validation_dates == ()
    assert partitions.test_dates == ()
    assert observed["calendar_start"] == training_start
    assert observed["calendar_end"] == training_end
    assert observed["start_date"] == training_start
    assert observed["end_date"] == training_end
    assert observed["symbols"] == ("A", "B")
    assert market["market_date"].max() <= pd.Timestamp(training_end)


def test_authoritative_identity_does_not_promote_non_common_instruments(
    tmp_path: Path,
) -> None:
    listing_path = tmp_path / "listings.csv"
    registry_path = tmp_path / "registry.csv"
    source = "https://dps.psx.com.pk/listings-table/main/nc"
    listing_rows = []
    for symbol, security_type, sector in (
        ("AAA", "ordinary_equity", "COMMERCIAL BANKS"),
        ("ETF", "etf", "EXCHANGE TRADED FUNDS"),
        ("MOD", "ordinary_equity", "MODARABAS"),
        ("AAAR", "right", "COMMERCIAL BANKS"),
    ):
        listing_rows.append(
            {
                "symbol": symbol,
                "company_name": f"{symbol} Limited",
                "security_type": security_type,
                "sector": sector,
                "board": "Main",
                "listing_segment": "normal_counter",
                "clearing_type": "NC",
                "listed_in": "ALLSHR",
                "shares": 1000,
                "free_float": 500,
                "officially_listed": True,
                "official_status": "listed",
                "non_compliance_reason": "",
                "source": source,
                "listing_refreshed_at": "2026-08-01T00:00:00+05:00",
                "snapshot_date": "2026-08-01",
            }
        )
    pd.DataFrame(listing_rows).to_csv(listing_path, index=False)
    registry_rows = []
    for row in listing_rows:
        registry_rows.append(
            {
                "symbol": row["symbol"],
                "company_name": row["company_name"],
                "security_type": row["security_type"],
                "sector": row["sector"],
                "officially_listed": True,
                "official_status": "listed",
                "lifecycle_status": "listed_recently_traded",
                "source": source,
                "previous_symbol": "",
                "successor_symbol": "",
                "corporate_action_type": "",
            }
        )
    pd.DataFrame(registry_rows).to_csv(registry_path, index=False)

    identity = load_authoritative_current_equity_identity(
        registry_path=registry_path,
        listing_snapshot_path=listing_path,
    )

    assert identity["symbol"].tolist() == ["AAA"]


def _write_market_parquet(path: Path) -> None:
    rows: list[dict[str, object]] = []
    for offset in range(12):
        for symbol_index, symbol in enumerate(("A", "B", "C")):
            close = 10.0 + symbol_index + offset * (symbol_index + 1) / 10.0
            rows.append(
                {
                    "market_date": date(2024, 1, 1) + timedelta(days=offset),
                    "symbol": symbol,
                    "ldcp": close - 0.1,
                    "open": close - 0.1,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "change": 0.1,
                    "change_percent": 1.0,
                    "volume": 100 + offset,
                }
            )
    pq.write_table(
        pa.Table.from_pandas(
            pd.DataFrame(rows), schema=MARKET_SCHEMA, preserve_index=False
        ),
        path,
    )


def test_market_mode_diagnostics_leave_source_parquet_unchanged(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    _write_market_parquet(path)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    market = load_market_data(path)
    partitions = canonical_temporal_date_partitions(market["market_date"])
    training = training_market_view(market, partitions)
    returns = construct_close_returns(
        training,
        ["A", "B", "C"],
        global_market_dates=partitions.training_dates,
    )
    matrix = build_return_matrix(
        returns.returns,
        value_column="log_return",
        symbols=["A", "B", "C"],
    )
    build_equal_weight_market_factor(matrix, ["A", "B", "C"], minimum_constituents=2)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
