"""Offline tests for the bounded sector-informed clustering audit."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from sklearn.metrics import adjusted_rand_score

from data_pipeline.src.clustering_methodology import correlation_to_distance
from data_pipeline.src.clustering_multiview import (
    SECTOR_LAMBDAS,
    MultiViewAuditError,
    build_sector_dissimilarity,
    combine_multiview_dissimilarities,
    evaluate_primary_multiview_grid,
    evaluate_temporal_multiview_grid,
    normalized_return_dissimilarity,
    sector_alignment_diagnostics,
    sector_domination_diagnostics,
)
from data_pipeline.src.clustering_protocol import (
    deterministic_temporal_windows,
    hierarchical_labels,
)


def _correlation_fixture() -> tuple[pd.DataFrame, dict[str, str]]:
    symbols = ["A", "B", "C", "D", "E", "F"]
    sectors = {
        "A": "BANKS",
        "B": "BANKS",
        "C": "ENERGY",
        "D": "ENERGY",
        "E": "TEXTILES",
        "F": "TEXTILES",
    }
    values = np.full((6, 6), 0.10, dtype="float64")
    np.fill_diagonal(values, 1.0)
    for left, right in ((0, 1), (2, 3), (4, 5)):
        values[left, right] = values[right, left] = 0.75
    return pd.DataFrame(values, index=symbols, columns=symbols), sectors


def test_sector_dissimilarity_is_categorical_zero_or_one() -> None:
    _, sectors = _correlation_fixture()

    distance = build_sector_dissimilarity(["D", "A", "C", "B"], sectors)

    assert list(distance.index) == ["A", "B", "C", "D"]
    assert distance.loc["A", "B"] == 0.0
    assert distance.loc["A", "C"] == 1.0
    assert set(np.unique(distance.to_numpy())) == {0.0, 1.0}


def test_sector_names_are_never_interpreted_as_ordinal_numbers() -> None:
    symbols = ["A", "B", "C"]
    first = build_sector_dissimilarity(
        symbols, {"A": "10", "B": "20", "C": "10"}
    )
    second = build_sector_dissimilarity(
        symbols, {"A": "Z", "B": "A", "C": "Z"}
    )

    pd.testing.assert_frame_equal(first, second)


def test_missing_authoritative_sector_fails_closed() -> None:
    with pytest.raises(MultiViewAuditError, match="unavailable"):
        build_sector_dissimilarity(["A", "B"], {"A": "BANKS"})


def test_lambda_set_is_exact_and_combination_is_deterministic() -> None:
    correlation, sectors = _correlation_fixture()
    return_distance = normalized_return_dissimilarity(correlation)
    sector_distance = build_sector_dissimilarity(correlation.index, sectors)

    first = combine_multiview_dissimilarities(
        return_distance, sector_distance, sector_lambda=0.10
    )
    second = combine_multiview_dissimilarities(
        return_distance.copy(), sector_distance.copy(), sector_lambda=0.10
    )

    assert SECTOR_LAMBDAS == (0.0, 0.10, 0.20, 0.30)
    pd.testing.assert_frame_equal(first, second)
    with pytest.raises(ValueError, match="predeclared"):
        combine_multiview_dissimilarities(
            return_distance, sector_distance, sector_lambda=0.15
        )


def test_lambda_zero_reproduces_return_only_complete_linkage() -> None:
    correlation, sectors = _correlation_fixture()
    normalized = normalized_return_dissimilarity(correlation)
    combined = combine_multiview_dissimilarities(
        normalized,
        build_sector_dissimilarity(correlation.index, sectors),
        sector_lambda=0.0,
    )

    direct_labels = hierarchical_labels(
        correlation_to_distance(correlation), linkage="complete", cluster_count=3
    )
    combined_labels = hierarchical_labels(
        combined, linkage="complete", cluster_count=3
    )

    assert adjusted_rand_score(direct_labels, combined_labels) == pytest.approx(1.0)


def test_primary_grid_is_deterministic_and_rejects_lambda_expansion() -> None:
    correlation, sectors = _correlation_fixture()

    first, first_labels = evaluate_primary_multiview_grid(
        correlation, sectors, cluster_counts=(2, 3)
    )
    second, second_labels = evaluate_primary_multiview_grid(
        correlation.copy(), sectors, cluster_counts=(2, 3)
    )

    pd.testing.assert_frame_equal(first, second)
    for key in first_labels:
        np.testing.assert_array_equal(first_labels[key], second_labels[key])
    assert len(first) == len(SECTOR_LAMBDAS) * 2
    with pytest.raises(ValueError, match="exactly"):
        evaluate_primary_multiview_grid(
            correlation,
            sectors,
            sector_lambdas=(0.0, 0.10, 0.15, 0.20, 0.30),
            cluster_counts=(2,),
        )


def test_sector_alignment_and_domination_diagnostics_are_explicit() -> None:
    labels = np.asarray([0, 0, 1, 1])
    alignment = sector_alignment_diagnostics(
        labels, ["BANK", "BANK", "ENERGY", "ENERGY"]
    )
    dominated = sector_domination_diagnostics(
        sector_nmi=alignment["sector_nmi"],
        sector_purity=alignment["sector_purity"],
        return_gap=0.101,
        baseline_return_gap=0.100,
    )
    improved = sector_domination_diagnostics(
        sector_nmi=1.0,
        sector_purity=1.0,
        return_gap=0.120,
        baseline_return_gap=0.100,
    )

    assert alignment["sector_nmi"] == pytest.approx(1.0)
    assert alignment["sector_purity"] == pytest.approx(1.0)
    assert alignment["normalized_within_cluster_sector_entropy"] == pytest.approx(0.0)
    assert dominated["sector_domination_flag"] is True
    assert improved["material_return_gap_improvement"] is True
    assert improved["sector_domination_flag"] is False


def _temporal_market_fixture() -> tuple[
    pd.DataFrame, tuple[str, ...], dict[str, str], pd.DatetimeIndex
]:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2020-01-01", periods=260, freq="B")
    symbols = tuple(f"S{index:02d}" for index in range(22))
    common = rng.normal(0.0002, 0.006, len(dates))
    rows: list[dict[str, object]] = []
    sectors: dict[str, str] = {}
    for index, symbol in enumerate(symbols):
        sectors[symbol] = f"SECTOR_{index % 4}"
        noise = rng.normal(0.0, 0.004 + index * 0.00001, len(dates))
        close = 100.0 * np.exp(np.cumsum(common + noise))
        rows.extend(
            {
                "market_date": market_date,
                "symbol": symbol,
                "open": value,
                "high": value,
                "low": value,
                "close": value,
                "volume": 1000 + index,
            }
            for market_date, value in zip(dates, close, strict=True)
        )
    frame = pd.DataFrame(rows).sort_values(
        ["market_date", "symbol"], kind="mergesort"
    ).reset_index(drop=True)
    return frame, symbols, sectors, dates


def test_temporal_grid_uses_only_declared_train_windows() -> None:
    market, symbols, sectors, dates = _temporal_market_fixture()
    windows = deterministic_temporal_windows(dates)
    baseline, core = evaluate_temporal_multiview_grid(
        market,
        symbols,
        symbols,
        windows,
        sectors,
        cluster_counts=(2, 3),
    )
    future = market.iloc[:22].copy(deep=True)
    future["market_date"] = pd.Timestamp("2030-01-01")
    future["close"] = 1_000_000.0
    with_future, future_core = evaluate_temporal_multiview_grid(
        pd.concat([market, future], ignore_index=True),
        symbols,
        symbols,
        windows,
        sectors,
        cluster_counts=(2, 3),
    )

    pd.testing.assert_frame_equal(baseline, with_future)
    assert core == future_core
    assert (baseline["temporal_common_symbol_count"] == 22).all()


def test_multiview_helpers_do_not_mutate_source_parquet(tmp_path: Path) -> None:
    path = tmp_path / "market.parquet"
    pq.write_table(
        pa.table(
            {
                "market_date": pa.array(
                    [pd.Timestamp("2024-01-01").date()], type=pa.date32()
                ),
                "symbol": ["A"],
                "close": [10.0],
            }
        ),
        path,
    )
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    correlation, sectors = _correlation_fixture()

    evaluate_primary_multiview_grid(
        correlation, sectors, cluster_counts=(2, 3)
    )

    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
