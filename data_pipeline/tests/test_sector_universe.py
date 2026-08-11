"""Offline regression tests for research-safe sector universe metadata."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json

import pandas as pd
import pytest

from feature_engineering.schemas import FEATURE_VERSION
from reinforcement_learning.environments.config import (
    DEFAULT_OBSERVATION_FEATURES,
    DYNAMIC_PORTFOLIO_FEATURES,
    ENVIRONMENT_VERSION,
)
from reinforcement_learning.recurrent_data_contract import (
    RL_RECURRENT_PARTITION_SCHEMA_VERSION,
    load_recurrent_contract_metadata,
)
from reinforcement_learning.sector_universe import (
    LEAVE_ONE_SYMBOL_OUT,
    READY_FOR_SECTOR_RESEARCH,
    SECTOR_EPISODE_INDEX_SCHEMA_VERSION,
    SECTOR_NORMALIZATION_POLICY_VERSION,
    SECTOR_TAXONOMY_VERSION,
    SECTOR_UNIVERSE_SCHEMA_VERSION,
    STANDARD_SECTOR_PRETRAINING,
    SectorUniverseError,
    build_current_verified_universe,
    build_sector_manifest,
    build_sector_statistics,
    build_train_episode_index,
    classify_historical_instruments,
    deterministic_universe_hash,
    normalize_sector,
    taxonomy_payload,
)


def _registry_rows() -> pd.DataFrame:
    common = {
        "company_name": "Example Limited",
        "officially_listed": True,
        "official_status": "listed",
        "activity_status": "recently_traded",
        "lifecycle_status": "listed_recently_traded",
        "last_seen_date": "2024-12-30",
        "source": "https://dps.psx.com.pk/listings-table/main/nc",
    }
    return pd.DataFrame(
        [
            {
                **common,
                "symbol": "BANKA",
                "security_type": "ordinary_equity",
                "sector": " COMMERCIAL BANKS ",
                "first_seen_date": "2020-01-02",
            },
            {
                **common,
                "symbol": "FUTURE",
                "security_type": "ordinary_equity",
                "sector": "COMMERCIAL BANKS",
                "first_seen_date": "2025-01-02",
            },
            {
                **common,
                "symbol": "COLD",
                "security_type": "ordinary_equity",
                "sector": "COMMERCIAL BANKS",
                "first_seen_date": "2024-01-02",
            },
            {
                **common,
                "symbol": "ETF",
                "security_type": "etf",
                "sector": "EXCHANGE TRADED FUNDS",
                "first_seen_date": "2020-01-02",
            },
            {
                **common,
                "symbol": "REIT",
                "security_type": "other",
                "sector": "REAL ESTATE INVESTMENT TRUST",
                "first_seen_date": "2020-01-02",
            },
        ]
    )


def _readiness_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "BANKA",
                "usable_feature_rows": 500,
                "first_usable_date": "2020-03-01",
                "last_usable_date": "2024-12-30",
                "readiness_status": "Ready",
            },
            {
                "symbol": "FUTURE",
                "usable_feature_rows": 500,
                "first_usable_date": "2025-03-01",
                "last_usable_date": "2025-12-30",
                "readiness_status": "Ready",
            },
            {
                "symbol": "COLD",
                "usable_feature_rows": 110,
                "first_usable_date": "2024-03-01",
                "last_usable_date": "2024-12-30",
                "readiness_status": "Insufficient History",
            },
            {
                "symbol": "ETF",
                "usable_feature_rows": 500,
                "first_usable_date": "2020-03-01",
                "last_usable_date": "2024-12-30",
                "readiness_status": "Unsupported Security Type",
            },
            {
                "symbol": "REIT",
                "usable_feature_rows": 500,
                "first_usable_date": "2020-03-01",
                "last_usable_date": "2024-12-30",
                "readiness_status": "Unsupported Security Type",
            },
        ]
    )


def _metadata(tmp_path: Path, symbol: str):
    symbol_dir = tmp_path / "data" / "processed" / "splits" / "symbols" / symbol
    recurrent = symbol_dir / "recurrent"
    recurrent.mkdir(parents=True)
    contract_path = recurrent / "recurrent_contract.json"
    contract_path.write_text("{}", encoding="utf-8")
    for filename in (
        "train_rl.csv",
        "rl_observation_scaler.joblib",
        "rl_observation_scaler.json",
    ):
        (symbol_dir / filename).write_bytes(f"{symbol}:{filename}".encode())
    return SimpleNamespace(
        symbol=symbol,
        sector="COMMERCIAL BANKS",
        contract_path=contract_path,
        recurrent_contract_version=RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        feature_version=FEATURE_VERSION,
        environment_version=ENVIRONMENT_VERSION,
        observation_features=DEFAULT_OBSERVATION_FEATURES,
        observation_shape=(
            len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES),
        ),
        normalization_scope="symbol",
        scaler_fit_partition="train",
        train=SimpleNamespace(rows=350, start="2020-03-01", end="2023-12-31"),
    )


def _current_fixture(tmp_path: Path) -> pd.DataFrame:
    values = {
        symbol: _metadata(tmp_path, symbol) for symbol in ("BANKA", "FUTURE")
    }

    def loader(symbol: str, **_: object):
        return values[symbol]

    return build_current_verified_universe(
        _registry_rows(),
        _readiness_rows(),
        cohort_cutoff="2024-12-31",
        listing_snapshot_date="2024-12-31",
        splits_dir=tmp_path / "data" / "processed" / "splits",
        project_root=tmp_path,
        metadata_loader=loader,
    )


def _approved_row(symbol: str, *, rows: int = 1_000) -> dict[str, object]:
    return {
        "symbol": symbol,
        "company_name": f"{symbol} Limited",
        "raw_sector": "COMMERCIAL BANKS",
        "sector_id": "commercial_banks",
        "sector_name": "Commercial Banks",
        "security_type": "ordinary_equity",
        "history_class": "MATURE",
        "first_observed_date": "2019-01-01",
        "first_usable_date": "2019-03-01",
        "sector_verified_current": True,
        "sector_verified_at_cutoff": False,
        "historical_sector_membership_verified": False,
        "historical_sector_membership_unknown": True,
        "recurrent_compatible": True,
        "feature_version": FEATURE_VERSION,
        "recurrent_contract_version": RL_RECURRENT_PARTITION_SCHEMA_VERSION,
        "environment_version": ENVIRONMENT_VERSION,
        "observation_features": json.dumps(list(DEFAULT_OBSERVATION_FEATURES)),
        "observation_shape": json.dumps(
            [len(DEFAULT_OBSERVATION_FEATURES) + len(DYNAMIC_PORTFOLIO_FEATURES)]
        ),
        "execution_semantics": "single_symbol_env_v1_real_ohlcv_next_open_execution",
        "normalization_scope": "symbol",
        "scaler_fit_partition": "train",
        "train_rows": rows,
        "train_start": "2019-03-01",
        "last_train_date": "2023-12-31",
        "rows_available_at_cutoff": rows,
        "eligible_at_cutoff": True,
        "exclusion_reason": "",
        "recurrent_contract_path": f"data/splits/{symbol}/recurrent_contract.json",
        "recurrent_contract_sha256": "a" * 64,
        "train_rl_path": f"data/splits/{symbol}/train_rl.csv",
        "train_rl_sha256": "b" * 64,
        "scaler_path": f"data/splits/{symbol}/rl_observation_scaler.joblib",
        "scaler_sha256": "c" * 64,
        "scaler_metadata_path": f"data/splits/{symbol}/rl_observation_scaler.json",
        "scaler_metadata_sha256": "d" * 64,
    }


def _sector_rows() -> pd.DataFrame:
    rows = [_approved_row(f"BANK{index}", rows=800 + index * 100) for index in range(6)]
    cold = _approved_row("COLD", rows=0)
    cold.update(
        {
            "history_class": "COLD_START",
            "recurrent_compatible": False,
            "eligible_at_cutoff": False,
            "exclusion_reason": "cold_start_not_independent_pretraining",
        }
    )
    rows.append(cold)
    return pd.DataFrame(rows)


def _manifest(rows: pd.DataFrame, **overrides: object) -> dict[str, object]:
    values = {
        "sector_rows": rows,
        "sector_id": "commercial_banks",
        "sector_name": "Commercial Banks",
        "cohort_cutoff": "2026-08-07",
        "listing_snapshot_date": "2026-08-02",
        "source_registry_sha256": "e" * 64,
        "source_listing_sha256": "f" * 64,
        "generated_at": "2026-08-12T00:00:00+00:00",
        "git_commit": "1" * 40,
    }
    values.update(overrides)
    return build_sector_manifest(**values)


def test_sector_taxonomy_is_explicit_and_unknown_is_not_guessed() -> None:
    banks = normalize_sector("  commercial banks  ")
    alias = normalize_sector(
        "Investment Banks / Investment Companies / Securities Companies"
    )
    unknown = normalize_sector("Imaginary Future Sector")
    assert banks.sector_id == "commercial_banks"
    assert alias.sector_id == "investment_banks_companies_securities"
    assert unknown.sector_id == "unknown"
    assert unknown.recognized is False
    payload = taxonomy_payload(generated_at="one")
    repeated = taxonomy_payload(generated_at="two")
    assert payload["taxonomy_version"] == SECTOR_TAXONOMY_VERSION
    assert payload["taxonomy_hash"] == repeated["taxonomy_hash"]


def test_historical_audit_separates_contract_like_from_unknown_without_guessing() -> None:
    symbols = ("ABL-APR", "ABL-CDEC", "ABL-ODL", "ALACR3", "AKBLTFC7", "ADOS")
    registry = pd.DataFrame(
        {
            "symbol": symbols,
            "lifecycle_status": "historical_only",
            "security_type": "unknown",
            "company_name": "",
            "sector": "",
            "first_seen_date": "2020-01-01",
            "last_seen_date": "2020-12-31",
            "trading_days": 10,
        }
    )
    result = classify_historical_instruments(registry)
    assert (result["ordinary_equity_verified"] == False).all()  # noqa: E712
    assert (result["sector_available"] == False).all()  # noqa: E712
    assert result["historical_audit_group"].value_counts().to_dict() == {
        "non_equity_or_contract_like": 5,
        "unknown_requires_investigation": 1,
    }
    assert result.loc[result["symbol"].eq("ADOS"), "recommended_action"].item().startswith(
        "authoritative"
    )


def test_current_universe_enforces_cutoff_and_preserves_security_types(tmp_path: Path) -> None:
    result = _current_fixture(tmp_path).set_index("symbol")
    assert bool(result.loc["BANKA", "eligible_at_cutoff"])
    assert not bool(result.loc["FUTURE", "eligible_at_cutoff"])
    assert result.loc["FUTURE", "exclusion_reason"] == (
        "first_observation_after_cohort_cutoff"
    )
    assert result.loc["COLD", "history_class"] == "COLD_START"
    assert result.loc["ETF", "research_security_category"] == "etf"
    assert result.loc["REIT", "research_security_category"] == "reit"
    assert bool(result.loc["BANKA", "sector_verified_at_cutoff"])
    assert not bool(result.loc["BANKA", "historical_sector_membership_verified"])


def test_current_universe_never_reads_a_test_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = pd.read_csv

    def guarded(path: object, *args: object, **kwargs: object):
        if Path(path).name in {"test.csv", "test_rl.csv"}:
            raise AssertionError("TEST frame was opened")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", guarded)
    result = _current_fixture(tmp_path)
    assert len(result) == 5


def test_manifest_hash_is_deterministic_and_identity_sensitive() -> None:
    rows = _sector_rows()
    first = _manifest(rows, generated_at="first")
    second = _manifest(rows, generated_at="second")
    changed_cutoff = _manifest(rows, cohort_cutoff="2026-08-06")
    changed_rows = _manifest(rows.loc[rows["symbol"].ne("BANK5")])
    assert first["universe_hash"] == second["universe_hash"]
    assert first["universe_hash"] != changed_cutoff["universe_hash"]
    assert first["universe_hash"] != changed_rows["universe_hash"]
    identity = deepcopy(first["deterministic_identity"])
    baseline = deterministic_universe_hash(identity)
    for field, replacement in (
        ("taxonomy_version", "taxonomy_v2"),
        ("feature_version", "feature_v2"),
        ("recurrent_contract_version", "recurrent_v2"),
        ("normalization_policy", "normalization_v2"),
    ):
        modified = deepcopy(identity)
        modified[field] = replacement
        assert deterministic_universe_hash(modified) != baseline


def test_leave_one_symbol_out_excludes_target_from_data_and_scaler() -> None:
    rows = _sector_rows()
    manifest = _manifest(
        rows,
        mode=LEAVE_ONE_SYMBOL_OUT,
        target_symbol="BANK2",
    )
    symbols = manifest["experiment_mode"]["pretraining_constituent_symbols"]
    contributors = manifest["normalization"]["normalization_contributor_symbols"]
    assert "BANK2" not in symbols
    assert "BANK2" not in contributors
    assert manifest["experiment_mode"]["target_excluded_from_pretraining"] is True
    exclusion = {item["symbol"]: item["reason"] for item in manifest["excluded_symbols"]}
    assert exclusion["BANK2"] == "target_excluded_leave_one_symbol_out"


def test_cold_start_target_can_be_declared_but_never_enters_pretraining() -> None:
    manifest = _manifest(
        _sector_rows(),
        mode=LEAVE_ONE_SYMBOL_OUT,
        target_symbol="COLD",
    )
    assert "COLD" not in manifest["experiment_mode"]["pretraining_constituent_symbols"]
    assert "COLD" not in manifest["normalization"]["normalization_contributor_symbols"]


def test_standard_mode_rejects_target_and_duplicate_symbols() -> None:
    rows = _sector_rows()
    with pytest.raises(SectorUniverseError, match="cannot declare a target"):
        _manifest(rows, mode=STANDARD_SECTOR_PRETRAINING, target_symbol="BANK1")
    duplicated = pd.concat([rows, rows.iloc[[0]]], ignore_index=True)
    with pytest.raises(SectorUniverseError, match="duplicate"):
        _manifest(duplicated)


def test_manifest_fails_closed_for_heterogeneous_contracts() -> None:
    rows = _sector_rows()
    rows.loc[rows["symbol"].eq("BANK1"), "observation_features"] = json.dumps(
        list(reversed(DEFAULT_OBSERVATION_FEATURES))
    )
    with pytest.raises(SectorUniverseError, match="heterogeneous"):
        _manifest(rows)


def test_train_episode_index_preserves_symbol_and_portfolio_isolation() -> None:
    manifest = _manifest(_sector_rows())
    index = build_train_episode_index(manifest)
    assert index["artifact_schema_version"] == SECTOR_EPISODE_INDEX_SCHEMA_VERSION
    assert index["partition"] == "train"
    assert index["validation_references_included"] is False
    assert index["test_references_included"] is False
    assert len({item["symbol"] for item in index["episodes"]}) == index["episode_count"]
    assert all(item["episode_start"] is True for item in index["episodes"])
    reset_fields = {
        "environment",
        "cash",
        "holdings",
        "realized_profit_loss",
        "portfolio_peak_and_drawdown",
        "recurrent_hidden_state",
    }
    assert all(
        set(item["reset_before_episode"]) == reset_fields
        and all(item["reset_before_episode"].values())
        for item in index["episodes"]
    )
    assert sum(item["equal_symbol_episode_sampling_share"] for item in index["episodes"]) == pytest.approx(1.0)
    assert sum(item["proportional_row_sampling_share"] for item in index["episodes"]) == pytest.approx(1.0)


def test_manifest_and_episode_paths_are_portable() -> None:
    manifest = _manifest(_sector_rows())
    episode = build_train_episode_index(manifest)
    serialized = json.dumps({"manifest": manifest, "episode": episode})
    assert "/Users/" not in serialized
    assert "\\Users\\" not in serialized
    assert all(
        not Path(item["source_train_rl_path"]).is_absolute()
        for item in episode["episodes"]
    )


def test_statistics_report_diversity_depth_and_history_classes() -> None:
    rows = _sector_rows()
    result = build_sector_statistics(rows).iloc[0]
    assert result["mature_symbols"] == 6
    assert result["cold_start_symbols"] == 1
    assert result["recurrent_compatible_symbols"] == 6
    assert result["approved_constituent_symbols"] == 6
    assert result["total_train_rows"] == sum(800 + index * 100 for index in range(6))
    assert result["research_readiness_status"] == READY_FOR_SECTOR_RESEARCH
    assert result["historical_membership_status"] == "LIMITED_CURRENT_SECTOR_ONLY"


def test_manifest_records_train_only_normalization_and_no_performance() -> None:
    manifest = _manifest(_sector_rows())
    serialized = json.dumps(manifest)
    assert manifest["normalization"]["policy_version"] == SECTOR_NORMALIZATION_POLICY_VERSION
    assert manifest["normalization"]["fit_partition"] == "train"
    assert manifest["cohort"]["validation_performance_used_for_membership"] is False
    assert manifest["cohort"]["test_data_used_for_membership"] is False
    assert "validation_metrics" not in serialized
    assert "test_metrics" not in serialized


def test_current_single_symbol_recurrent_contract_remains_valid() -> None:
    metadata = load_recurrent_contract_metadata("MCB")
    assert metadata.recurrent_contract_version == RL_RECURRENT_PARTITION_SCHEMA_VERSION
    assert metadata.train.sealed is False
    assert metadata.validation.sealed is False
    assert metadata.test.sealed is True


def test_schema_versions_are_explicit() -> None:
    manifest = _manifest(_sector_rows())
    assert manifest["artifact_schema_version"] == SECTOR_UNIVERSE_SCHEMA_VERSION
    assert manifest["taxonomy_version"] == SECTOR_TAXONOMY_VERSION
    assert manifest["experiment_mode"]["mode"] == STANDARD_SECTOR_PRETRAINING
