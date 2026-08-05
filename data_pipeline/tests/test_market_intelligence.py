"""Offline coverage for official-index and market-intelligence behavior."""

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from market_intelligence.feature_joiner import build_index_context, join_market_context
from market_intelligence.index_client import IndexClientError, PsxIndexClient
from market_intelligence.index_config import SUPPORTED_INDEX_CODES, require_supported_index
from market_intelligence.index_metrics import calculate_index_metrics
from market_intelligence.index_parser import parse_index_series
from market_intelligence.index_store import build_combined_master, update_index_csv, write_raw_snapshot
from market_intelligence.market_breadth import calculate_market_breadth
from market_intelligence.market_health import MARKET_HEALTH_WEIGHTS, calculate_market_health, market_health_label
from market_intelligence.refresh_indices import refresh_indices


class Response:
    def __init__(self, payload, content_type="application/json"):
        self.payload = payload
        self.headers = {"Content-Type": content_type}
    def raise_for_status(self): return None
    def json(self): return self.payload


class Session:
    def __init__(self, response): self.response, self.headers, self.calls = response, {}, []
    def get(self, url, timeout): self.calls.append((url, timeout)); return self.response


def payload():
    return {"status": 1, "message": "", "data": [[1628252141, 100, 10, 99], [1628338541, 102, 20, 101]]}


def test_allowlist_and_client_envelope_validation():
    assert SUPPORTED_INDEX_CODES == ("KSE100", "KSE30", "KMI30", "ALLSHR")
    with pytest.raises(ValueError): require_supported_index("PSX100")
    session = Session(Response(payload()))
    assert PsxIndexClient(session=session).fetch_index_series("KSE100")["status"] == 1
    assert session.calls[0][0].endswith("/KSE100")
    for response in (Response(payload(), "text/html"), Response({"status": 0, "data": []}), Response({"status": 1, "data": []})):
        with pytest.raises(IndexClientError): PsxIndexClient(session=Session(response)).fetch_index_series("KSE100")


def test_parser_converts_karachi_sorts_rejects_and_deduplicates():
    source = {"data": [[1628338541, 102, 20, 101], [1628252141, 100, 10, 99], [1628252142, 101, 11, 100], [1, None]]}
    result = parse_index_series(source, "KSE100", fetched_at=datetime(2026, 1, 1, tzinfo=ZoneInfo("Asia/Karachi")))
    assert result.data["date"].tolist() == ["2021-08-06", "2021-08-07"]
    assert result.data.iloc[0]["timestamp"] == 1628252142
    assert result.data.iloc[1]["daily_change"] == 1
    assert len(result.rejected) == 1


def test_storage_is_atomic_and_combined_master_is_idempotent(tmp_path: Path):
    raw = tmp_path / "raw" / "KSE100.json"
    write_raw_snapshot(payload(), raw)
    assert raw.exists() and not list(raw.parent.glob("*.tmp"))
    parsed = parse_index_series(payload(), "KSE100").data
    path = update_index_csv("KSE100", parsed, tmp_path / "KSE100.csv")
    update_index_csv("KSE100", parsed, path)
    combined = build_combined_master((path,), output_path=tmp_path / "master.csv")
    assert len(combined) == 2


def test_metrics_breadth_health_and_feature_join_are_leakage_safe():
    dates = pd.date_range("2025-01-01", periods=60)
    indices = pd.DataFrame({"index_code": "KSE100", "date": dates, "value": range(100, 160), "volume": 1000})
    metric = calculate_index_metrics(indices, "KSE100")
    assert metric.one_week_return is not None and metric.versus_ma_50_percent is not None
    empty_metric = calculate_index_metrics(indices, "KSE30")
    assert empty_metric.latest_value is None
    equities = pd.DataFrame({"symbol": ["A", "B", "A"], "date": ["2025-01-01", "2025-01-01", "2025-01-02"], "change": [1, -1, 0], "volume": [10, 20, 30]})
    breadth = calculate_market_breadth(equities)
    assert breadth.reference_date.isoformat() == "2025-01-02" and breadth.universe_size == 1
    assert sum(MARKET_HEALTH_WEIGHTS.values()) == 100
    health = calculate_market_health({"KSE100": metric}, breadth)
    assert health.score is not None and 0 <= health.score <= 100
    assert market_health_label(29) == "Strongly Bearish" and market_health_label(71) == "Strongly Bullish"
    context = build_index_context(indices)
    target = pd.DataFrame({"symbol": ["A", "A"], "date": ["2025-01-03", "2025-01-04"]})
    joined = join_market_context(target, context, max_forward_fill_days=0)
    assert joined.loc[0, "kse100_value"] == 102
    assert joined.loc[1, "kse100_value"] == 103


def test_refresh_continues_after_partial_failure_and_preserves_cache(tmp_path: Path, monkeypatch):
    class Client:
        def fetch_index_series(self, code):
            if code == "KSE30": raise RuntimeError("offline")
            return payload()
    import market_intelligence.index_config as config
    for code in ("KSE100", "KSE30"):
        definition = config.SUPPORTED_INDICES[code]
        monkeypatch.setattr(definition.__class__, "raw_path", property(lambda self: tmp_path / "raw" / f"{self.code}.json"))
        monkeypatch.setattr(definition.__class__, "master_path", property(lambda self: tmp_path / "master" / f"{self.code}.csv"))
        break
    (tmp_path / "master").mkdir()
    (tmp_path / "master" / "KSE30.csv").write_text("index_code,date\nKSE30,2025-01-01\n")
    result = refresh_indices(("KSE100", "KSE30"), client=Client(), metadata_path=tmp_path / "meta.json", combined_path=tmp_path / "combined.csv")
    assert result.successful_indices == ("KSE100",)
    assert result.failed_indices and result.cached_data_used is True
