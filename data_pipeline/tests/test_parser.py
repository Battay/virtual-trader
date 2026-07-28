"""Tests for PSX HTML parsing."""

from datetime import date

from data_pipeline.src.parser import parse_market_html


def test_parses_equity_cells_and_numeric_values() -> None:
    html = """
    <table>
      <tr data-type="equity">
        <td data-value="OGDC">ignored</td><td data-value="220.50"></td>
        <td data-value="221.00"></td><td data-value="225.25"></td>
        <td data-value="219.75"></td><td data-value="224.00"></td>
        <td data-value="3.50"></td><td data-value="1.59%"></td>
        <td data-value="1,234,567"></td>
      </tr>
      <tr data-type="index"><td>Ignored</td></tr>
    </table>
    """

    records, rejected = parse_market_html(html, date(2026, 7, 27))

    assert rejected == []
    assert records == [
        {
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
    ]


def test_reports_rows_with_an_unexpected_column_count() -> None:
    html = '<tr data-type="equity"><td data-value="OGDC"></td><td>1</td></tr>'

    records, rejected = parse_market_html(html, date(2026, 7, 27))

    assert records == []
    assert len(rejected) == 1
    assert rejected[0]["reason"] == "expected 9 columns, found 2"


def test_reports_missing_numeric_values_without_fabricating_them() -> None:
    html = """
    <tr data-type="equity">
      <td>ABC</td><td>10</td><td></td><td>12</td><td>9</td>
      <td>11</td><td>1</td><td>10</td><td>2,000</td>
    </tr>
    """

    records, rejected = parse_market_html(html, date(2026, 7, 27))

    assert records == []
    assert "open is empty" in rejected[0]["reason"]
