"""Offline tests for the future recurrent/transfer history policy."""

import pytest

from reinforcement_learning.history_policy import (
    COLD_START_MINIMUM_USABLE_OBSERVATIONS,
    MATURE_MINIMUM_USABLE_OBSERVATIONS,
    HistoryClass,
    classify_usable_history,
)


@pytest.mark.parametrize(
    ("usable_observations", "expected"),
    (
        (126, HistoryClass.MATURE),
        (127, HistoryClass.MATURE),
        (1_000, HistoryClass.MATURE),
        (100, HistoryClass.COLD_START),
        (125, HistoryClass.COLD_START),
        (99, HistoryClass.INSUFFICIENT),
        (0, HistoryClass.INSUFFICIENT),
    ),
)
def test_history_classification_boundaries(
    usable_observations: int,
    expected: HistoryClass,
) -> None:
    result = classify_usable_history(usable_observations)

    assert result.history_class is expected
    assert result.usable_observations == usable_observations
    assert result.label == expected.label
    assert result.future_training_route


def test_history_policy_is_explicitly_observation_based() -> None:
    assert MATURE_MINIMUM_USABLE_OBSERVATIONS == 126
    assert COLD_START_MINIMUM_USABLE_OBSERVATIONS == 100
    assert classify_usable_history(125).history_class is HistoryClass.COLD_START

    with pytest.raises(ValueError, match="usable observations"):
        classify_usable_history("2020-01-01")


@pytest.mark.parametrize("invalid", (-1, 99.5, float("nan"), True, None))
def test_history_policy_rejects_missing_or_invented_counts(invalid: object) -> None:
    with pytest.raises(ValueError, match="usable observations"):
        classify_usable_history(invalid)
