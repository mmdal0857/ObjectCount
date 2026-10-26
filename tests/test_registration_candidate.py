import pytest

from objectcount.registration.candidate import (
    ThresholdResult,
    select_operating_confidence,
)


def test_selects_highest_recall_among_zero_false_positive_thresholds():
    rows = [
        ThresholdResult(0.25, recall=0.98, false_positives=2),
        ThresholdResult(0.40, recall=0.94, false_positives=0),
        ThresholdResult(0.55, recall=0.87, false_positives=0),
    ]

    selected = select_operating_confidence(rows)

    assert selected.threshold == 0.40


def test_prefers_higher_threshold_when_recall_is_tied():
    rows = [
        ThresholdResult(0.40, recall=0.94, false_positives=0),
        ThresholdResult(0.55, recall=0.94, false_positives=0),
    ]

    assert select_operating_confidence(rows).threshold == 0.55


def test_rejects_when_no_threshold_has_zero_false_positives():
    rows = [ThresholdResult(0.50, recall=0.90, false_positives=1)]

    with pytest.raises(ValueError, match="zero false positives"):
        select_operating_confidence(rows)


def test_rejects_empty_threshold_results():
    with pytest.raises(ValueError, match="zero false positives"):
        select_operating_confidence([])
