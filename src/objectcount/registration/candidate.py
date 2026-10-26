"""프로덕트 후보의 보수적 운영 임계값 선택."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdResult:
    threshold: float
    recall: float
    false_positives: int


def select_operating_confidence(
        rows: list[ThresholdResult]) -> ThresholdResult:
    eligible = [row for row in rows if row.false_positives == 0]
    if not eligible:
        raise ValueError("no threshold has zero false positives")
    return max(eligible, key=lambda row: (row.recall, row.threshold))
