"""ROI 폴리곤 — 벨트 영역 밖 오탐 차단 (스펙 §3-①). 중심점 기준 필터."""
from __future__ import annotations

from .types import Detection


class RoiPolygon:
    def __init__(self, points: list[tuple[float, float]]) -> None:
        if len(points) < 3:
            raise ValueError("ROI 폴리곤은 꼭짓점이 3개 이상이어야 합니다")
        self.points = [(float(x), float(y)) for x, y in points]

    def contains(self, x: float, y: float) -> bool:
        """ray casting — 점이 폴리곤 내부에 있는지."""
        inside = False
        n = len(self.points)
        for i in range(n):
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < x_cross:
                    inside = not inside
        return inside

    def filter(self, detections: list[Detection]) -> list[Detection]:
        return [d for d in detections if self.contains(*d.center)]


def parse_roi(text: str) -> RoiPolygon:
    """CLI 인자 "x1,y1;x2,y2;..." 파싱."""
    points = []
    for pair in text.split(";"):
        x, y = pair.split(",")
        points.append((float(x), float(y)))
    return RoiPolygon(points)
