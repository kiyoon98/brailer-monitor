"""Trip and video-level catch aggregation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from .events import BrailerEvent, ReviewStatus


@dataclass(frozen=True)
class CatchSummary:
    transfer_count: int
    total_weight_kg_geom: float
    total_weight_kg_std: float
    total_weight_kg_est: float
    pending_review_count: int
    excluded_count: int
    by_video: dict[str, dict[str, float | int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(
    events: list[BrailerEvent],
    review_confidence_threshold: float = 0.65,
) -> CatchSummary:
    included = [e for e in events if e.review_status != ReviewStatus.EXCLUDED]
    pending = sum(
        1
        for e in included
        if e.review_status == ReviewStatus.PENDING and e.confidence < review_confidence_threshold
    )
    excluded = sum(1 for e in events if e.review_status == ReviewStatus.EXCLUDED)

    by_video: dict[str, dict[str, float | int]] = defaultdict(
        lambda: {
            "transfer_count": 0,
            "weight_kg_geom": 0.0,
            "weight_kg_std": 0.0,
            "weight_kg_est": 0.0,
        }
    )
    for event in included:
        key = event.video_clip_ref or "unknown"
        bucket = by_video[key]
        bucket["transfer_count"] = int(bucket["transfer_count"]) + 1
        bucket["weight_kg_geom"] = float(bucket["weight_kg_geom"]) + event.weight_kg_geom
        bucket["weight_kg_std"] = float(bucket["weight_kg_std"]) + event.weight_kg_std
        bucket["weight_kg_est"] = float(bucket["weight_kg_est"]) + event.weight_kg_est

    return CatchSummary(
        transfer_count=len(included),
        total_weight_kg_geom=round(sum(e.weight_kg_geom for e in included), 3),
        total_weight_kg_std=round(sum(e.weight_kg_std for e in included), 3),
        total_weight_kg_est=round(sum(e.weight_kg_est for e in included), 3),
        pending_review_count=pending,
        excluded_count=excluded,
        by_video=dict(by_video),
    )
