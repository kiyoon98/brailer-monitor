"""Brailer transfer event model and serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any


class ReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXCLUDED = "excluded"


@dataclass(frozen=True)
class BrailerEvent:
    timestamp: str
    camera_id: str
    track_id: str
    event_type: str = "brailer_transfer"
    fill_ratio: float = 0.0
    volume_m3: float = 0.0
    weight_kg_geom: float = 0.0
    weight_kg_std: float = 0.0
    weight_kg_est: float = 0.0
    confidence: float = 0.0
    video_clip_ref: str = ""
    review_status: ReviewStatus = ReviewStatus.PENDING

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["review_status"] = self.review_status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BrailerEvent:
        status = payload.get("review_status", ReviewStatus.PENDING)
        if isinstance(status, str):
            status = ReviewStatus(status)
        return cls(
            timestamp=str(payload["timestamp"]),
            camera_id=str(payload["camera_id"]),
            track_id=str(payload["track_id"]),
            event_type=str(payload.get("event_type", "brailer_transfer")),
            fill_ratio=float(payload.get("fill_ratio", 0.0)),
            volume_m3=float(payload.get("volume_m3", 0.0)),
            weight_kg_geom=float(payload.get("weight_kg_geom", 0.0)),
            weight_kg_std=float(payload.get("weight_kg_std", 0.0)),
            weight_kg_est=float(payload.get("weight_kg_est", 0.0)),
            confidence=float(payload.get("confidence", 0.0)),
            video_clip_ref=str(payload.get("video_clip_ref", "")),
            review_status=status,
        )

    def with_review(self, status: ReviewStatus) -> BrailerEvent:
        return replace(self, review_status=status)


CSV_FIELDS = [
    "timestamp",
    "camera_id",
    "track_id",
    "event_type",
    "fill_ratio",
    "volume_m3",
    "weight_kg_geom",
    "weight_kg_std",
    "weight_kg_est",
    "confidence",
    "video_clip_ref",
    "review_status",
]


def load_events_json(path: Path) -> list[BrailerEvent]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("events JSON must be a list")
    return [BrailerEvent.from_dict(item) for item in payload]


def save_events_json(events: list[BrailerEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([event.to_dict() for event in events], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def save_events_csv(events: list[BrailerEvent], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for event in events:
            row = event.to_dict()
            row["review_status"] = event.review_status.value
            writer.writerow(row)


def load_events_csv(path: Path) -> list[BrailerEvent]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [BrailerEvent.from_dict(row) for row in reader]
