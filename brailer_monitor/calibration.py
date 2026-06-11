"""Camera calibration and capacity configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TransferZone:
    polygon: list[tuple[int, int]]
    unload_line_start: tuple[int, int]
    unload_line_end: tuple[int, int]


@dataclass(frozen=True)
class CameraCalibration:
    camera_id: str
    cm_per_pixel: float
    roi_name: str = "transfer_zone"
    frame_width: int = 1920
    frame_height: int = 1080
    bulk_density_kg_per_m3: float = 650.0
    packing_factor: float = 0.85
    brailer_diameter_cm: float = 250.0
    transfer_zone: TransferZone | None = None

    def length_cm(self, length_px: float) -> float:
        if length_px <= 0:
            raise ValueError("length_px must be greater than zero")
        if self.cm_per_pixel <= 0:
            raise ValueError("cm_per_pixel must be greater than zero")
        return length_px * self.cm_per_pixel

    def area_cm2(self, area_px: float) -> float:
        if area_px <= 0:
            return 0.0
        scale = self.cm_per_pixel
        return area_px * scale * scale

    def volume_cm3_from_bbox(self, width_px: float, height_px: float, depth_px: float | None = None) -> float:
        """Approximate ellipsoid volume from 2D bbox dimensions."""
        a = self.length_cm(width_px) / 2
        b = self.length_cm(height_px) / 2
        if depth_px is None:
            c = min(a, b)
        else:
            c = self.length_cm(depth_px) / 2
        return (4.0 / 3.0) * 3.141592653589793 * a * b * c


@dataclass(frozen=True)
class StandardCapacityConfig:
    standard_capacity_kg: float = 1500.0
    confidence_weight_std: float = 0.4
    confidence_weight_geom: float = 0.6
    min_detection_confidence: float = 0.35
    review_confidence_threshold: float = 0.65

    def combine_weights(
        self,
        weight_kg_geom: float,
        weight_kg_std: float,
        detection_confidence: float,
        has_geometry: bool,
    ) -> tuple[float, float]:
        """Return (weight_kg_est, event_confidence)."""
        if not has_geometry:
            return weight_kg_std, detection_confidence

        geom_conf = detection_confidence
        std_conf = 0.9 if weight_kg_std > 0 else 0.0
        total_w = self.confidence_weight_geom + self.confidence_weight_std
        est = (
            weight_kg_geom * self.confidence_weight_geom
            + weight_kg_std * self.confidence_weight_std
        ) / total_w
        conf = (
            geom_conf * self.confidence_weight_geom + std_conf * self.confidence_weight_std
        ) / total_w
        return est, conf


def _parse_polygon(raw: list[list[int]]) -> list[tuple[int, int]]:
    return [(int(p[0]), int(p[1])) for p in raw]


def load_calibration(path: Path) -> CameraCalibration:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    zone_payload = payload.get("transfer_zone", {})
    line_payload = payload.get("unload_line", {})
    polygon = _parse_polygon(zone_payload.get("polygon", []))
    line_start = tuple(line_payload.get("start", [0, 0]))
    line_end = tuple(line_payload.get("end", [0, 0]))
    transfer_zone = None
    if polygon:
        transfer_zone = TransferZone(
            polygon=polygon,
            unload_line_start=(int(line_start[0]), int(line_start[1])),
            unload_line_end=(int(line_end[0]), int(line_end[1])),
        )
    return CameraCalibration(
        camera_id=str(payload["camera_id"]),
        cm_per_pixel=float(payload["cm_per_pixel"]),
        roi_name=str(payload.get("roi_name", "transfer_zone")),
        frame_width=int(payload.get("frame_width", 1920)),
        frame_height=int(payload.get("frame_height", 1080)),
        bulk_density_kg_per_m3=float(payload.get("bulk_density_kg_per_m3", 650.0)),
        packing_factor=float(payload.get("packing_factor", 0.85)),
        brailer_diameter_cm=float(payload.get("brailer_diameter_cm", 250.0)),
        transfer_zone=transfer_zone,
    )


def load_capacity_config(path: Path) -> StandardCapacityConfig:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return StandardCapacityConfig(
        standard_capacity_kg=float(payload.get("standard_capacity_kg", 1500.0)),
        confidence_weight_std=float(payload.get("confidence_weight_std", 0.4)),
        confidence_weight_geom=float(payload.get("confidence_weight_geom", 0.6)),
        min_detection_confidence=float(payload.get("min_detection_confidence", 0.35)),
        review_confidence_threshold=float(payload.get("review_confidence_threshold", 0.65)),
    )
