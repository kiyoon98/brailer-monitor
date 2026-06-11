"""Method A: geometry-based volume and biomass estimation."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .calibration import CameraCalibration
from .detector import Detection


@dataclass(frozen=True)
class VolumeEstimate:
    fill_ratio: float
    volume_m3: float
    weight_kg_geom: float
    outer_area_px: float
    fill_area_px: float


class VolumeEstimator:
    """Estimate brailer volume from segmentation mask and pixel calibration."""

    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration

    def estimate(self, frame: np.ndarray, detection: Detection) -> VolumeEstimate | None:
        if detection.mask is None:
            return self._estimate_from_bbox(detection)

        mask = self._resize_mask(detection.mask, frame.shape[:2])
        outer_area_px = float(np.count_nonzero(mask > 0.5))
        if outer_area_px <= 0:
            return None

        fill_area_px = self._estimate_fill_area(frame, mask)
        fill_ratio = min(1.0, fill_area_px / outer_area_px) if outer_area_px > 0 else 0.0

        x1, y1, x2, y2 = detection.bbox_xyxy
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        depth_px = min(width_px, height_px)

        volume_cm3 = self.calibration.volume_cm3_from_bbox(width_px, height_px, depth_px)
        volume_cm3 *= fill_ratio
        volume_m3 = volume_cm3 / 1_000_000.0

        weight_kg = (
            volume_m3
            * self.calibration.bulk_density_kg_per_m3
            * self.calibration.packing_factor
        )

        return VolumeEstimate(
            fill_ratio=round(fill_ratio, 4),
            volume_m3=round(volume_m3, 4),
            weight_kg_geom=round(weight_kg, 3),
            outer_area_px=outer_area_px,
            fill_area_px=fill_area_px,
        )

    def _estimate_from_bbox(self, detection: Detection) -> VolumeEstimate | None:
        x1, y1, x2, y2 = detection.bbox_xyxy
        width_px = max(1.0, x2 - x1)
        height_px = max(1.0, y2 - y1)
        depth_px = min(width_px, height_px)
        outer_area_px = width_px * height_px
        fill_ratio = 0.75

        volume_cm3 = self.calibration.volume_cm3_from_bbox(width_px, height_px, depth_px)
        volume_cm3 *= fill_ratio
        volume_m3 = volume_cm3 / 1_000_000.0
        weight_kg = (
            volume_m3
            * self.calibration.bulk_density_kg_per_m3
            * self.calibration.packing_factor
        )
        return VolumeEstimate(
            fill_ratio=fill_ratio,
            volume_m3=round(volume_m3, 4),
            weight_kg_geom=round(weight_kg, 3),
            outer_area_px=outer_area_px,
            fill_area_px=outer_area_px * fill_ratio,
        )

    @staticmethod
    def _resize_mask(mask: np.ndarray, frame_shape: tuple[int, int]) -> np.ndarray:
        h, w = frame_shape
        if mask.shape[0] == h and mask.shape[1] == w:
            return mask
        return cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def _estimate_fill_area(frame: np.ndarray, mask: np.ndarray) -> float:
        """Estimate fish fill inside brailer mask using dark-region thresholding."""
        masked = frame.copy()
        binary_mask = (mask > 0.5).astype(np.uint8)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
        roi = cv2.bitwise_and(gray, gray, mask=binary_mask)
        if np.count_nonzero(binary_mask) == 0:
            return 0.0
        mean_val = float(np.mean(roi[binary_mask > 0]))
        threshold = min(mean_val * 0.85, 90.0)
        _, dark = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY_INV)
        dark = cv2.bitwise_and(dark, dark, mask=binary_mask)
        return float(np.count_nonzero(dark))
