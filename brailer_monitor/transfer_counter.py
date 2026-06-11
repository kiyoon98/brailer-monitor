"""Method B: brailer transfer counting via supervision zones."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration import CameraCalibration, StandardCapacityConfig
from .detector import Detection


@dataclass(frozen=True)
class ZoneCrossingResult:
    unload_crossed_ids: set[int]
    in_zone_ids: set[int]


class TransferCounter:
    """Detect when tracked brailers complete a transfer using polygon + line zones."""

    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        self._zone = None
        self._line_zone = None
        self._init_supervision()

    def _init_supervision(self) -> None:
        import supervision as sv

        zone = self.calibration.transfer_zone
        if zone is None or not zone.polygon:
            return

        polygon = np.array(zone.polygon, dtype=np.int32)
        self._zone = sv.PolygonZone(polygon=polygon)
        self._line_zone = sv.LineZone(
            start=sv.Point(zone.unload_line_start[0], zone.unload_line_start[1]),
            end=sv.Point(zone.unload_line_end[0], zone.unload_line_end[1]),
        )

    def process(self, detections: list[Detection]) -> ZoneCrossingResult:
        if self._zone is None or self._line_zone is None:
            track_ids = {d.track_id for d in detections if d.track_id is not None}
            return ZoneCrossingResult(unload_crossed_ids=set(), in_zone_ids=track_ids)

        import supervision as sv

        if not detections:
            empty = sv.Detections.empty()
            self._zone.trigger(empty)
            self._line_zone.trigger(empty, empty)
            return ZoneCrossingResult(unload_crossed_ids=set(), in_zone_ids=set())

        xyxy = np.array([d.bbox_xyxy for d in detections], dtype=np.float32)
        conf = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id = np.array([d.class_id for d in detections], dtype=np.int32)
        tracker_id = np.array(
            [d.track_id if d.track_id is not None else -1 for d in detections],
            dtype=np.int32,
        )
        sv_det = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id,
            tracker_id=tracker_id,
        )

        in_zone = self._zone.trigger(sv_det)
        in_zone_ids = set(int(tid) for tid in tracker_id[in_zone] if tid >= 0)

        crossed_in, _ = self._line_zone.trigger(sv_det, sv_det)
        unload_crossed_ids = set(int(tid) for tid in tracker_id[crossed_in] if tid >= 0)

        return ZoneCrossingResult(
            unload_crossed_ids=unload_crossed_ids,
            in_zone_ids=in_zone_ids,
        )

    @staticmethod
    def standard_weight(capacity: StandardCapacityConfig) -> float:
        return capacity.standard_capacity_kg
