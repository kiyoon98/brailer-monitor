"""Video analysis pipeline: detect, track, estimate, emit events."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import cv2

from .calibration import CameraCalibration, StandardCapacityConfig
from .detector import BrailerDetector
from .events import BrailerEvent, ReviewStatus
from .tracker import TrackManager
from .transfer_counter import TransferCounter
from .volume_estimator import VolumeEstimator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalyzeOptions:
    model_path: Path
    frame_stride: int = 2
    tracker: str = "bytetrack.yaml"
    device: str | int = 0
    max_frames: int | None = None


def _format_timestamp(base_seconds: float) -> str:
    whole = int(base_seconds)
    frac = int((base_seconds - whole) * 1000)
    td = timedelta(seconds=whole)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"PT{hours:02d}H{minutes:02d}M{seconds:02d}.{frac:03d}S"


def _is_brailer_detection(class_name: str, class_id: int) -> bool:
    """Return true only for brailer detections, not arbitrary high-confidence classes."""
    normalized = class_name.strip().lower()
    if "brailer" in normalized:
        return True
    return class_id == 0 and normalized in {"", "0", "class_0"}


def analyze_video(
    video_path: Path,
    calibration: CameraCalibration,
    capacity: StandardCapacityConfig,
    options: AnalyzeOptions,
) -> list[BrailerEvent]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    detector = BrailerDetector(
        model_path=options.model_path,
        confidence_threshold=capacity.min_detection_confidence,
        device=options.device,
        use_segmentation=True,
    )
    transfer_counter = TransferCounter(calibration)
    volume_estimator = VolumeEstimator(calibration)
    track_manager = TrackManager()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frame_index = 0
    processed = 0
    events: list[BrailerEvent] = []
    recorded_tracks: set[int] = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % options.frame_stride != 0:
                frame_index += 1
                continue

            detections = detector.track(frame, tracker=options.tracker)
            brailer_detections = [
                d
                for d in detections
                if _is_brailer_detection(d.class_name, d.class_id)
            ]

            crossing = transfer_counter.process(brailer_detections)
            track_manager.update(frame_index, brailer_detections, crossing.unload_crossed_ids)

            timestamp = _format_timestamp(frame_index / fps)
            for state in track_manager.completed:
                if state.track_id in recorded_tracks:
                    continue
                recorded_tracks.add(state.track_id)
                det = state.best_detection
                if det is None:
                    continue

                volume = volume_estimator.estimate(frame, det)
                weight_geom = volume.weight_kg_geom if volume else 0.0
                fill_ratio = volume.fill_ratio if volume else 0.0
                volume_m3 = volume.volume_m3 if volume else 0.0
                weight_std = transfer_counter.standard_weight(capacity)

                weight_est, confidence = capacity.combine_weights(
                    weight_kg_geom=weight_geom,
                    weight_kg_std=weight_std,
                    detection_confidence=state.max_confidence,
                    has_geometry=volume is not None,
                )
                review_status = (
                    ReviewStatus.PENDING
                    if confidence < capacity.review_confidence_threshold
                    else ReviewStatus.ACCEPTED
                )

                events.append(
                    BrailerEvent(
                        timestamp=timestamp,
                        camera_id=calibration.camera_id,
                        track_id=f"trk-{state.track_id:04d}",
                        fill_ratio=fill_ratio,
                        volume_m3=volume_m3,
                        weight_kg_geom=weight_geom,
                        weight_kg_std=weight_std,
                        weight_kg_est=round(weight_est, 3),
                        confidence=round(confidence, 4),
                        video_clip_ref=video_path.name,
                        review_status=review_status,
                    )
                )
                logger.info(
                    "Transfer event track=%s est=%.1f kg (geom=%.1f, std=%.1f)",
                    state.track_id,
                    weight_est,
                    weight_geom,
                    weight_std,
                )

            processed += 1
            frame_index += 1
            if options.max_frames is not None and processed >= options.max_frames:
                break
    finally:
        cap.release()

    logger.info("Analysis complete: %d transfer events from %s", len(events), video_path.name)
    return events
