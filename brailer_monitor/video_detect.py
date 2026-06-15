"""Run YOLO detection on video and collect per-frame results."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import BrailerDetector, Detection

logger = logging.getLogger(__name__)


class DetectionCancelled(Exception):
    """Raised when the user stops an in-progress video detection job."""


@dataclass
class FrameDetection:
    frame_index: int
    timestamp_sec: float
    detections: list[dict[str, Any]]
    preview_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _draw_detections(frame: np.ndarray, detections: list[Detection]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        color = (0, 200, 80)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.class_name} {det.confidence:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        if det.mask is not None:
            mask = det.mask
            if mask.shape[:2] != vis.shape[:2]:
                mask = cv2.resize(mask.astype(np.float32), (vis.shape[1], vis.shape[0])) > 0.5
            overlay = vis.copy()
            overlay[mask.astype(bool)] = (overlay[mask.astype(bool)] * 0.5 + np.array(color) * 0.5).astype(
                np.uint8
            )
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
    return vis


def _detection_area_px(det: Detection, frame_w: int, frame_h: int) -> int:
    if det.mask is not None:
        mask = det.mask
        if mask.ndim == 3:
            mask = mask[0]
        if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
            mask = cv2.resize(
                mask.astype(np.float32),
                (frame_w, frame_h),
                interpolation=cv2.INTER_NEAREST,
            )
        return int(np.count_nonzero(mask > 0.5))
    x1, y1, x2, y2 = det.bbox_xyxy
    return int(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _detection_to_dict(det: Detection, frame_w: int, frame_h: int) -> dict[str, Any]:
    return {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": round(det.confidence, 4),
        "bbox_xyxy": [round(v, 1) for v in det.bbox_xyxy],
        "track_id": det.track_id,
        "area_px": _detection_area_px(det, frame_w, frame_h),
    }


def detect_video(
    video_path: Path,
    model_path: Path,
    *,
    output_dir: Path,
    frame_stride: int = 1,
    confidence: float = 0.35,
    device: str | int = 0,
    use_segmentation: bool | None = None,
    max_frames: int | None = None,
    save_previews: bool = True,
    on_progress: Callable[[int, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Detect objects in each sampled frame; save manifest + preview images."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    if use_segmentation is None:
        use_segmentation = _meta_task_type() == "segment" or "seg" in model_path.name.lower()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frames_out: list[FrameDetection] = []
    frame_index = 0
    processed = 0
    with_detections = 0

    if frame_stride < 1:
        frame_stride = 1
    planned = len(range(0, total_frames, frame_stride))
    if max_frames is not None:
        planned = min(planned, max_frames)
    total_planned = max(planned, 1)

    if on_progress is not None:
        on_progress(0, total_planned, 0)

    if should_cancel and should_cancel():
        raise DetectionCancelled("Detection cancelled by user")

    detector = BrailerDetector(
        model_path=model_path,
        confidence_threshold=confidence,
        device=device,
        use_segmentation=use_segmentation,
    )

    while frame_index < total_frames:
        if should_cancel and should_cancel():
            raise DetectionCancelled("Detection cancelled by user")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.predict(frame)
        det_dicts = [_detection_to_dict(d, width, height) for d in detections]
        preview_name: str | None = None

        if save_previews and detections:
            preview_name = f"frame_{frame_index:06d}.jpg"
            vis = _draw_detections(frame, detections)
            cv2.imwrite(str(preview_dir / preview_name), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frames_out.append(
            FrameDetection(
                frame_index=frame_index,
                timestamp_sec=round(frame_index / fps, 3),
                detections=det_dicts,
                preview_path=preview_name,
            )
        )

        processed += 1
        if det_dicts:
            with_detections += 1
        if on_progress is not None:
            on_progress(processed, total_planned, with_detections)

        if max_frames is not None and processed >= max_frames:
            break
        frame_index += frame_stride

    cap.release()

    manifest = {
        "video": str(video_path.resolve()),
        "model": str(model_path.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frame_stride": frame_stride,
        "frames_processed": len(frames_out),
        "frames_with_detections": sum(1 for f in frames_out if f.detections),
        "frames": [f.to_dict() for f in frames_out],
    }
    manifest_path = output_dir / "detections.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    logger.info(
        "Detection done: %d frames, %d with objects -> %s",
        len(frames_out),
        manifest["frames_with_detections"],
        manifest_path,
    )
    return manifest


def _meta_task_type() -> str:
    meta_path = Path("data/dataset/import_meta.json")
    if not meta_path.exists():
        return "detect"
    return json.loads(meta_path.read_text(encoding="utf-8")).get("task_type", "detect")
