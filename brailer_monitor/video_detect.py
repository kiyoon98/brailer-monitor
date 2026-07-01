"""Run YOLO detection on video and collect per-frame results."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import BrailerDetector, Detection

logger = logging.getLogger(__name__)
DEFAULT_SAM_MODEL = "sam2_t.pt"


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


def _draw_detections(frame: np.ndarray, detections: list[dict[str, Any]]) -> np.ndarray:
    vis = frame.copy()
    for det in detections:
        bbox = det.get("bbox_xyxy") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        color = (0, 200, 80)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{det.get('class_name', 'object')} {float(det.get('confidence') or 0):.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
        polygon = det.get("polygon_xy") or []
        if polygon:
            pts = np.array([[int(round(float(x))), int(round(float(y)))] for x, y in polygon], dtype=np.int32)
            mask_u8 = np.zeros(vis.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_u8, [pts], 1)
            mask = mask_u8.astype(bool)
            overlay = vis.copy()
            overlay[mask] = (overlay[mask] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
            vis = cv2.addWeighted(overlay, 0.4, vis, 0.6, 0)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    return vis


def _mask_to_frame_bool(mask: np.ndarray, frame_w: int, frame_h: int) -> np.ndarray:
    if mask.ndim == 3:
        mask = mask[0]
    if mask.shape[0] != frame_h or mask.shape[1] != frame_w:
        mask = cv2.resize(
            mask.astype(np.float32),
            (frame_w, frame_h),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0.5


def _clip_mask_to_bbox(frame_mask: np.ndarray, bbox_xyxy: tuple[float, float, float, float]) -> np.ndarray:
    clipped = np.zeros_like(frame_mask, dtype=bool)
    frame_h, frame_w = frame_mask.shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    left = max(0, min(frame_w, int(np.floor(x1))))
    top = max(0, min(frame_h, int(np.floor(y1))))
    right = max(0, min(frame_w, int(np.ceil(x2))))
    bottom = max(0, min(frame_h, int(np.ceil(y2))))
    if right <= left or bottom <= top:
        return clipped
    clipped[top:bottom, left:right] = frame_mask[top:bottom, left:right]
    return clipped


def _mask_stats(
    mask: np.ndarray,
    frame_w: int,
    frame_h: int,
    bbox_xyxy: tuple[float, float, float, float],
) -> dict[str, Any]:
    frame_mask = _mask_to_frame_bool(mask, frame_w, frame_h)
    frame_mask = _clip_mask_to_bbox(frame_mask, bbox_xyxy)
    ys, xs = np.where(frame_mask)
    area_px = int(xs.size)
    if area_px == 0:
        return {"mask_area_px": 0, "mask_width_px": 0, "mask_height_px": 0, "polygon_xy": []}

    contours, _ = cv2.findContours(frame_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygon_xy: list[list[float]] = []
    if contours:
        contour = max(contours, key=cv2.contourArea)
        epsilon = max(1.0, 0.003 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon_xy = [[round(float(x), 1), round(float(y), 1)] for [[x, y]] in approx.tolist()]

    return {
        "mask_area_px": area_px,
        "mask_width_px": int(xs.max() - xs.min() + 1),
        "mask_height_px": int(ys.max() - ys.min() + 1),
        "polygon_xy": polygon_xy,
    }


class SamBoxSegmenter:
    """Run SAM/SAM2 with YOLO bounding boxes as prompts."""

    def __init__(self, model_path: str | Path | None = None, *, device: str | int = 0):
        self.model_path = str(model_path or os.environ.get("BRAILER_SAM_MODEL") or DEFAULT_SAM_MODEL)
        self.device = device
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is None:
            from ultralytics import SAM

            self._model = SAM(self.model_path)
            logger.info("Loaded SAM model from %s", self.model_path)
        return self._model

    def segment(self, frame: np.ndarray, bboxes: list[list[float]]) -> list[np.ndarray | None]:
        if not bboxes:
            return []
        model = self._load_model()
        results = model(frame, bboxes=bboxes, device=self.device, verbose=False)
        if not results:
            return [None for _ in bboxes]
        result = results[0]
        masks_obj = getattr(result, "masks", None)
        if masks_obj is None or getattr(masks_obj, "data", None) is None:
            return [None for _ in bboxes]
        masks = masks_obj.data.cpu().numpy()
        out: list[np.ndarray | None] = []
        for index in range(len(bboxes)):
            out.append(masks[index] if index < len(masks) else None)
        return out


def _bbox_area_px(det: Detection) -> int:
    x1, y1, x2, y2 = det.bbox_xyxy
    return int(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _detection_to_dict(
    det: Detection,
    frame_w: int,
    frame_h: int,
    *,
    sam_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    mask = sam_mask if sam_mask is not None else det.mask
    out = {
        "class_id": det.class_id,
        "class_name": det.class_name,
        "confidence": round(det.confidence, 4),
        "bbox_xyxy": [round(v, 1) for v in det.bbox_xyxy],
        "track_id": det.track_id,
        "area_px": _bbox_area_px(det),
        "segmentation_source": "sam2" if sam_mask is not None else "yolo" if det.mask is not None else "bbox",
    }
    if mask is not None:
        stats = _mask_stats(mask, frame_w, frame_h, det.bbox_xyxy)
        out.update(stats)
        out["area_px"] = stats["mask_area_px"]
    return out


def detect_video(
    video_path: Path,
    model_path: Path,
    *,
    output_dir: Path,
    frame_stride: int = 1,
    confidence: float = 0.35,
    device: str | int = 0,
    imgsz: int = 416,
    use_segmentation: bool | None = None,
    use_sam: bool = False,
    max_frames: int | None = None,
    save_previews: bool = True,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_detection: Callable[[dict[str, Any]], None] | None = None,
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
        imgsz=imgsz,
    )
    sam_segmenter = SamBoxSegmenter(device=device) if use_sam else None

    while frame_index < total_frames:
        if should_cancel and should_cancel():
            raise DetectionCancelled("Detection cancelled by user")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.predict(frame)
        sam_masks: list[np.ndarray | None] = [None for _ in detections]
        if sam_segmenter is not None and detections:
            sam_masks = sam_segmenter.segment(
                frame,
                [[float(v) for v in det.bbox_xyxy] for det in detections],
            )
        det_dicts = [
            _detection_to_dict(d, width, height, sam_mask=sam_masks[index] if index < len(sam_masks) else None)
            for index, d in enumerate(detections)
        ]
        preview_name: str | None = None

        if save_previews and detections:
            preview_name = f"frame_{frame_index:06d}.jpg"
            vis = _draw_detections(frame, det_dicts)
            cv2.imwrite(str(preview_dir / preview_name), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frame_detection = FrameDetection(
            frame_index=frame_index,
            timestamp_sec=round(frame_index / fps, 3),
            detections=det_dicts,
            preview_path=preview_name,
        )
        frames_out.append(frame_detection)

        processed += 1
        if det_dicts:
            with_detections += 1
            if on_detection is not None:
                event = frame_detection.to_dict()
                event["width"] = width
                event["height"] = height
                on_detection(event)
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
        "imgsz": imgsz,
        "use_sam": use_sam,
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


def detect_stream(
    stream_url: str,
    model_path: Path,
    *,
    output_dir: Path,
    frame_stride: int = 5,
    confidence: float = 0.35,
    device: str | int = 0,
    imgsz: int = 416,
    use_segmentation: bool | None = None,
    use_sam: bool = False,
    save_previews: bool = True,
    on_progress: Callable[[int, int, int], None] | None = None,
    on_detection: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Detect objects in a live stream until cancellation; save a manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    if use_segmentation is None:
        use_segmentation = _meta_task_type() == "segment" or "seg" in model_path.name.lower()
    if frame_stride < 1:
        frame_stride = 1

    def _open_capture() -> cv2.VideoCapture:
        capture = cv2.VideoCapture(stream_url)
        if capture.isOpened():
            return capture
        capture.release()
        raise RuntimeError(f"Cannot open stream: {stream_url}")

    cap = _open_capture()

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    detector = BrailerDetector(
        model_path=model_path,
        confidence_threshold=confidence,
        device=device,
        use_segmentation=use_segmentation,
        imgsz=imgsz,
    )
    sam_segmenter = SamBoxSegmenter(device=device) if use_sam else None

    frames_out: list[FrameDetection] = []
    frame_index = 0
    processed = 0
    with_detections = 0
    failed_reads = 0
    reconnects = 0
    last_reconnect_at = 0.0
    started_at = time.monotonic()

    if on_progress is not None:
        on_progress(0, 0, 0)

    while True:
        if should_cancel and should_cancel():
            break

        ok, frame = cap.read()
        if not ok:
            failed_reads += 1
            if failed_reads >= 15:
                now = time.monotonic()
                if now - last_reconnect_at >= 1.0:
                    reconnects += 1
                    last_reconnect_at = now
                    logger.warning(
                        "Stream read failed %d times; reconnecting to %s (attempt %d)",
                        failed_reads,
                        stream_url,
                        reconnects,
                    )
                    cap.release()
                    while True:
                        if should_cancel and should_cancel():
                            break
                        try:
                            cap = _open_capture()
                            fps = cap.get(cv2.CAP_PROP_FPS) or fps or 15.0
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or width or 0)
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or height or 0)
                            failed_reads = 0
                            break
                        except RuntimeError:
                            time.sleep(1.0)
                    if should_cancel and should_cancel():
                        break
            time.sleep(0.2)
            continue
        failed_reads = 0

        if frame_index % frame_stride != 0:
            frame_index += 1
            continue

        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]

        detections = detector.predict(frame)
        sam_masks: list[np.ndarray | None] = [None for _ in detections]
        if sam_segmenter is not None and detections:
            sam_masks = sam_segmenter.segment(
                frame,
                [[float(v) for v in det.bbox_xyxy] for det in detections],
            )
        det_dicts = [
            _detection_to_dict(d, width, height, sam_mask=sam_masks[index] if index < len(sam_masks) else None)
            for index, d in enumerate(detections)
        ]
        preview_name: str | None = None

        if save_previews and detections:
            preview_name = f"frame_{frame_index:06d}.jpg"
            vis = _draw_detections(frame, det_dicts)
            cv2.imwrite(str(preview_dir / preview_name), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frame_detection = FrameDetection(
            frame_index=frame_index,
            timestamp_sec=round(time.monotonic() - started_at, 3),
            detections=det_dicts,
            preview_path=preview_name,
        )
        frames_out.append(frame_detection)

        processed += 1
        if det_dicts:
            with_detections += 1
            if on_detection is not None:
                event = frame_detection.to_dict()
                event["width"] = width
                event["height"] = height
                on_detection(event)
        if on_progress is not None:
            on_progress(processed, 0, with_detections)

        frame_index += 1

    cap.release()

    duration = frames_out[-1].timestamp_sec if frames_out else round(time.monotonic() - started_at, 3)
    manifest = {
        "video": stream_url,
        "model": str(model_path.resolve()),
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": 0,
        "duration_sec": duration,
        "frame_stride": frame_stride,
        "imgsz": imgsz,
        "use_sam": use_sam,
        "stream": True,
        "frames_processed": len(frames_out),
        "frames_with_detections": sum(1 for f in frames_out if f.detections),
        "frames": [f.to_dict() for f in frames_out],
    }
    manifest_path = output_dir / "detections.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Stream detection done: %d frames, %d with objects -> %s",
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
