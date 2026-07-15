"""Run YOLO detection on video and collect per-frame results."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import BrailerDetector, Detection
from .video_time import absolute_frame_time

logger = logging.getLogger(__name__)
DEFAULT_SAM_MODEL = "sam2_t.pt"
ENSEMBLE_IOU_THRESHOLD = 0.5
ENSEMBLE_MIN_OVERLAP_RATIO = 0.65
DARK_VIDEO_SAMPLE_COUNT = 5
DARK_MEAN_STRICT_THRESHOLD = 35.0
DARK_MEAN_THRESHOLD = 45.0
DARK_STD_THRESHOLD = 18.0
DARK_P90_THRESHOLD = 70.0
SEA_HUE_MIN = 88
SEA_HUE_MAX = 128
SEA_MIN_SATURATION = 35
SEA_MIN_VALUE = 30
SEA_RATIO_SAMPLE_MAX_DIM = 360
SEA_RATIO_METHOD = "hsv_lab_grabcut_v3"
DEFAULT_SEA_ANALYSIS_INTERVAL_SEC = 5.0
MAX_SEA_ANALYSIS_INTERVAL_SEC = 300.0
SEA_HORIZON_CENTER_X_MIN = 0.25
SEA_HORIZON_CENTER_X_MAX = 0.75
SEA_HORIZON_ROW_RATIO_THRESHOLD = 0.18
SEA_HORIZON_SCAN_MAX_Y = 0.72
SEA_ROI_X_MIN = 0.18
SEA_ROI_X_MAX = 0.82
SEA_ROI_Y_MAX = 0.78
SEA_COMPONENT_MIN_AREA_RATIO = 0.0005
SEA_COMPONENT_STRONG_AREA_RATIO = 0.012
SEA_COMPONENT_MIN_WIDTH_RATIO = 0.08
SEA_COMPONENT_MIN_ROI_OVERLAP_RATIO = 0.08
SEA_EDGE_SEED_MIN_SCORE = 0.15
SEA_EDGE_SEED_BEST_RATIO = 0.85
SEA_EDGE_SEED_MIN_PIXELS = 80
DEFAULT_DETECT_ROI_MARGIN = 0.15


class DetectionCancelled(Exception):
    """Raised when the user stops an in-progress video detection job."""


@dataclass
class FrameDetection:
    frame_index: int
    timestamp_sec: float
    detections: list[dict[str, Any]]
    preview_path: str | None = None
    sea_ratio: float | None = None
    sea_percent: float | None = None
    sea_area_px: int | None = None
    sea_method: str | None = None
    sea_horizon_y: int | None = None
    sea_roi_xyxy: list[int] | None = None
    sea_candidate_area_px: int | None = None
    semantic_sea_ratio: float | None = None
    legacy_sea_ratio: float | None = None
    vessel_ratio: float | None = None
    sea_confidence: float | None = None
    sea_quality: str | None = None
    sea_state: str | None = None
    sea_event: str | None = None
    sea_baseline_ratio: float | None = None
    sea_drop_ratio: float | None = None
    vessel_baseline_ratio: float | None = None
    vessel_increase_ratio: float | None = None
    sea_fallback_reason: str | None = None
    sea_horizon_score: float | None = None
    detect_roi: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _draw_polygon_mask(
    vis: np.ndarray,
    polygon: list[Any],
    *,
    fill_color: tuple[int, int, int],
    line_color: tuple[int, int, int],
    alpha: float = 0.4,
    thickness: int = 2,
) -> np.ndarray:
    if not polygon:
        return vis
    pts = np.array([[int(round(float(x))), int(round(float(y)))] for x, y in polygon], dtype=np.int32)
    mask_u8 = np.zeros(vis.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_u8, [pts], 1)
    mask = mask_u8.astype(bool)
    overlay = vis.copy()
    overlay[mask] = (overlay[mask] * 0.5 + np.array(fill_color) * 0.5).astype(np.uint8)
    vis = cv2.addWeighted(overlay, alpha, vis, 1.0 - alpha, 0)
    cv2.polylines(vis, [pts], True, line_color, thickness)
    return vis


def _draw_detection_roi(vis: np.ndarray, detect_roi: dict[str, Any] | None) -> np.ndarray:
    if not detect_roi:
        return vis
    xyxy = detect_roi.get("xyxy_px") or []
    if len(xyxy) != 4:
        return vis
    x1, y1, x2, y2 = [int(round(float(v))) for v in xyxy]
    color = (0, 165, 255)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        vis,
        "detect ROI",
        (x1, max(y1 - 7, 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )
    return vis


def _draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
    *,
    detect_roi: dict[str, Any] | None = None,
) -> np.ndarray:
    vis = frame.copy()
    vis = _draw_detection_roi(vis, detect_roi)
    for det in detections:
        bbox = det.get("bbox_xyxy") or []
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in bbox]
        color = (0, 200, 80)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        model_label = det.get("model_name") or ", ".join(det.get("ensemble_model_names") or [])
        suffix = f" · {model_label}" if model_label else ""
        label = f"{det.get('class_name', 'object')} {float(det.get('confidence') or 0):.2f}{suffix}"
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
        yolo_polygon = det.get("yolo_polygon_xy") or []
        if not yolo_polygon and det.get("segmentation_source") == "yolo":
            yolo_polygon = polygon
            polygon = []
        if yolo_polygon:
            vis = _draw_polygon_mask(
                vis,
                yolo_polygon,
                fill_color=(255, 255, 0),
                line_color=(255, 255, 0),
                alpha=0.25,
                thickness=2,
            )
        if polygon:
            vis = _draw_polygon_mask(
                vis,
                polygon,
                fill_color=(0, 255, 255),
                line_color=(0, 255, 255),
                alpha=0.4,
                thickness=2,
            )
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


def _normalize_model_specs(
    model_path: Path | None,
    model_specs: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    specs = list(model_specs or [])
    if not specs:
        if model_path is None:
            raise ValueError("No detection model provided")
        specs = [{"path": str(model_path)}]

    normalized: list[dict[str, Any]] = []
    for index, spec in enumerate(specs):
        raw_path = str(spec.get("path") or spec.get("weights_path") or "").strip()
        if not raw_path:
            raise ValueError("Model spec is missing path")
        path = Path(raw_path)
        name = str(spec.get("name") or path.stem)
        normalized.append(
            {
                "id": str(spec.get("id") or f"model_{index + 1}"),
                "name": name,
                "path": str(path),
                "task_type": str(spec.get("task_type") or ""),
            }
        )
    return normalized


def _use_segmentation_for_spec(default: bool | None, spec: dict[str, Any]) -> bool:
    if default is not None:
        return default
    task_type = str(spec.get("task_type") or "").lower()
    if task_type:
        return task_type == "segment"
    path = Path(str(spec.get("path") or ""))
    return _meta_task_type() == "segment" or "seg" in path.name.lower()


def _build_detectors(
    model_path: Path | None,
    model_specs: list[dict[str, Any]] | None,
    *,
    confidence: float,
    device: str | int,
    use_segmentation: bool | None,
    imgsz: int,
) -> tuple[list[tuple[dict[str, Any], BrailerDetector]], list[dict[str, Any]]]:
    specs = _normalize_model_specs(model_path, model_specs)
    detectors: list[tuple[dict[str, Any], BrailerDetector]] = []
    for spec in specs:
        detector = BrailerDetector(
            model_path=Path(str(spec["path"])),
            confidence_threshold=confidence,
            device=device,
            use_segmentation=_use_segmentation_for_spec(use_segmentation, spec),
            imgsz=imgsz,
        )
        detectors.append((spec, detector))
    return detectors, specs


def _predict_ensemble(
    detectors: list[tuple[dict[str, Any], BrailerDetector]],
    frame: np.ndarray,
) -> list[Detection]:
    detections: list[Detection] = []
    for spec, detector in detectors:
        for det in detector.predict(frame):
            detections.append(
                replace(
                    det,
                    model_id=str(spec.get("id") or ""),
                    model_name=str(spec.get("name") or ""),
                    model_path=str(spec.get("path") or ""),
                )
            )
    return merge_ensemble_detections(detections)


def _bbox_area_px(det: Detection) -> int:
    x1, y1, x2, y2 = det.bbox_xyxy
    return int(max(0.0, x2 - x1) * max(0.0, y2 - y1))


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = _bbox_intersection_area(a, b)
    if inter <= 0:
        return 0.0
    area_a = _bbox_area(a)
    area_b = _bbox_area(b)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _bbox_min_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    inter = _bbox_intersection_area(a, b)
    min_area = min(_bbox_area(a), _bbox_area(b))
    return inter / min_area if min_area > 0 else 0.0


def _canonical_class_name(class_name: str) -> str:
    normalized = class_name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"brailer", "brailers"}:
        return "brailer"
    return normalized


def _same_ensemble_class(a: Detection, b: Detection) -> bool:
    class_a = _canonical_class_name(a.class_name)
    class_b = _canonical_class_name(b.class_name)
    if class_a and class_b:
        return class_a == class_b
    return a.class_id == b.class_id


def _merge_class_name(det: Detection) -> str:
    return "brailer" if _canonical_class_name(det.class_name) == "brailer" else det.class_name


def _same_ensemble_object(
    a: Detection,
    b: Detection,
    *,
    iou_threshold: float,
) -> bool:
    if not _same_ensemble_class(a, b):
        return False
    return (
        _bbox_iou(a.bbox_xyxy, b.bbox_xyxy) >= iou_threshold
        or _bbox_min_overlap_ratio(a.bbox_xyxy, b.bbox_xyxy) >= ENSEMBLE_MIN_OVERLAP_RATIO
    )


def merge_ensemble_detections(
    detections: list[Detection],
    *,
    iou_threshold: float = ENSEMBLE_IOU_THRESHOLD,
) -> list[Detection]:
    """Merge compatible detections from multiple models using overlap clusters."""
    if len(detections) <= 1:
        return detections

    remaining = sorted(detections, key=lambda det: det.confidence, reverse=True)
    merged: list[Detection] = []
    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        grew = True
        while grew:
            grew = False
            keep: list[Detection] = []
            for det in remaining:
                if any(_same_ensemble_object(det, member, iou_threshold=iou_threshold) for member in cluster):
                    cluster.append(det)
                    grew = True
                else:
                    keep.append(det)
            remaining = keep

        best_confidence = max(cluster, key=lambda det: det.confidence)
        best_shape = max(cluster, key=lambda det: (_bbox_area_px(det), det.confidence))
        model_ids = tuple(
            dict.fromkeys(
                str(det.model_id)
                for det in sorted(cluster, key=lambda item: item.confidence, reverse=True)
                if det.model_id
            )
        )
        model_names = tuple(
            dict.fromkeys(
                str(det.model_name)
                for det in sorted(cluster, key=lambda item: item.confidence, reverse=True)
                if det.model_name
            )
        )
        merged.append(
            replace(
                best_shape,
                confidence=best_confidence.confidence,
                class_id=best_confidence.class_id,
                class_name=_merge_class_name(best_confidence),
                model_id=best_confidence.model_id,
                model_name=best_confidence.model_name,
                model_path=best_confidence.model_path,
                ensemble_model_ids=model_ids,
                ensemble_model_names=model_names,
            )
        )
    return sorted(merged, key=lambda det: det.confidence, reverse=True)


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
    if det.model_id:
        out["model_id"] = det.model_id
    if det.model_name:
        out["model_name"] = det.model_name
    if det.model_path:
        out["model_path"] = det.model_path
    if det.ensemble_model_ids:
        out["ensemble_model_ids"] = list(det.ensemble_model_ids)
    if det.ensemble_model_names:
        out["ensemble_model_names"] = list(det.ensemble_model_names)
    if mask is not None:
        stats = _mask_stats(mask, frame_w, frame_h, det.bbox_xyxy)
        out.update(stats)
        out["area_px"] = stats["mask_area_px"]
    if sam_mask is not None and det.mask is not None:
        yolo_stats = _mask_stats(det.mask, frame_w, frame_h, det.bbox_xyxy)
        out["yolo_mask_area_px"] = yolo_stats["mask_area_px"]
        out["yolo_mask_width_px"] = yolo_stats["mask_width_px"]
        out["yolo_mask_height_px"] = yolo_stats["mask_height_px"]
        out["yolo_polygon_xy"] = yolo_stats["polygon_xy"]
    return out


def _coerce_roi_margin(value: Any, default: float = DEFAULT_DETECT_ROI_MARGIN) -> float:
    try:
        margin = float(value)
    except (TypeError, ValueError):
        margin = default
    if margin > 1.0:
        margin /= 100.0
    return max(0.0, min(0.49, margin))


def normalize_detection_roi_margins(margins: dict[str, Any] | None) -> dict[str, float]:
    source = margins or {}
    normalized = {
        "top": _coerce_roi_margin(source.get("top")),
        "right": _coerce_roi_margin(source.get("right")),
        "bottom": _coerce_roi_margin(source.get("bottom")),
        "left": _coerce_roi_margin(source.get("left")),
    }

    horizontal = normalized["left"] + normalized["right"]
    if horizontal >= 0.98:
        scale = 0.98 / horizontal if horizontal > 0 else 1.0
        normalized["left"] *= scale
        normalized["right"] *= scale
    vertical = normalized["top"] + normalized["bottom"]
    if vertical >= 0.98:
        scale = 0.98 / vertical if vertical > 0 else 1.0
        normalized["top"] *= scale
        normalized["bottom"] *= scale
    return {key: round(value, 4) for key, value in normalized.items()}


def detection_roi_for_frame(
    frame_w: int,
    frame_h: int,
    margins: dict[str, Any] | None = None,
) -> dict[str, Any]:
    width = max(1, int(frame_w))
    height = max(1, int(frame_h))
    normalized = normalize_detection_roi_margins(margins)
    x1 = int(round(width * normalized["left"]))
    x2 = int(round(width * (1.0 - normalized["right"])))
    y1 = int(round(height * normalized["top"]))
    y2 = int(round(height * (1.0 - normalized["bottom"])))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    region_percent = {
        "x_min": round(normalized["left"] * 100.0, 2),
        "x_max": round((1.0 - normalized["right"]) * 100.0, 2),
        "y_min": round(normalized["top"] * 100.0, 2),
        "y_max": round((1.0 - normalized["bottom"]) * 100.0, 2),
    }
    margins_percent = {key: round(value * 100.0, 2) for key, value in normalized.items()}
    return {
        "margins": normalized,
        "margins_percent": margins_percent,
        "region_percent": region_percent,
        "xyxy_px": [x1, y1, x2, y2],
        "frame_width": width,
        "frame_height": height,
        "width_px": x2 - x1,
        "height_px": y2 - y1,
        "label": (
            f"x {region_percent['x_min']:g}-{region_percent['x_max']:g}%, "
            f"y {region_percent['y_min']:g}-{region_percent['y_max']:g}%"
        ),
    }


def _detection_center_in_roi(det: Detection, detect_roi: dict[str, Any]) -> bool:
    xyxy = detect_roi.get("xyxy_px") or []
    if len(xyxy) != 4:
        return True
    roi_x1, roi_y1, roi_x2, roi_y2 = [float(v) for v in xyxy]
    x1, y1, x2, y2 = det.bbox_xyxy
    cx = (float(x1) + float(x2)) / 2.0
    cy = (float(y1) + float(y2)) / 2.0
    return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2


def filter_detections_by_roi(detections: list[Detection], detect_roi: dict[str, Any]) -> list[Detection]:
    return [det for det in detections if _detection_center_in_roi(det, detect_roi)]


def _sea_empty_stats() -> dict[str, Any]:
    return {
        "sea_ratio": 0.0,
        "sea_percent": 0.0,
        "sea_area_px": 0,
        "sea_method": SEA_RATIO_METHOD,
        "sea_horizon_y": None,
        "sea_roi_xyxy": None,
        "sea_candidate_area_px": 0,
    }


def _sea_candidate_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    return (
        (hue >= SEA_HUE_MIN)
        & (hue <= SEA_HUE_MAX)
        & (saturation >= SEA_MIN_SATURATION)
        & (value >= SEA_MIN_VALUE)
    )


def _estimate_horizon_y_from_texture(frame: np.ndarray) -> int:
    frame_h, frame_w = frame.shape[:2]
    if frame_h < 20 or frame_w < 20:
        return 0

    left = max(0, min(frame_w - 1, int(frame_w * 0.45)))
    right = max(left + 1, min(frame_w, int(frame_w * 0.95)))
    scan_top = max(0, min(frame_h - 1, int(frame_h * 0.08)))
    scan_bottom = max(scan_top + 1, min(frame_h, int(frame_h * 0.65)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = _local_gray_std(gray)
    row_gray = gray[:scan_bottom, left:right].mean(axis=1)
    row_std = local_std[:scan_bottom, left:right].mean(axis=1)
    if row_gray.size < 12:
        return 0

    kernel = np.ones(7, dtype=np.float32) / 7.0
    row_gray = np.convolve(row_gray, kernel, mode="same")
    row_std = np.convolve(row_std, kernel, mode="same")
    best_score = 0.0
    best_y = 0
    start = max(scan_top, 4)
    stop = min(scan_bottom - 4, row_gray.size - 4)
    for y in range(start, stop):
        above = float(row_gray[max(0, y - 6) : y].mean())
        below = float(row_gray[y : min(row_gray.size, y + 8)].mean())
        brightness_drop = max(0.0, above - below)
        texture_before = float(row_std[max(0, y - 12) : max(1, y - 4)].mean())
        texture_after = float(row_std[y : min(row_std.size, y + 8)].mean())
        texture_jump = max(0.0, texture_after - texture_before)
        score = brightness_drop * 0.9 + texture_jump * 2.0
        if texture_after < 7.0 and brightness_drop < 12.0:
            score = 0.0
        if score > best_score:
            best_score = score
            best_y = y
    return int(best_y) if best_score >= 10.0 else 0


def _estimate_horizon_y(candidate_mask: np.ndarray, frame: np.ndarray | None = None) -> int:
    frame_h, frame_w = candidate_mask.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return 0
    left = max(0, min(frame_w - 1, int(frame_w * SEA_HORIZON_CENTER_X_MIN)))
    right = max(left + 1, min(frame_w, int(frame_w * SEA_HORIZON_CENTER_X_MAX)))
    scan_bottom = max(1, min(frame_h, int(frame_h * SEA_HORIZON_SCAN_MAX_Y)))
    center_band = candidate_mask[:scan_bottom, left:right]
    if center_band.size == 0:
        return 0

    row_ratios = center_band.mean(axis=1)
    if row_ratios.size >= 7:
        kernel = np.ones(7, dtype=np.float32) / 7.0
        row_ratios = np.convolve(row_ratios, kernel, mode="same")
    hits = np.where(row_ratios >= SEA_HORIZON_ROW_RATIO_THRESHOLD)[0]
    hits = hits[hits >= int(frame_h * 0.02)]
    if hits.size == 0:
        return _estimate_horizon_y_from_texture(frame) if frame is not None else 0
    return max(0, int(hits[0]) - int(frame_h * 0.015))


def _local_gray_std(gray: np.ndarray) -> np.ndarray:
    kernel_size = 9 if min(gray.shape[:2]) >= 80 else 5
    mean = cv2.boxFilter(gray, -1, (kernel_size, kernel_size), normalize=True)
    mean_sq = cv2.boxFilter(gray * gray, -1, (kernel_size, kernel_size), normalize=True)
    return np.sqrt(np.maximum(0.0, mean_sq - mean * mean))


def _sea_ratio_sample_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    frame_h, frame_w = frame.shape[:2]
    max_dim = max(frame_h, frame_w)
    if max_dim <= SEA_RATIO_SAMPLE_MAX_DIM:
        return frame, 1.0
    scale = SEA_RATIO_SAMPLE_MAX_DIM / float(max_dim)
    sample_w = max(1, int(round(frame_w * scale)))
    sample_h = max(1, int(round(frame_h * scale)))
    return cv2.resize(frame, (sample_w, sample_h), interpolation=cv2.INTER_AREA), scale


def _scale_xyxy(values: list[int], scale: float, frame_w: int, frame_h: int) -> list[int]:
    if scale <= 0:
        return [0, 0, frame_w, frame_h]
    x1, y1, x2, y2 = values
    return [
        max(0, min(frame_w, int(round(x1 / scale)))),
        max(0, min(frame_h, int(round(y1 / scale)))),
        max(0, min(frame_w, int(round(x2 / scale)))),
        max(0, min(frame_h, int(round(y2 / scale)))),
    ]


def _edge_seed_masks(
    region_mask: np.ndarray,
    hsv: np.ndarray,
    candidate_mask: np.ndarray,
    local_std: np.ndarray,
    horizon_y: int,
) -> tuple[np.ndarray, list[float]]:
    frame_h, frame_w = region_mask.shape[:2]
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    not_deck_green = ~((hue >= 42) & (hue <= 82) & (saturation >= 45))
    not_red = ~(((hue <= 12) | (hue >= 165)) & (saturation >= 45))
    seed_y1 = max(0, min(frame_h - 1, horizon_y + int(frame_h * 0.04)))
    seed_y2 = max(seed_y1 + 1, min(frame_h, int(frame_h * 0.86)))
    seed_specs = ((0, int(frame_w * 0.32)), (int(frame_w * 0.58), frame_w))
    candidates: list[tuple[float, np.ndarray]] = []
    for x1, x2 in seed_specs:
        seed_rect = np.zeros_like(region_mask, dtype=bool)
        seed_rect[seed_y1:seed_y2, x1:x2] = True
        valid = (
            seed_rect
            & region_mask
            & not_deck_green
            & not_red
            & (value >= 35)
            & (value <= 235)
            & ((saturation <= 95) | candidate_mask | (local_std >= 8.0))
        )
        score = float(np.count_nonzero(valid)) / float(max(1, np.count_nonzero(seed_rect & region_mask)))
        candidates.append((score, valid))

    best_score = max((score for score, _valid in candidates), default=0.0)
    selected = np.zeros_like(region_mask, dtype=bool)
    scores: list[float] = []
    for score, valid in candidates:
        scores.append(round(score, 4))
        if (
            np.count_nonzero(valid) >= SEA_EDGE_SEED_MIN_PIXELS
            and score >= SEA_EDGE_SEED_MIN_SCORE
            and score >= best_score * SEA_EDGE_SEED_BEST_RATIO
        ):
            selected |= valid
    return selected, scores


def _grabcut_background_sea_mask(
    frame: np.ndarray,
    candidate_mask: np.ndarray,
    horizon_y: int,
) -> tuple[np.ndarray | None, list[int], int]:
    frame_h, frame_w = frame.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return None, [0, 0, 0, 0], 0

    roi_y1 = max(0, min(frame_h - 1, horizon_y))
    roi_y2 = frame_h
    region_mask = np.zeros((frame_h, frame_w), dtype=bool)
    region_mask[roi_y1:roi_y2, :] = True
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    local_std = _local_gray_std(gray)

    seed_mask, _seed_scores = _edge_seed_masks(region_mask, hsv, candidate_mask, local_std, horizon_y)
    if np.count_nonzero(seed_mask) < SEA_EDGE_SEED_MIN_PIXELS:
        return None, [0, roi_y1, frame_w, roi_y2], 0

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    median_lab = np.median(lab[seed_mask], axis=0)
    diff = lab - median_lab
    color_distance = np.sqrt((diff[:, :, 0] * 0.33) ** 2 + diff[:, :, 1] ** 2 + diff[:, :, 2] ** 2)
    seed_distances = color_distance[seed_mask]
    color_threshold = float(np.percentile(seed_distances, 90) + 10.0) if seed_distances.size else 36.0
    color_threshold = max(26.0, min(46.0, color_threshold))

    not_deck_green = ~((hue >= 42) & (hue <= 82) & (saturation >= 45))
    not_red = ~(((hue <= 12) | (hue >= 165)) & (saturation >= 45))
    not_dark = value >= 32
    not_overbright = (value <= 245) | (saturation >= 15)
    water_like = (
        region_mask
        & not_deck_green
        & not_red
        & not_dark
        & not_overbright
        & (
            ((color_distance <= color_threshold) & ((saturation > 58) | (local_std >= 6.0) | seed_mask))
            | (candidate_mask & (saturation <= 150))
        )
    )

    grabcut_mask = np.full((frame_h, frame_w), cv2.GC_BGD, dtype=np.uint8)
    grabcut_mask[region_mask] = cv2.GC_PR_BGD
    grabcut_mask[water_like] = cv2.GC_PR_FGD
    grabcut_mask[seed_mask] = cv2.GC_FGD
    sure_background = (
        (~region_mask)
        | ((value < 28) & region_mask)
        | (((hue >= 42) & (hue <= 82) & (saturation >= 55)) & region_mask)
        | ((((hue <= 12) | (hue >= 165)) & (saturation >= 55)) & region_mask)
    )
    grabcut_mask[sure_background] = cv2.GC_BGD
    grabcut_mask[(value < 45) & region_mask & ~candidate_mask] = cv2.GC_PR_BGD
    if not np.any(grabcut_mask == cv2.GC_FGD) or not np.any(grabcut_mask == cv2.GC_BGD):
        return None, [0, roi_y1, frame_w, roi_y2], int(np.count_nonzero(water_like))

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(frame, grabcut_mask, None, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None, [0, roi_y1, frame_w, roi_y2], int(np.count_nonzero(water_like))

    foreground = (
        ((grabcut_mask == cv2.GC_FGD) | (grabcut_mask == cv2.GC_PR_FGD))
        & region_mask
        & (water_like | seed_mask)
    )
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(foreground.astype(np.uint8), 8)
    min_area = max(24, int(frame_h * frame_w * SEA_COMPONENT_MIN_AREA_RATIO))
    filtered = np.zeros_like(foreground, dtype=bool)
    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = labels == component_id
        seed_overlap = int(np.count_nonzero(component & seed_mask))
        if seed_overlap >= max(10, int(area * 0.015)):
            filtered[component] = True

    return filtered, [0, roi_y1, frame_w, roi_y2], int(np.count_nonzero(water_like))


def _filter_background_sea_mask(candidate_mask: np.ndarray, horizon_y: int) -> tuple[np.ndarray, list[int]]:
    frame_h, frame_w = candidate_mask.shape[:2]
    if frame_h <= 0 or frame_w <= 0:
        return candidate_mask, [0, 0, 0, 0]

    kernel_size = 5 if min(frame_h, frame_w) >= 80 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_u8 = candidate_mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    smoothed = mask_u8 > 0

    roi_x1 = max(0, min(frame_w - 1, int(frame_w * SEA_ROI_X_MIN)))
    roi_x2 = max(roi_x1 + 1, min(frame_w, int(frame_w * SEA_ROI_X_MAX)))
    roi_y1 = max(0, min(frame_h - 1, horizon_y))
    roi_y2 = max(roi_y1 + 1, min(frame_h, int(frame_h * SEA_ROI_Y_MAX)))
    reference_roi = np.zeros_like(smoothed, dtype=bool)
    reference_roi[roi_y1:roi_y2, roi_x1:roi_x2] = True

    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(smoothed.astype(np.uint8), 8)
    frame_area = frame_h * frame_w
    min_area = max(24, int(frame_area * SEA_COMPONENT_MIN_AREA_RATIO))
    strong_area = max(min_area, int(frame_area * SEA_COMPONENT_STRONG_AREA_RATIO))
    min_width = max(3, int(frame_w * SEA_COMPONENT_MIN_WIDTH_RATIO))
    filtered = np.zeros_like(smoothed, dtype=bool)

    for component_id in range(1, component_count):
        area = int(stats[component_id, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        x = int(stats[component_id, cv2.CC_STAT_LEFT])
        y = int(stats[component_id, cv2.CC_STAT_TOP])
        width = int(stats[component_id, cv2.CC_STAT_WIDTH])
        height = int(stats[component_id, cv2.CC_STAT_HEIGHT])
        component = labels == component_id
        reference_overlap = int(np.count_nonzero(component & reference_roi))
        overlap_ratio = reference_overlap / area if area > 0 else 0.0
        touches_side = x <= 1 or x + width >= frame_w - 1
        touches_upper_background = y <= max(horizon_y + int(frame_h * 0.08), int(frame_h * 0.12))
        broad_component = width >= min_width or area >= strong_area
        central_background = (
            reference_overlap >= min_area
            and overlap_ratio >= SEA_COMPONENT_MIN_ROI_OVERLAP_RATIO
            and broad_component
        )
        large_background = area >= strong_area and broad_component and (touches_side or touches_upper_background)
        lower_foreground_fragment = (
            y > int(frame_h * 0.58)
            and y + height > int(frame_h * 0.78)
            and overlap_ratio < SEA_COMPONENT_MIN_ROI_OVERLAP_RATIO
            and not (area >= strong_area and width >= min_width * 2)
        )
        if (central_background or large_background) and not lower_foreground_fragment:
            filtered[component] = True

    return filtered, [roi_x1, roi_y1, roi_x2, roi_y2]


def sea_mask_for_frame(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the corrected sea mask and the per-frame sea statistics."""
    if frame is None or frame.size == 0:
        return np.zeros((0, 0), dtype=bool), _sea_empty_stats()

    frame_h, frame_w = frame.shape[:2]
    sample_frame, scale = _sea_ratio_sample_frame(frame)
    sample_h, sample_w = sample_frame.shape[:2]
    candidate_mask = _sea_candidate_mask(sample_frame)
    horizon_y = _estimate_horizon_y(candidate_mask, sample_frame)
    sea_mask_sample, roi_xyxy_sample, water_like_area = _grabcut_background_sea_mask(
        sample_frame,
        candidate_mask,
        horizon_y,
    )
    if sea_mask_sample is None:
        sea_mask_sample, roi_xyxy_sample = _filter_background_sea_mask(candidate_mask, horizon_y)
        water_like_area = int(np.count_nonzero(candidate_mask))

    if scale != 1.0:
        sea_mask = cv2.resize(
            sea_mask_sample.astype(np.uint8),
            (frame_w, frame_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        horizon_full = max(0, min(frame_h - 1, int(round(horizon_y / scale))))
        roi_xyxy = _scale_xyxy(roi_xyxy_sample, scale, frame_w, frame_h)
        candidate_area_px = int(round((water_like_area / float(max(1, sample_w * sample_h))) * frame_w * frame_h))
    else:
        sea_mask = sea_mask_sample
        horizon_full = int(horizon_y)
        roi_xyxy = roi_xyxy_sample
        candidate_area_px = int(water_like_area)

    ratio = float(np.count_nonzero(sea_mask)) / float(sea_mask.size) if sea_mask.size else 0.0
    stats = {
        "sea_ratio": round(ratio, 4),
        "sea_percent": round(ratio * 100.0, 2),
        "sea_area_px": int(round(ratio * frame_w * frame_h)),
        "sea_method": SEA_RATIO_METHOD,
        "sea_horizon_y": horizon_full,
        "sea_roi_xyxy": roi_xyxy,
        "sea_candidate_area_px": candidate_area_px,
    }
    return sea_mask, stats


def estimate_sea_ratio(frame: np.ndarray) -> dict[str, Any]:
    """Estimate visible sea fraction while suppressing deck/hull color fragments."""
    _mask, stats = sea_mask_for_frame(frame)
    return stats


def normalize_sea_analysis_interval(value: float | int) -> float:
    interval = float(value)
    if not 0.0 <= interval <= MAX_SEA_ANALYSIS_INTERVAL_SEC:
        raise ValueError(
            f"sea_analysis_interval_sec must be between 0 and {MAX_SEA_ANALYSIS_INTERVAL_SEC:g}"
        )
    return interval


def sea_analysis_due(current_time: float, last_time: float | None, interval_sec: float) -> bool:
    if last_time is None or interval_sec <= 0.0:
        return True
    return float(current_time) - float(last_time) >= interval_sec - 1e-6


def _sea_ratio_summary(frames: list[FrameDetection]) -> dict[str, Any]:
    ratios = [float(frame.sea_ratio) for frame in frames if frame.sea_ratio is not None]
    methods = sorted({str(frame.sea_method) for frame in frames if frame.sea_method})
    method = ",".join(methods) if methods else None
    if not ratios:
        return {
            "frame_count": 0,
            "avg_sea_ratio": None,
            "min_sea_ratio": None,
            "max_sea_ratio": None,
            "avg_sea_percent": None,
            "method": method,
        }
    avg = sum(ratios) / len(ratios)
    return {
        "frame_count": len(ratios),
        "avg_sea_ratio": round(avg, 4),
        "min_sea_ratio": round(min(ratios), 4),
        "max_sea_ratio": round(max(ratios), 4),
        "avg_sea_percent": round(avg * 100.0, 2),
        "method": method,
    }


def _sea_state_summary(frames: list[FrameDetection]) -> dict[str, Any]:
    enabled = [frame for frame in frames if frame.sea_quality is not None]
    states: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    confidences: list[float] = []
    for frame in enabled:
        state = frame.sea_state or "unknown"
        states[state] = states.get(state, 0) + 1
        if frame.sea_confidence is not None:
            confidences.append(float(frame.sea_confidence))
        if frame.sea_event:
            events.append(
                {
                    "event": frame.sea_event,
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                }
            )
    return {
        "frame_count": len(enabled),
        "state_counts": states,
        "unknown_count": states.get("unknown", 0),
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "events": events,
    }


def _frame_darkness_stats(frame: np.ndarray) -> dict[str, float | bool]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    std = float(gray.std())
    p90 = float(np.percentile(gray, 90))
    too_dark = (mean < DARK_MEAN_STRICT_THRESHOLD and p90 < DARK_P90_THRESHOLD) or (
        mean < DARK_MEAN_THRESHOLD and std < DARK_STD_THRESHOLD
    )
    return {
        "mean": round(mean, 2),
        "std": round(std, 2),
        "p90": round(p90, 2),
        "too_dark": bool(too_dark),
    }


def assess_video_darkness(
    cap: cv2.VideoCapture,
    total_frames: int,
    *,
    sample_count: int = DARK_VIDEO_SAMPLE_COUNT,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Sample a video and return whether every readable sample is clearly dark."""
    sample_count = max(1, sample_count)
    if total_frames > 1:
        indices = np.linspace(0, total_frames - 1, min(sample_count, total_frames), dtype=int).tolist()
        indices = list(dict.fromkeys(int(index) for index in indices))
    else:
        indices = [0]

    samples: list[dict[str, Any]] = []
    for index in indices:
        if should_cancel and should_cancel():
            raise DetectionCancelled("Detection cancelled by user")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            continue
        stats = _frame_darkness_stats(frame)
        stats["frame_index"] = index
        samples.append(stats)

    dark_count = sum(1 for sample in samples if sample.get("too_dark"))
    sample_total = len(samples)
    return {
        "sample_count": sample_total,
        "dark_sample_count": dark_count,
        "all_samples_dark": sample_total > 0 and dark_count == sample_total,
        "thresholds": {
            "mean_strict_lt": DARK_MEAN_STRICT_THRESHOLD,
            "mean_lt": DARK_MEAN_THRESHOLD,
            "std_lt": DARK_STD_THRESHOLD,
            "p90_lt": DARK_P90_THRESHOLD,
        },
        "samples": samples,
    }


def detect_video(
    video_path: Path,
    model_path: Path | None,
    *,
    output_dir: Path,
    model_specs: list[dict[str, Any]] | None = None,
    frame_stride: int = 1,
    confidence: float = 0.6,
    device: str | int = 0,
    imgsz: int = 416,
    use_segmentation: bool | None = None,
    use_sam: bool = True,
    detect_roi_margins: dict[str, Any] | None = None,
    calculate_sea_ratio: bool = False,
    sea_only: bool = False,
    sea_engine: str = "hybrid",
    sea_device: str | int = "cpu",
    sea_analysis_interval_sec: float = DEFAULT_SEA_ANALYSIS_INTERVAL_SEC,
    sea_state_path: Path | None = None,
    skip_dark_video: bool = False,
    max_frames: int | None = None,
    save_previews: bool = True,
    on_progress: Callable[[int, int, int, dict[str, Any] | None], None] | None = None,
    on_detection: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Detect objects in each sampled frame; save manifest + preview images."""
    sea_only = bool(sea_only)
    if sea_only:
        calculate_sea_ratio = True
        use_sam = False
    sea_analysis_interval_sec = normalize_sea_analysis_interval(sea_analysis_interval_sec)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect_roi = detection_roi_for_frame(width, height, detect_roi_margins)

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
        on_progress(0, total_planned, 0, None)

    if should_cancel and should_cancel():
        raise DetectionCancelled("Detection cancelled by user")

    model_manifest = [] if sea_only else _normalize_model_specs(model_path, model_specs)
    dark_assessment: dict[str, Any] | None = None
    if skip_dark_video:
        dark_assessment = assess_video_darkness(
            cap,
            total_frames,
            should_cancel=should_cancel,
        )
        if dark_assessment.get("all_samples_dark"):
            cap.release()
            if on_progress is not None:
                on_progress(total_planned, total_planned, 0, None)
            manifest = {
                "video": str(video_path.resolve()),
                "model": None if sea_only else str(Path(model_manifest[0]["path"]).resolve()),
                "models": model_manifest,
                "ensemble": len(model_manifest) > 1,
                "object_detection_enabled": not sea_only,
                "sea_only": sea_only,
                "fps": fps,
                "width": width,
                "height": height,
                "total_frames": total_frames,
                "frame_stride": frame_stride,
                "confidence": confidence,
                "imgsz": imgsz,
                "use_sam": use_sam,
                "detect_roi": detect_roi,
                "sea_ratio_enabled": bool(calculate_sea_ratio),
                "sea_engine": sea_engine,
                "sea_device": str(sea_device),
                "sea_analysis_interval_sec": sea_analysis_interval_sec,
                "dark_skip_enabled": True,
                "dark_video_assessment": dark_assessment,
                "skipped": True,
                "skip_reason": "dark_video",
                "frames_processed": 0,
                "frames_with_detections": 0,
                "sea_ratio_summary": _sea_ratio_summary(frames_out),
                "sea_state_summary": _sea_state_summary(frames_out),
                "frames": [],
            }
            manifest_path = output_dir / "detections.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            logger.info("Skipped dark video: %s -> %s", video_path, manifest_path)
            return manifest

    if sea_only:
        detectors = []
    else:
        detectors, model_manifest = _build_detectors(
            model_path,
            model_manifest,
            confidence=confidence,
            device=device,
            use_segmentation=use_segmentation,
            imgsz=imgsz,
        )
    sam_segmenter = SamBoxSegmenter(device=device) if use_sam and not sea_only else None
    if calculate_sea_ratio:
        from .sea_area_analysis import SeaAreaAnalyzer

        sea_analyzer = SeaAreaAnalyzer(device=sea_device, engine=sea_engine, state_path=sea_state_path)
    else:
        sea_analyzer = None
    last_sea_analysis_at: float | None = None

    while frame_index < total_frames:
        if should_cancel and should_cancel():
            raise DetectionCancelled("Detection cancelled by user")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        frame_timestamp = frame_index / fps
        frame_absolute_time = absolute_frame_time(video_path.name, frame_timestamp)
        sea_timestamp = frame_absolute_time.timestamp() if frame_absolute_time is not None else frame_timestamp
        if sea_analyzer is not None and sea_analysis_due(
            frame_timestamp,
            last_sea_analysis_at,
            sea_analysis_interval_sec,
        ):
            sea_stats = sea_analyzer.analyze(frame, timestamp_sec=sea_timestamp)
            last_sea_analysis_at = frame_timestamp
        else:
            sea_stats = {}
        detections = [] if sea_only else _predict_ensemble(detectors, frame)
        if not sea_only:
            detections = filter_detections_by_roi(detections, detect_roi)
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
            vis = _draw_detections(frame, det_dicts, detect_roi=detect_roi)
            cv2.imwrite(str(preview_dir / preview_name), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frame_detection = FrameDetection(
            frame_index=frame_index,
            timestamp_sec=round(frame_index / fps, 3),
            detections=det_dicts,
            preview_path=preview_name,
            detect_roi=detect_roi,
            **sea_stats,
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
            on_progress(processed, total_planned, with_detections, sea_stats or None)

        if max_frames is not None and processed >= max_frames:
            break
        frame_index += frame_stride

    cap.release()

    manifest = {
        "video": str(video_path.resolve()),
        "model": None if sea_only else str(Path(model_manifest[0]["path"]).resolve()),
        "models": model_manifest,
        "ensemble": len(model_manifest) > 1,
        "object_detection_enabled": not sea_only,
        "sea_only": sea_only,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": total_frames,
        "frame_stride": frame_stride,
        "confidence": confidence,
        "imgsz": imgsz,
        "use_sam": use_sam,
        "detect_roi": detect_roi,
        "sea_ratio_enabled": bool(calculate_sea_ratio),
        "sea_engine": sea_engine,
        "sea_device": str(sea_device),
        "sea_analysis_interval_sec": sea_analysis_interval_sec,
        "dark_skip_enabled": skip_dark_video,
        "dark_video_assessment": dark_assessment,
        "skipped": False,
        "frames_processed": len(frames_out),
        "frames_with_detections": sum(1 for f in frames_out if f.detections),
        "sea_ratio_summary": _sea_ratio_summary(frames_out),
        "sea_state_summary": _sea_state_summary(frames_out),
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
    model_path: Path | None,
    *,
    output_dir: Path,
    model_specs: list[dict[str, Any]] | None = None,
    frame_stride: int = 5,
    confidence: float = 0.6,
    device: str | int = 0,
    imgsz: int = 416,
    use_segmentation: bool | None = None,
    use_sam: bool = True,
    detect_roi_margins: dict[str, Any] | None = None,
    calculate_sea_ratio: bool = False,
    sea_only: bool = False,
    sea_engine: str = "hybrid",
    sea_device: str | int = "cpu",
    sea_analysis_interval_sec: float = DEFAULT_SEA_ANALYSIS_INTERVAL_SEC,
    sea_state_path: Path | None = None,
    save_previews: bool = True,
    on_progress: Callable[[int, int, int, dict[str, Any] | None], None] | None = None,
    on_detection: Callable[[dict[str, Any]], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Detect objects in a live stream until cancellation; save a manifest."""
    sea_only = bool(sea_only)
    if sea_only:
        calculate_sea_ratio = True
        use_sam = False
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    if save_previews:
        preview_dir.mkdir(parents=True, exist_ok=True)

    if frame_stride < 1:
        frame_stride = 1
    sea_analysis_interval_sec = normalize_sea_analysis_interval(sea_analysis_interval_sec)

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
    detect_roi: dict[str, Any] | None = None

    if sea_only:
        detectors = []
        model_manifest: list[dict[str, Any]] = []
    else:
        detectors, model_manifest = _build_detectors(
            model_path,
            model_specs,
            confidence=confidence,
            device=device,
            use_segmentation=use_segmentation,
            imgsz=imgsz,
        )
    sam_segmenter = SamBoxSegmenter(device=device) if use_sam and not sea_only else None
    if calculate_sea_ratio:
        from .sea_area_analysis import SeaAreaAnalyzer

        sea_analyzer = SeaAreaAnalyzer(device=sea_device, engine=sea_engine, state_path=sea_state_path)
    else:
        sea_analyzer = None
    last_sea_analysis_at: float | None = None

    frames_out: list[FrameDetection] = []
    frame_index = 0
    processed = 0
    with_detections = 0
    failed_reads = 0
    reconnects = 0
    last_reconnect_at = 0.0
    started_at = time.monotonic()

    if on_progress is not None:
        on_progress(0, 0, 0, None)

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
        detect_roi = detection_roi_for_frame(width, height, detect_roi_margins)

        sea_elapsed = time.monotonic() - started_at
        if sea_analyzer is not None and sea_analysis_due(
            sea_elapsed,
            last_sea_analysis_at,
            sea_analysis_interval_sec,
        ):
            sea_stats = sea_analyzer.analyze(frame, timestamp_sec=sea_elapsed)
            last_sea_analysis_at = sea_elapsed
        else:
            sea_stats = {}
        detections = [] if sea_only else _predict_ensemble(detectors, frame)
        if not sea_only:
            detections = filter_detections_by_roi(detections, detect_roi)
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
            vis = _draw_detections(frame, det_dicts, detect_roi=detect_roi)
            cv2.imwrite(str(preview_dir / preview_name), vis, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        frame_detection = FrameDetection(
            frame_index=frame_index,
            timestamp_sec=round(time.monotonic() - started_at, 3),
            detections=det_dicts,
            preview_path=preview_name,
            detect_roi=detect_roi,
            **sea_stats,
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
            on_progress(processed, 0, with_detections, sea_stats or None)

        frame_index += 1

    cap.release()

    duration = frames_out[-1].timestamp_sec if frames_out else round(time.monotonic() - started_at, 3)
    manifest = {
        "video": stream_url,
        "model": None if sea_only else str(Path(model_manifest[0]["path"]).resolve()),
        "models": model_manifest,
        "ensemble": len(model_manifest) > 1,
        "object_detection_enabled": not sea_only,
        "sea_only": sea_only,
        "fps": fps,
        "width": width,
        "height": height,
        "total_frames": 0,
        "duration_sec": duration,
        "frame_stride": frame_stride,
        "confidence": confidence,
        "imgsz": imgsz,
        "use_sam": use_sam,
        "detect_roi": detect_roi or detection_roi_for_frame(width or 1, height or 1, detect_roi_margins),
        "sea_ratio_enabled": bool(calculate_sea_ratio),
        "sea_engine": sea_engine,
        "sea_device": str(sea_device),
        "sea_analysis_interval_sec": sea_analysis_interval_sec,
        "stream": True,
        "frames_processed": len(frames_out),
        "frames_with_detections": sum(1 for f in frames_out if f.detections),
        "sea_ratio_summary": _sea_ratio_summary(frames_out),
        "sea_state_summary": _sea_state_summary(frames_out),
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
