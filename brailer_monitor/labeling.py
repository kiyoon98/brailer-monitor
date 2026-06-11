"""SAM-based auto-labeling for extracted brailer frames."""

from __future__ import annotations

import logging
import random
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .frame_extractor import BrailerBBox, ExtractedFrame, find_brailer_bbox

logger = logging.getLogger(__name__)

CLASS_ID = 0
CLASS_NAME = "brailer_loaded"


def sam_polygon(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    sam_model: Any,
) -> list[tuple[float, float]] | None:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_x = int((x2 - x1) * 0.08)
    pad_y = int((y2 - y1) * 0.08)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    try:
        results = sam_model(frame, bboxes=np.array([[x1, y1, x2, y2]]), verbose=False)
    except (IndexError, RuntimeError, ValueError) as exc:
        logger.debug("SAM inference failed for bbox %s: %s", bbox, exc)
        return None

    if not results:
        return None

    result = results[0]
    if result.masks is None or len(result.masks.data) == 0:
        return None

    mask = result.masks.data[0].cpu().numpy()
    mask = cv2.resize(mask.astype(np.float32), (w, h)) > 0.5
    area_ratio = cv2.countNonZero(mask.astype(np.uint8)) / (h * w)
    if area_ratio < 0.003 or area_ratio > 0.10:
        return None

    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    epsilon = 0.006 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    if len(approx) < 4:
        return None

    polygon_norm: list[tuple[float, float]] = []
    for point in approx.reshape(-1, 2):
        px = float(np.clip(point[0] / w, 0.0, 1.0))
        py = float(np.clip(point[1] / h, 0.0, 1.0))
        polygon_norm.append((px, py))
    return polygon_norm


def yolo_seg_line(class_id: int, polygon_norm: list[tuple[float, float]]) -> str:
    parts = [str(class_id)]
    for x, y in polygon_norm:
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    return " ".join(parts)


def label_extracted_frames(
    frames: list[ExtractedFrame],
    labels_dir: Path,
    sam_model: Any,
    *,
    preview_dir: Path | None = None,
) -> list[dict]:
    """Generate YOLO-seg polygon labels for pre-extracted frames."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for item in frames:
        frame = cv2.imread(str(item.image_path))
        if frame is None:
            logger.warning("Skipping unreadable frame: %s", item.image_path)
            continue

        bbox = item.bbox
        if bbox is None:
            detected = find_brailer_bbox(frame)
            if detected is None:
                continue
            bbox_tuple = detected.as_tuple
        else:
            bbox_tuple = bbox.as_tuple

        polygon = sam_polygon(frame, bbox_tuple, sam_model)
        if polygon is None:
            continue

        h, w = frame.shape[:2]
        pts = np.array([[int(x * w), int(y * h)] for x, y in polygon], dtype=np.int32)
        area_ratio = cv2.contourArea(pts) / (h * w)

        label_path = labels_dir / f"{item.image_path.stem}.txt"
        label_path.write_text(yolo_seg_line(CLASS_ID, polygon) + "\n", encoding="utf-8")

        if preview_dir:
            vis = frame.copy()
            cv2.rectangle(vis, bbox_tuple[:2], bbox_tuple[2:], (255, 128, 0), 1)
            cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
            cv2.imwrite(str(preview_dir / f"{item.image_path.stem}_preview.jpg"), vis)

        records.append(
            {
                "image": item.image_path.name,
                "frame_index": item.frame_index,
                "timestamp_sec": round(item.timestamp_sec, 2),
                "segment_id": item.segment_id,
                "area_ratio": round(area_ratio, 4),
                "bbox": list(bbox_tuple),
                "points": len(polygon),
            }
        )

    logger.info("Labeled %d / %d frames", len(records), len(frames))
    return records


def split_train_val(
    images_dir: Path,
    labels_dir: Path,
    dataset_root: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    images = sorted(images_dir.glob("*.jpg"))
    random.seed(seed)
    random.shuffle(images)

    val_count = max(1, int(len(images) * val_ratio))
    val_set = {img.name for img in images[:val_count]}

    train_img = dataset_root / "images" / "train"
    val_img = dataset_root / "images" / "val"
    train_lbl = dataset_root / "labels" / "train"
    val_lbl = dataset_root / "labels" / "val"
    for directory in (train_img, val_img, train_lbl, val_lbl):
        directory.mkdir(parents=True, exist_ok=True)
        for old in directory.glob("*"):
            old.unlink()

    train_n = val_n = 0
    for img in images:
        lbl = labels_dir / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        if img.name in val_set:
            shutil.copy2(img, val_img / img.name)
            shutil.copy2(lbl, val_lbl / lbl.name)
            val_n += 1
        else:
            shutil.copy2(img, train_img / img.name)
            shutil.copy2(lbl, train_lbl / lbl.name)
            train_n += 1
    return train_n, val_n


def update_dataset_yaml(root: Path, train_n: int, val_n: int) -> None:
    yaml_path = root / "config" / "dataset.yaml"
    content = f"""# Auto-generated from brailer segment extraction + labeling
path: {root / 'data' / 'dataset'}
train: images/train
val: images/val

names:
  0: {CLASS_NAME}

# train images: {train_n}, val images: {val_n}
"""
    yaml_path.write_text(content, encoding="utf-8")
