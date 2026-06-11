"""Detect brailer-visible segments in EM video and extract frames for labeling."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BrailerBBox:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class BrailerSegment:
    segment_id: int
    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float
    detection_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExtractedFrame:
    image_path: Path
    frame_index: int
    timestamp_sec: float
    segment_id: int
    bbox: BrailerBBox | None

    def to_dict(self) -> dict:
        payload = {
            "image": self.image_path.name,
            "frame_index": self.frame_index,
            "timestamp_sec": round(self.timestamp_sec, 3),
            "segment_id": self.segment_id,
        }
        if self.bbox is not None:
            payload["bbox"] = list(self.bbox.as_tuple)
            payload["score"] = round(self.bbox.score, 2)
        return payload


@dataclass(frozen=True)
class ExtractOptions:
    scan_stride: int = 15
    extract_stride: int = 15
    gap_tolerance_sec: float = 3.0
    segment_padding_sec: float = 0.5
    jpeg_quality: int = 92
    draw_bbox_preview: bool = False


def find_brailer_bbox(frame: np.ndarray) -> BrailerBBox | None:
    """Coarse dark-blob detector for a loaded brailer in the upper-center ROI."""
    h, w = frame.shape[:2]
    y1 = int(h * 0.48)
    x0 = int(w * 0.22)
    x1 = int(w * 0.78)
    roi = frame[0:y1, x0:x1]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 105))
    mask = cv2.bitwise_and(mask, cv2.inRange(gray, 0, 115))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(cv2.inRange(gray, 180, 255)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = h * w
    best_box: tuple[int, int, int, int] | None = None
    best_score = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < frame_area * 0.002 or area > frame_area * 0.09:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        if bw < 35 or bh < 35:
            continue
        aspect = bw / max(bh, 1)
        if aspect < 0.35 or aspect > 2.5:
            continue
        cy = y + bh / 2
        if cy > y1 * 0.85:
            continue
        upper_bonus = 1.2 - (cy / max(y1, 1)) * 0.5
        score = area * upper_bonus
        if score > best_score:
            best_score = score
            best_box = (x + x0, y, x + x0 + bw, y + bh)

    if best_box is None:
        return None
    return BrailerBBox(
        x1=best_box[0],
        y1=best_box[1],
        x2=best_box[2],
        y2=best_box[3],
        score=best_score,
    )


def scan_brailer_segments(
    video_path: Path,
    options: ExtractOptions | None = None,
) -> tuple[list[BrailerSegment], float, int]:
    """
    Scan video at coarse stride and return merged brailer-visible segments.

    Returns (segments, fps, total_frames).
    """
    opts = options or ExtractOptions()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    gap_tolerance_frames = max(1, int(opts.gap_tolerance_sec * fps))
    padding_frames = max(0, int(opts.segment_padding_sec * fps))

    segments: list[BrailerSegment] = []
    in_segment = False
    seg_start = 0
    seg_end = 0
    detection_count = 0
    missed = 0
    segment_id = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index % opts.scan_stride == 0:
            bbox = find_brailer_bbox(frame)
            if bbox is not None:
                if not in_segment:
                    in_segment = True
                    seg_start = frame_index
                    detection_count = 0
                seg_end = frame_index
                detection_count += 1
                missed = 0
            elif in_segment:
                missed += opts.scan_stride
                if missed >= gap_tolerance_frames:
                    start = max(0, seg_start - padding_frames)
                    end = min(total_frames - 1, seg_end + padding_frames)
                    segments.append(
                        BrailerSegment(
                            segment_id=segment_id,
                            start_frame=start,
                            end_frame=end,
                            start_sec=start / fps,
                            end_sec=end / fps,
                            detection_count=detection_count,
                        )
                    )
                    segment_id += 1
                    in_segment = False
                    missed = 0

        frame_index += 1

    if in_segment:
        start = max(0, seg_start - padding_frames)
        end = min(total_frames - 1, seg_end + padding_frames)
        segments.append(
            BrailerSegment(
                segment_id=segment_id,
                start_frame=start,
                end_frame=end,
                start_sec=start / fps,
                end_sec=end / fps,
                detection_count=detection_count,
            )
        )

    cap.release()
    logger.info(
        "Found %d brailer segments in %s (%.1fs, %d frames)",
        len(segments),
        video_path.name,
        total_frames / fps,
        total_frames,
    )
    return segments, fps, total_frames


def _frame_stem(prefix: str, timestamp_sec: float, frame_index: int) -> str:
    return f"{prefix}_{int(timestamp_sec):05d}s_f{frame_index:05d}"


def extract_brailer_frames(
    video_path: Path,
    output_dir: Path,
    *,
    prefix: str | None = None,
    options: ExtractOptions | None = None,
    segments: list[BrailerSegment] | None = None,
    preview_dir: Path | None = None,
) -> tuple[list[ExtractedFrame], list[BrailerSegment]]:
    """
    Extract frames from brailer-visible segments only.

    If segments is None, scans the video first.
    """
    opts = options or ExtractOptions()
    output_dir.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    if segments is None:
        segments, fps, _ = scan_brailer_segments(video_path, opts)
    else:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.release()

    if not segments:
        logger.warning("No brailer segments found in %s", video_path)
        return [], segments

    frame_prefix = prefix or video_path.stem
    extracted: list[ExtractedFrame] = []
    cap = cv2.VideoCapture(str(video_path))

    for segment in segments:
        for frame_index in range(segment.start_frame, segment.end_frame + 1, opts.extract_stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue

            timestamp = frame_index / fps
            bbox = find_brailer_bbox(frame)
            stem = _frame_stem(frame_prefix, timestamp, frame_index)
            image_path = output_dir / f"{stem}.jpg"

            cv2.imwrite(
                str(image_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), opts.jpeg_quality],
            )

            if preview_dir and bbox is not None:
                vis = frame.copy()
                cv2.rectangle(
                    vis,
                    (bbox.x1, bbox.y1),
                    (bbox.x2, bbox.y2),
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(preview_dir / f"{stem}_preview.jpg"), vis)

            extracted.append(
                ExtractedFrame(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp_sec=timestamp,
                    segment_id=segment.segment_id,
                    bbox=bbox,
                )
            )

    cap.release()
    logger.info(
        "Extracted %d frames from %d segments -> %s",
        len(extracted),
        len(segments),
        output_dir,
    )
    return extracted, segments


def save_segment_manifest(
    segments: list[BrailerSegment],
    extracted: list[ExtractedFrame],
    path: Path,
    *,
    video_path: Path | None = None,
    fps: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": video_path.name if video_path else None,
        "fps": fps,
        "segment_count": len(segments),
        "frame_count": len(extracted),
        "segments": [s.to_dict() for s in segments],
        "frames": [f.to_dict() for f in extracted],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_segment_manifest(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
