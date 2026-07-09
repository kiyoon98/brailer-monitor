"""Accumulated detection timeline across multiple video jobs."""

from __future__ import annotations

import json
import os
import uuid
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median
from typing import Any

from .video_time import absolute_frame_time, format_absolute_time, parse_video_start_time

TEMPORAL_ISOLATION_WINDOW_SEC = 10.0
TEMPORAL_SHORT_BURST_MAX_FRAMES = 3
TEMPORAL_SHORT_BURST_MAX_DURATION_SEC = 1.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_timeline() -> dict[str, Any]:
    return {"updated_at": _now_iso(), "videos": [], "events": []}


def _parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def load_timeline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_timeline()
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return _empty_timeline()
    try:
        timeline = json.loads(raw)
    except json.JSONDecodeError as exc:
        if exc.msg != "Extra data":
            raise
        timeline, end = json.JSONDecoder().raw_decode(raw)
        if raw[end:].strip():
            # A previous interrupted write can leave an otherwise valid JSON
            # document followed by stale bytes. Keep the valid prefix readable.
            pass
    if not isinstance(timeline, dict):
        raise ValueError(f"Invalid timeline payload: {path}")
    timeline.setdefault("videos", [])
    timeline.setdefault("events", [])
    return timeline


def save_timeline(path: Path, timeline: dict[str, Any]) -> None:
    timeline["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        temp_path.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def reset_timeline(path: Path) -> None:
    save_timeline(path, _empty_timeline())


def _video_duration_sec(manifest: dict[str, Any]) -> float:
    fps = float(manifest.get("fps") or 15.0)
    total_frames = int(manifest.get("total_frames") or 0)
    if total_frames > 0 and fps > 0:
        return total_frames / fps
    frames = manifest.get("frames") or []
    if frames:
        return float(frames[-1].get("timestamp_sec", 0))
    return 0.0


def detection_area_px(det: dict[str, Any]) -> int:
    """Pixel area inside the detection mask, or bounding-box area as fallback."""
    mask_area = det.get("mask_area_px")
    if mask_area is not None:
        mask_area_int = int(mask_area)
        if mask_area_int > 0:
            return mask_area_int
    area = det.get("area_px")
    if area is not None:
        area_int = int(area)
        if area_int > 0:
            return area_int
    bbox = det.get("bbox_xyxy") or []
    if len(bbox) == 4:
        return int(max(0.0, float(bbox[2]) - float(bbox[0])) * max(0.0, float(bbox[3]) - float(bbox[1])))
    return 0


def _frame_to_dict(
    event: dict[str, Any],
    *,
    class_name: str,
    confidence: float,
    area_px: int,
    bbox_xyxy: list[float] | None = None,
    polygon_xy: list[list[float]] | None = None,
    mask_area_px: int | None = None,
    mask_width_px: int | None = None,
    mask_height_px: int | None = None,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "job_id": event["job_id"],
        "video_name": event["video_name"],
        "frame_index": event.get("frame_index"),
        "timestamp_sec": event.get("timestamp_sec"),
        "absolute_time": event.get("absolute_time"),
        "absolute_time_label": event.get("absolute_time_label"),
        "preview_path": event.get("preview_path"),
        "class_name": class_name,
        "confidence": confidence,
        "area_px": area_px,
    }
    if bbox_xyxy:
        out["bbox_xyxy"] = bbox_xyxy
    if polygon_xy:
        out["polygon_xy"] = polygon_xy
        out["polygon_point_count"] = len(polygon_xy)
    if mask_area_px is not None:
        out["mask_area_px"] = int(mask_area_px)
    if mask_width_px is not None:
        out["mask_width_px"] = int(mask_width_px)
    if mask_height_px is not None:
        out["mask_height_px"] = int(mask_height_px)
    return out


def expand_class_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per (frame, class) with the highest confidence for that class."""
    expanded: list[dict[str, Any]] = []
    for event in events:
        best: dict[str, dict[str, Any]] = {}
        for det in event.get("detections") or []:
            cls = det.get("class_name")
            if not cls:
                continue
            prev = best.get(cls)
            if prev is None or float(det.get("confidence", 0)) > float(prev.get("confidence", 0)):
                best[cls] = det
        for cls, det in best.items():
            bbox = det.get("bbox_xyxy")
            polygon = det.get("polygon_xy")
            expanded.append(
                _frame_to_dict(
                    event,
                    class_name=cls,
                    confidence=float(det.get("confidence", 0)),
                    area_px=detection_area_px(det),
                    bbox_xyxy=list(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
                    polygon_xy=list(polygon) if isinstance(polygon, list) and polygon else None,
                    mask_area_px=det.get("mask_area_px"),
                    mask_width_px=det.get("mask_width_px"),
                    mask_height_px=det.get("mask_height_px"),
                )
            )
    return expanded


def merge_consecutive_frames(
    frames: list[dict[str, Any]],
    *,
    frame_stride: int,
) -> list[list[dict[str, Any]]]:
    """Group sampled frames into segments when frame gaps stay within one stride."""
    if not frames:
        return []
    stride = max(frame_stride, 1)
    ordered = sorted(frames, key=lambda item: item.get("frame_index") or 0)
    groups: list[list[dict[str, Any]]] = [[ordered[0]]]
    for frame in ordered[1:]:
        prev_idx = groups[-1][-1].get("frame_index") or 0
        next_idx = frame.get("frame_index") or 0
        if next_idx - prev_idx <= stride:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _frame_sort_key(frame: dict[str, Any]) -> tuple[str, str, float, int]:
    return (
        str(frame.get("absolute_time") or ""),
        str(frame.get("video_name") or ""),
        float(frame.get("timestamp_sec") or 0.0),
        int(frame.get("frame_index") or 0),
    )


def _segment_start_dt(frames: list[dict[str, Any]]) -> datetime | None:
    return _parse_iso_datetime(frames[0].get("absolute_time")) if frames else None


def _segment_end_dt(frames: list[dict[str, Any]]) -> datetime | None:
    return _parse_iso_datetime(frames[-1].get("absolute_time")) if frames else None


def merge_segment_groups_by_time_gap(
    groups: list[list[dict[str, Any]]],
    *,
    max_gap_sec: float,
) -> list[list[dict[str, Any]]]:
    """Merge adjacent detection segments for the same class when the time gap is small."""
    if not groups or max_gap_sec <= 0:
        return groups

    ordered = sorted(
        groups,
        key=lambda group: (
            str(group[0].get("class_name") or ""),
            _frame_sort_key(group[0]),
        ),
    )
    merged: list[list[dict[str, Any]]] = []
    for group in ordered:
        group = sorted(group, key=_frame_sort_key)
        if not merged:
            merged.append(group)
            continue

        prev = merged[-1]
        same_class = prev[-1].get("class_name") == group[0].get("class_name")
        prev_end = _segment_end_dt(prev)
        next_start = _segment_start_dt(group)
        can_merge = False
        if same_class and prev_end is not None and next_start is not None:
            gap = (next_start - prev_end).total_seconds()
            can_merge = 0 <= gap <= max_gap_sec

        if can_merge:
            prev.extend(group)
            prev.sort(key=_frame_sort_key)
        else:
            merged.append(group)
    return sorted(merged, key=lambda group: _frame_sort_key(group[0]))


def _segment_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
    frames = sorted(frames, key=_frame_sort_key)
    first = frames[0]
    last = frames[-1]
    start_dt = first.get("absolute_time")
    end_dt = last.get("absolute_time")
    if start_dt and not isinstance(start_dt, str):
        start_dt = start_dt.isoformat()
    if end_dt and not isinstance(end_dt, str):
        end_dt = end_dt.isoformat()

    start_label = first.get("absolute_time_label")
    end_label = last.get("absolute_time_label")
    if start_label and end_label and start_label != end_label:
        time_label = f"{start_label} – {end_label}"
    else:
        time_label = start_label or end_label

    confidences = [float(f.get("confidence", 0)) for f in frames]
    areas = [int(f.get("area_px") or 0) for f in frames]
    best_area_frame = max(frames, key=lambda f: int(f.get("area_px") or 0), default=frames[0])
    video_names = list(dict.fromkeys(str(f.get("video_name") or "") for f in frames if f.get("video_name")))
    job_ids = list(dict.fromkeys(str(f.get("job_id") or "") for f in frames if f.get("job_id")))
    segment_id = f"{first['job_id']}:{first['class_name']}:{first.get('frame_index', 0)}"

    out: dict[str, Any] = {
        "segment_id": segment_id,
        "job_id": first["job_id"],
        "video_name": first["video_name"] if len(video_names) <= 1 else f"{first['video_name']} 외 {len(video_names) - 1}개",
        "video_names": video_names,
        "job_ids": job_ids,
        "class_name": first["class_name"],
        "start_absolute_time": start_dt,
        "end_absolute_time": end_dt,
        "start_absolute_time_label": start_label,
        "end_absolute_time_label": end_label,
        "time_label": time_label,
        "start_timestamp_sec": first.get("timestamp_sec"),
        "end_timestamp_sec": last.get("timestamp_sec"),
        "frame_count": len(frames),
        "max_confidence": max(confidences) if confidences else 0.0,
        "max_area_px": max(areas) if areas else 0,
        "preview_path": first.get("preview_path"),
        "preview_job_id": first["job_id"],
    }
    bbox = best_area_frame.get("bbox_xyxy")
    if isinstance(bbox, list) and len(bbox) == 4:
        out["bbox_xyxy"] = bbox
    polygon = best_area_frame.get("polygon_xy")
    if isinstance(polygon, list) and polygon:
        out["polygon_xy"] = polygon
        out["polygon_point_count"] = len(polygon)
    for key in ("mask_area_px", "mask_width_px", "mask_height_px"):
        value = best_area_frame.get(key)
        if value is not None:
            out[key] = int(value)
    return out


def _timeline_merge_gap_sec(timeline: dict[str, Any]) -> float:
    try:
        return max(0.0, float(timeline.get("segment_merge_gap_sec") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def build_segment_frame_groups(
    timeline: dict[str, Any],
    *,
    merge_gap_sec: float | None = None,
) -> list[list[dict[str, Any]]]:
    videos_by_name = {video["video_name"]: video for video in timeline.get("videos", [])}
    class_frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for frame in expand_class_events(timeline.get("events", [])):
        key = (frame["video_name"], frame["class_name"])
        class_frames[key].append(frame)

    groups: list[list[dict[str, Any]]] = []
    for (video_name, _class_name), frames in class_frames.items():
        stride = int(videos_by_name.get(video_name, {}).get("frame_stride") or 5)
        for group in merge_consecutive_frames(frames, frame_stride=stride):
            groups.append(group)

    gap = _timeline_merge_gap_sec(timeline) if merge_gap_sec is None else max(0.0, float(merge_gap_sec))
    if gap > 0:
        groups = merge_segment_groups_by_time_gap(groups, max_gap_sec=gap)
    return sorted(groups, key=lambda group: _frame_sort_key(group[0]))


def build_segments(
    timeline: dict[str, Any],
    *,
    merge_gap_sec: float | None = None,
) -> list[dict[str, Any]]:
    segments = [_segment_from_frames(group) for group in build_segment_frame_groups(timeline, merge_gap_sec=merge_gap_sec)]
    segments.sort(
        key=lambda segment: (
            segment.get("start_absolute_time") or "",
            segment.get("video_name") or "",
            segment.get("start_timestamp_sec") or 0,
        )
    )
    return segments


def _bbox_values(det: dict[str, Any]) -> list[float] | None:
    bbox = det.get("bbox_xyxy") or []
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _flatten_timeline_detections(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for event_index, event in enumerate(timeline.get("events", []) or []):
        for det_index, det in enumerate(event.get("detections") or []):
            bbox = _bbox_values(det)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            area = detection_area_px(det)
            if area <= 0:
                area = int((x2 - x1) * (y2 - y1))
            entries.append(
                {
                    "key": (event_index, det_index),
                    "event": event,
                    "det": det,
                    "bbox": bbox,
                    "width": x2 - x1,
                    "height": y2 - y1,
                    "center": ((x1 + x2) / 2.0, (y1 + y2) / 2.0),
                    "diag": ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5,
                    "area": area,
                    "group": (str(event.get("video_name") or ""), str(det.get("class_name") or "")),
                    "frame_index": int(event.get("frame_index") or 0),
                    "timestamp_sec": float(event.get("timestamp_sec") or 0.0),
                    "job_id": str(event.get("job_id") or ""),
                }
            )
    return entries


def _group_entries(entries: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["group"]].append(entry)
    return groups


def _position_outlier_keys(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    for group in _group_entries(entries).values():
        if len(group) < 5:
            continue
        xs = [entry["center"][0] for entry in group]
        ys = [entry["center"][1] for entry in group]
        mx = median(xs)
        my = median(ys)
        distances = [((entry["center"][0] - mx) ** 2 + (entry["center"][1] - my) ** 2) ** 0.5 for entry in group]
        med_dist = median(distances)
        mad = median([abs(distance - med_dist) for distance in distances])
        med_diag = median([entry["diag"] for entry in group])
        threshold = med_dist + max(3.0 * mad, 1.5 * med_diag, 80.0)
        for entry, distance in zip(group, distances):
            if distance > threshold:
                remove.add(entry["key"])
    return remove


def _size_outlier_keys(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    for group in _group_entries(entries).values():
        areas = [float(entry["area"]) for entry in group if float(entry["area"]) > 0]
        if len(areas) < 3:
            continue
        med_area = median(areas)
        if med_area <= 0:
            continue
        for entry in group:
            area = float(entry["area"])
            if area <= med_area * 0.5 or area >= med_area * 2.0:
                remove.add(entry["key"])
    return remove


def _tall_thin_box_keys(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    for entry in entries:
        width = float(entry.get("width") or 0.0)
        height = float(entry.get("height") or 0.0)
        if width <= 0 or height <= 0:
            continue
        if width / height < 0.5:
            remove.add(entry["key"])
    return remove


def _static_short_track_keys(entries: list[dict[str, Any]], videos: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    sample_steps: dict[str, float] = {}
    for video in videos:
        video_name = str(video.get("video_name") or "")
        try:
            fps = float(video.get("fps") or 0.0)
            stride = int(video.get("frame_stride") or 0)
        except (TypeError, ValueError):
            fps = 0.0
            stride = 0
        sample_steps[video_name] = stride / fps if fps > 0 and stride > 0 else 1.0

    for (video_name, _class_name), group in _group_entries(entries).items():
        ordered = sorted(group, key=lambda entry: (entry["frame_index"], entry["timestamp_sec"]))
        runs: list[list[dict[str, Any]]] = []
        for entry in ordered:
            expected_step = max(0.001, sample_steps.get(video_name, 1.0))
            max_time_gap = max(1.0, expected_step * 1.75)
            if not runs:
                runs.append([entry])
                continue
            prev = runs[-1][-1]
            time_gap = entry["timestamp_sec"] - prev["timestamp_sec"]
            if 0 < time_gap <= max_time_gap:
                runs[-1].append(entry)
            else:
                runs.append([entry])

        for run in runs:
            duration_sec = float(run[-1]["timestamp_sec"]) - float(run[0]["timestamp_sec"])
            if duration_sec < 3.0 or duration_sec > 4.0:
                continue
            cx0, cy0 = run[0]["center"]
            max_move = max(((entry["center"][0] - cx0) ** 2 + (entry["center"][1] - cy0) ** 2) ** 0.5 for entry in run)
            med_diag = median([entry["diag"] for entry in run])
            if max_move <= max(2.0, med_diag * 0.03):
                remove.update(entry["key"] for entry in run)
    return remove


def _similar_temporal_detection(a: dict[str, Any], b: dict[str, Any]) -> bool:
    area_a = float(a.get("area") or 0.0)
    area_b = float(b.get("area") or 0.0)
    width_a = float(a.get("width") or 0.0)
    width_b = float(b.get("width") or 0.0)
    height_a = float(a.get("height") or 0.0)
    height_b = float(b.get("height") or 0.0)
    if min(area_a, area_b, width_a, width_b, height_a, height_b) <= 0:
        return False

    area_ratio = area_b / area_a
    width_ratio = width_b / width_a
    height_ratio = height_b / height_a
    if not (0.5 <= area_ratio <= 2.0 and 0.5 <= width_ratio <= 2.0 and 0.5 <= height_ratio <= 2.0):
        return False

    ax, ay = a["center"]
    bx, by = b["center"]
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    max_diag = max(float(a.get("diag") or 0.0), float(b.get("diag") or 0.0))
    return distance <= max(80.0, max_diag * 0.75)


def _temporal_isolated_keys(
    entries: list[dict[str, Any]],
    *,
    window_sec: float = TEMPORAL_ISOLATION_WINDOW_SEC,
    protect_tail_sec: float = 0.0,
) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    window = max(0.0, float(window_sec))
    if window <= 0:
        return remove

    for group in _group_entries(entries).values():
        latest_timestamp = max((float(entry["timestamp_sec"]) for entry in group), default=0.0)
        def can_remove(entry: dict[str, Any]) -> bool:
            return protect_tail_sec <= 0 or float(entry["timestamp_sec"]) < latest_timestamp - protect_tail_sec

        if len(group) < 2:
            remove.update(entry["key"] for entry in group if can_remove(entry))
            continue
        ordered = sorted(group, key=lambda entry: (entry["timestamp_sec"], entry["frame_index"]))
        for index, entry in enumerate(ordered):
            if not can_remove(entry):
                continue
            has_neighbor = False
            for other_index, other in enumerate(ordered):
                if other_index == index:
                    continue
                time_gap = abs(float(other["timestamp_sec"]) - float(entry["timestamp_sec"]))
                if time_gap <= 0.0001:
                    continue
                if time_gap > window:
                    continue
                if _similar_temporal_detection(entry, other):
                    has_neighbor = True
                    break
            if not has_neighbor:
                remove.add(entry["key"])

        runs: list[list[dict[str, Any]]] = []
        for entry in ordered:
            if not runs:
                runs.append([entry])
                continue
            prev = runs[-1][-1]
            time_gap = float(entry["timestamp_sec"]) - float(prev["timestamp_sec"])
            if 0.0 < time_gap <= TEMPORAL_SHORT_BURST_MAX_DURATION_SEC and _similar_temporal_detection(prev, entry):
                runs[-1].append(entry)
            else:
                runs.append([entry])

        for run in runs:
            if len(run) < 2:
                continue
            if len(run) > TEMPORAL_SHORT_BURST_MAX_FRAMES:
                continue
            if not all(can_remove(entry) for entry in run):
                continue
            duration_sec = float(run[-1]["timestamp_sec"]) - float(run[0]["timestamp_sec"])
            if duration_sec <= TEMPORAL_SHORT_BURST_MAX_DURATION_SEC:
                remove.update(entry["key"] for entry in run)
    return remove


def _crop_color_stats(video_path: Path, frame_index: int, bbox: list[float]) -> dict[str, float] | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    left = max(0, min(width, int(x1)))
    top = max(0, min(height, int(y1)))
    right = max(0, min(width, int(x2)))
    bottom = max(0, min(height, int(y2)))
    if right <= left or bottom <= top:
        return None
    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(np.float32)
    sat = hsv[:, :, 1].astype(np.float32)
    val = hsv[:, :, 2].astype(np.float32)
    vivid = (sat > 80) & (val > 80)
    red = vivid & ((hue < 10) | (hue > 170))
    mean_bgr = crop.reshape(-1, 3).mean(axis=0)
    return {
        "red_ratio": float(red.mean()) if red.size else 0.0,
        "mean_b": float(mean_bgr[0]),
        "mean_g": float(mean_bgr[1]),
        "mean_r": float(mean_bgr[2]),
        "mean_sat": float(sat.mean()) if sat.size else 0.0,
    }


def _color_outlier_keys(entries: list[dict[str, Any]], jobs_root: Path | None) -> set[tuple[int, int]]:
    if jobs_root is None:
        return set()
    stats_by_key: dict[tuple[int, int], dict[str, float]] = {}
    for entry in entries:
        job_id = entry.get("job_id") or ""
        video_path = jobs_root / job_id / "video.mp4"
        if not video_path.exists():
            continue
        stats = _crop_color_stats(video_path, entry["frame_index"], entry["bbox"])
        if stats is not None:
            stats_by_key[entry["key"]] = stats

    remove: set[tuple[int, int]] = {
        key for key, stats in stats_by_key.items() if stats["red_ratio"] >= 0.18
    }
    for group in _group_entries(entries).values():
        group_stats = [(entry, stats_by_key.get(entry["key"])) for entry in group]
        group_stats = [(entry, stats) for entry, stats in group_stats if stats is not None]
        if len(group_stats) < 5:
            continue
        med_b = median([stats["mean_b"] for _entry, stats in group_stats])
        med_g = median([stats["mean_g"] for _entry, stats in group_stats])
        med_r = median([stats["mean_r"] for _entry, stats in group_stats])
        for entry, stats in group_stats:
            distance = (
                (stats["mean_b"] - med_b) ** 2
                + (stats["mean_g"] - med_g) ** 2
                + (stats["mean_r"] - med_r) ** 2
            ) ** 0.5
            if stats["mean_sat"] > 50 and distance > 70:
                remove.add(entry["key"])
    return remove


def _remove_detections(timeline: dict[str, Any], keys: set[tuple[int, int]]) -> tuple[int, int]:
    if not keys:
        return 0, 0
    removed_detections = 0
    removed_events = 0
    events: list[dict[str, Any]] = []
    for event_index, event in enumerate(timeline.get("events", []) or []):
        detections = event.get("detections") or []
        kept = [det for det_index, det in enumerate(detections) if (event_index, det_index) not in keys]
        removed_detections += len(detections) - len(kept)
        if kept:
            new_event = dict(event)
            new_event["detections"] = kept
            events.append(new_event)
        else:
            removed_events += 1
    timeline["events"] = events
    return removed_detections, removed_events


def compact_timeline_segments(
    path: Path,
    *,
    max_gap_sec: float = 8.0,
    merge_segments: bool = True,
    remove_position_outliers: bool = False,
    remove_size_outliers: bool = False,
    remove_tall_thin_boxes: bool = False,
    remove_static_short_tracks: bool = False,
    remove_temporal_isolated: bool = False,
    temporal_isolation_protect_tail_sec: float = 0.0,
    remove_color_outliers: bool = False,
    jobs_root: Path | None = None,
) -> dict[str, Any]:
    """Persist selected post-processing filters and display/report merge gap."""
    timeline = load_timeline(path)
    if max_gap_sec < 0:
        raise ValueError("max_gap_sec must be greater than or equal to 0")
    before_events = len(timeline.get("events", []))
    before = len(build_segments(timeline, merge_gap_sec=0))
    entries = _flatten_timeline_detections(timeline)
    remove_by_condition: dict[str, int] = {}
    remove_keys: set[tuple[int, int]] = set()
    for name, enabled, finder in (
        ("position_outlier", remove_position_outliers, lambda: _position_outlier_keys(entries)),
        ("size_outlier", remove_size_outliers, lambda: _size_outlier_keys(entries)),
        ("tall_thin_box", remove_tall_thin_boxes, lambda: _tall_thin_box_keys(entries)),
        ("static_short_track", remove_static_short_tracks, lambda: _static_short_track_keys(entries, timeline.get("videos", []))),
        (
            "temporal_isolated",
            remove_temporal_isolated,
            lambda: _temporal_isolated_keys(entries, protect_tail_sec=temporal_isolation_protect_tail_sec),
        ),
        ("color_outlier", remove_color_outliers, lambda: _color_outlier_keys(entries, jobs_root)),
    ):
        if not enabled:
            remove_by_condition[name] = 0
            continue
        keys = finder()
        remove_by_condition[name] = len(keys)
        remove_keys.update(keys)

    removed_detections, removed_events = _remove_detections(timeline, remove_keys)
    filtered_segment_count = len(build_segments(timeline, merge_gap_sec=0))
    timeline["segment_merge_gap_sec"] = float(max_gap_sec) if merge_segments else 0.0
    after = len(build_segments(timeline))
    merged_segment_count = max(0, filtered_segment_count - after)
    postprocess = {
        "applied_at": _now_iso(),
        "merge_segments": bool(merge_segments),
        "segment_merge_gap_sec": float(max_gap_sec) if merge_segments else 0.0,
        "remove_position_outliers": bool(remove_position_outliers),
        "remove_size_outliers": bool(remove_size_outliers),
        "remove_tall_thin_boxes": bool(remove_tall_thin_boxes),
        "remove_static_short_tracks": bool(remove_static_short_tracks),
        "remove_temporal_isolated": bool(remove_temporal_isolated),
        "temporal_isolation_window_sec": TEMPORAL_ISOLATION_WINDOW_SEC,
        "temporal_short_burst_max_frames": TEMPORAL_SHORT_BURST_MAX_FRAMES,
        "temporal_short_burst_max_duration_sec": TEMPORAL_SHORT_BURST_MAX_DURATION_SEC,
        "temporal_isolation_protect_tail_sec": float(temporal_isolation_protect_tail_sec),
        "remove_color_outliers": bool(remove_color_outliers),
        "before_event_count": before_events,
        "removed_detection_count": removed_detections,
        "removed_event_count": removed_events,
        "removed_by_condition": remove_by_condition,
        "before_segment_count": before,
        "filtered_segment_count": filtered_segment_count,
        "segment_count": after,
        "merged_segment_count": merged_segment_count,
    }
    timeline["postprocess"] = postprocess
    save_timeline(path, timeline)
    after_timeline = load_timeline(path)
    return {
        "event_count": len(after_timeline.get("events", [])),
        "video_count": len(after_timeline.get("videos", [])),
        "before_event_count": before_events,
        "removed_detection_count": removed_detections,
        "removed_event_count": removed_events,
        "removed_by_condition": remove_by_condition,
        "before_segment_count": before,
        "filtered_segment_count": filtered_segment_count,
        "segment_count": after,
        "merged_segment_count": merged_segment_count,
        "merge_segments": bool(merge_segments),
        "segment_merge_gap_sec": float(max_gap_sec) if merge_segments else 0.0,
        "postprocess": postprocess,
    }


def timeline_range(timeline: dict[str, Any]) -> dict[str, Any]:
    videos = timeline.get("videos", [])
    starts: list[datetime] = []
    ends: list[datetime] = []

    for video in videos:
        start = parse_video_start_time(video.get("video_name", ""))
        if start is None:
            continue
        duration = float(video.get("duration_sec") or 0)
        starts.append(start)
        ends.append(start + timedelta(seconds=duration))

    if not starts:
        return {
            "range_start": None,
            "range_end": None,
            "range_start_label": None,
            "range_end_label": None,
        }

    range_start = min(starts)
    range_end = max(ends)
    return {
        "range_start": range_start.isoformat(),
        "range_end": range_end.isoformat(),
        "range_start_label": format_absolute_time(range_start),
        "range_end_label": format_absolute_time(range_end),
    }


def get_segment_frames(path: Path, segment_id: str) -> dict[str, Any]:
    timeline = load_timeline(path)
    segments = build_segments(timeline)
    segment = next((item for item in segments if item["segment_id"] == segment_id), None)
    if segment is None:
        raise KeyError(segment_id)

    groups = build_segment_frame_groups(timeline)
    group = next(
        (item for item in groups if _segment_from_frames(item)["segment_id"] == segment_id),
        None,
    )
    if group is None:
        raise KeyError(segment_id)

    return {
        "segment": _segment_from_frames(group),
        "frames": group,
    }


def _event_from_frame(
    *,
    job_id: str,
    video_name: str,
    frame: dict[str, Any],
) -> dict[str, Any] | None:
    detections = frame.get("detections") or []
    if not detections:
        return None
    ts = float(frame.get("timestamp_sec", 0))
    video_start = parse_video_start_time(video_name)
    abs_dt = absolute_frame_time(video_name, ts)
    return {
        "job_id": job_id,
        "video_name": video_name,
        "frame_index": frame.get("frame_index"),
        "timestamp_sec": ts,
        "video_start": video_start.isoformat() if video_start else None,
        "absolute_time": abs_dt.isoformat() if abs_dt else None,
        "absolute_time_label": format_absolute_time(abs_dt),
        "preview_path": frame.get("preview_path"),
        "detections": detections,
    }


def _upsert_video_record(
    timeline: dict[str, Any],
    *,
    job_id: str,
    video_name: str,
    manifest: dict[str, Any],
    minimum_duration_sec: float = 0.0,
) -> None:
    video_start = parse_video_start_time(video_name)
    video_start_iso = video_start.isoformat() if video_start else None
    duration_sec = max(_video_duration_sec(manifest), float(minimum_duration_sec or 0.0))
    video_end = None
    if video_start is not None:
        video_end = (video_start + timedelta(seconds=duration_sec)).isoformat()

    record = {
        "job_id": job_id,
        "video_name": video_name,
        "video_start": video_start_iso,
        "video_end": video_end,
        "duration_sec": round(duration_sec, 3),
        "fps": manifest.get("fps"),
        "frame_stride": manifest.get("frame_stride"),
        "confidence": manifest.get("confidence"),
        "total_frames": manifest.get("total_frames"),
        "frames_processed": manifest.get("frames_processed", 0),
        "frames_with_detections": manifest.get("frames_with_detections", 0),
        "model": manifest.get("model"),
        "models": manifest.get("models") or [],
        "ensemble": bool(manifest.get("ensemble")),
        "dark_skip_enabled": bool(manifest.get("dark_skip_enabled")),
        "dark_video_assessment": manifest.get("dark_video_assessment"),
        "skipped": bool(manifest.get("skipped")),
        "skip_reason": manifest.get("skip_reason"),
        "added_at": _now_iso(),
    }
    videos = timeline.setdefault("videos", [])
    for index, existing in enumerate(videos):
        if existing.get("job_id") == job_id:
            previous_added = existing.get("added_at")
            if previous_added:
                record["added_at"] = previous_added
            videos[index] = record
            return
    videos.append(record)


def _sort_timeline_events(timeline: dict[str, Any]) -> None:
    timeline["events"].sort(
        key=lambda event: (
            event.get("absolute_time") or "",
            event.get("video_name") or "",
            event.get("frame_index") or 0,
        )
    )


def merge_frame_detection(
    path: Path,
    *,
    job_id: str,
    video_name: str,
    frame: dict[str, Any],
    manifest: dict[str, Any] | None = None,
) -> int:
    """Merge one detected frame immediately; return 1 when it adds a new event."""
    event = _event_from_frame(job_id=job_id, video_name=video_name, frame=frame)
    if event is None:
        return 0

    timeline = load_timeline(path)
    frame_index = event.get("frame_index")
    existing_count = len(timeline.get("events", []))
    timeline["events"] = [
        item
        for item in timeline.get("events", [])
        if not (item.get("job_id") == job_id and item.get("frame_index") == frame_index)
    ]
    removed_duplicate = len(timeline["events"]) != existing_count
    timeline["events"].append(event)
    _upsert_video_record(
        timeline,
        job_id=job_id,
        video_name=video_name,
        manifest=manifest or {},
        minimum_duration_sec=float(event.get("timestamp_sec") or 0.0),
    )
    _sort_timeline_events(timeline)
    save_timeline(path, timeline)
    return 0 if removed_duplicate else 1


def merge_job_manifest(
    path: Path,
    *,
    job_id: str,
    video_name: str,
    manifest: dict[str, Any],
    replace_job: bool = False,
    replace_video: bool = False,
) -> int:
    """Append detection events from a completed job; return new event count."""
    timeline = load_timeline(path)
    if replace_job or replace_video:
        timeline["events"] = [
            event
            for event in timeline.get("events", [])
            if not (
                (replace_job and event.get("job_id") == job_id)
                or (replace_video and event.get("video_name") == video_name)
            )
        ]
        timeline["videos"] = [
            video
            for video in timeline.get("videos", [])
            if not (
                (replace_job and video.get("job_id") == job_id)
                or (replace_video and video.get("video_name") == video_name)
            )
        ]

    added = 0
    existing_keys = {
        (event.get("job_id"), event.get("frame_index"))
        for event in timeline.get("events", [])
    }
    for frame in manifest.get("frames", []):
        event = _event_from_frame(job_id=job_id, video_name=video_name, frame=frame)
        if event is None:
            continue
        key = (event.get("job_id"), event.get("frame_index"))
        if key in existing_keys:
            timeline["events"] = [
                item
                for item in timeline.get("events", [])
                if (item.get("job_id"), item.get("frame_index")) != key
            ]
        timeline["events"].append(event)
        existing_keys.add(key)
        added += 1

    _upsert_video_record(
        timeline,
        job_id=job_id,
        video_name=video_name,
        manifest=manifest,
    )

    _sort_timeline_events(timeline)
    save_timeline(path, timeline)
    return added


def list_timeline(
    path: Path,
    *,
    offset: int = 0,
    limit: int = 60,
) -> dict[str, Any]:
    timeline = load_timeline(path)
    segments = build_segments(timeline)
    total = len(segments)
    page = segments[offset : offset + limit]
    summary = timeline_range(timeline)
    axis_segments = [
        {
            "segment_id": segment["segment_id"],
            "job_id": segment["job_id"],
            "class_name": segment["class_name"],
            "start_absolute_time": segment["start_absolute_time"],
            "end_absolute_time": segment["end_absolute_time"],
            "time_label": segment["time_label"],
            "preview_path": segment["preview_path"],
            "preview_job_id": segment["preview_job_id"],
            "frame_count": segment["frame_count"],
            "max_confidence": segment.get("max_confidence"),
            "max_area_px": segment.get("max_area_px"),
            "bbox_xyxy": segment.get("bbox_xyxy"),
            "polygon_xy": segment.get("polygon_xy"),
            "polygon_point_count": segment.get("polygon_point_count"),
            "mask_area_px": segment.get("mask_area_px"),
            "mask_width_px": segment.get("mask_width_px"),
            "mask_height_px": segment.get("mask_height_px"),
        }
        for segment in segments
    ]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "video_count": len(timeline.get("videos", [])),
        "event_count": len(timeline.get("events", [])),
        "segment_merge_gap_sec": _timeline_merge_gap_sec(timeline),
        "segments": page,
        "axis_segments": axis_segments,
        "videos": timeline.get("videos", []),
        **summary,
    }


def timeline_summary(path: Path) -> dict[str, Any]:
    timeline = load_timeline(path)
    segments = build_segments(timeline)
    return {
        "event_count": len(timeline.get("events", [])),
        "segment_count": len(segments),
        "video_count": len(timeline.get("videos", [])),
        "segment_merge_gap_sec": _timeline_merge_gap_sec(timeline),
    }
