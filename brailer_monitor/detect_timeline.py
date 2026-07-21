"""Accumulated detection timeline across multiple video jobs."""

from __future__ import annotations

import json
import os
import re
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
TEMPORAL_WORK_WINDOW_SEC = 12.0 * 60.0 * 60.0
TEMPORAL_WORK_POSITION_MIN_PX = 40.0
TEMPORAL_WORK_POSITION_DIAG_RATIO = 0.75
WORK_REPEATED_PROTECTED_CONDITIONS = frozenset(
    {"position_outlier", "size_outlier", "temporal_isolated", "color_outlier"}
)
LARGE_LOWER_SEA_MIN_GROUP_SIZE = 10
LARGE_LOWER_SEA_AREA_MEDIAN_RATIO = 4.0
LARGE_LOWER_SEA_BOTTOM_Y_RATIO = 0.98
LARGE_LOWER_SEA_CENTER_Y_RATIO = 0.65
POSITION_WORK_MIN_GROUP_SIZE = 10
POSITION_LOWER_ROI_Y_RATIO = 0.70
POSITION_ROI_EDGE_MARGIN_RATIO = 0.02
POSITION_ROI_EDGE_MIN_VIDEO_COUNT = 3
STATIC_POSITION_MAX_GAP_SEC = 8.0
STATIC_POSITION_MIN_DURATION_SEC = 3.0
STATIC_POSITION_MIN_FRAMES = 6
STATIC_POSITION_MAX_CENTER_MOVE_PX = 30.0
STATIC_POSITION_MAX_CENTER_MOVE_DIAG_RATIO = 0.2
STATIC_POSITION_STRICT_MIN_FRAMES = 3
STATIC_POSITION_STRICT_MIN_DURATION_SEC = 0.5
STATIC_POSITION_STRICT_MAX_CENTER_MOVE_PX = 2.0
STATIC_POSITION_STRICT_MAX_SIZE_CHANGE_RATIO = 0.08
EDGE_DEFAULT_FRAME_WIDTH = 1280.0
EDGE_CENTER_X_RATIO = 0.85
EDGE_SIDE_X_RATIO = 0.985
_VIDEO_SOURCE_SUFFIX_RE = re.compile(r"_\d{6}_\d{6}$")


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


def _datetime_timestamp(value: datetime | None) -> float | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def _video_source_key(video_name: str) -> str:
    stem = Path(video_name.rsplit("/", 1)[-1]).stem
    return _VIDEO_SOURCE_SUFFIX_RE.sub("", stem)


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
    for key in ("sea_ratio", "sea_percent", "sea_area_px", "sea_method"):
        value = event.get(key)
        if value is not None:
            out[key] = value
    detect_roi = event.get("detect_roi")
    if detect_roi is not None:
        out["detect_roi"] = detect_roi
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
    sea_ratios: list[float] = []
    for frame in frames:
        try:
            sea_ratio = float(frame.get("sea_ratio"))
        except (TypeError, ValueError):
            continue
        if 0.0 <= sea_ratio <= 1.0:
            sea_ratios.append(sea_ratio)
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
    if sea_ratios:
        avg_sea_ratio = sum(sea_ratios) / len(sea_ratios)
        out["sea_ratio"] = round(avg_sea_ratio, 4)
        out["avg_sea_ratio"] = round(avg_sea_ratio, 4)
        out["min_sea_ratio"] = round(min(sea_ratios), 4)
        out["max_sea_ratio"] = round(max(sea_ratios), 4)
        out["avg_sea_percent"] = round(avg_sea_ratio * 100.0, 2)
    detect_roi = first.get("detect_roi")
    if detect_roi is not None:
        out["detect_roi"] = detect_roi
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
    video_widths: dict[str, float] = {}
    video_heights: dict[str, float] = {}
    for video in timeline.get("videos", []) or []:
        try:
            width = float(video.get("width") or 0.0)
            height = float(video.get("height") or 0.0)
        except (TypeError, ValueError):
            width = 0.0
            height = 0.0
        video_name = str(video.get("video_name") or "")
        if width > 0:
            video_widths[video_name] = width
        if height > 0:
            video_heights[video_name] = height
    for event_index, event in enumerate(timeline.get("events", []) or []):
        video_name = str(event.get("video_name") or "")
        absolute_dt = _parse_iso_datetime(event.get("absolute_time"))
        if absolute_dt is None:
            absolute_dt = absolute_frame_time(video_name, float(event.get("timestamp_sec") or 0.0))
        try:
            frame_width = float(event.get("width") or video_widths.get(video_name) or EDGE_DEFAULT_FRAME_WIDTH)
            frame_height = float(event.get("height") or video_heights.get(video_name) or 720.0)
        except (TypeError, ValueError):
            frame_width = EDGE_DEFAULT_FRAME_WIDTH
            frame_height = 720.0
        for det_index, det in enumerate(event.get("detections") or []):
            bbox = _bbox_values(det)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            area = detection_area_px(det)
            if area <= 0:
                area = int((x2 - x1) * (y2 - y1))
            class_name = str(det.get("class_name") or "")
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
                    "frame_width": frame_width,
                    "frame_height": frame_height,
                    "group": (video_name, class_name),
                    "work_group": (_video_source_key(video_name), class_name),
                    "absolute_timestamp": _datetime_timestamp(absolute_dt),
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


def _position_outlier_keys_for_group(
    group: list[dict[str, Any]],
    *,
    min_group_size: int,
) -> set[tuple[int, int]]:
    if len(group) < min_group_size:
        return set()
    xs = [entry["center"][0] for entry in group]
    ys = [entry["center"][1] for entry in group]
    mx = median(xs)
    my = median(ys)
    distances = [((entry["center"][0] - mx) ** 2 + (entry["center"][1] - my) ** 2) ** 0.5 for entry in group]
    med_dist = median(distances)
    mad = median([abs(distance - med_dist) for distance in distances])
    med_diag = median([entry["diag"] for entry in group])
    threshold = med_dist + max(3.0 * mad, 1.5 * med_diag, 80.0)
    return {
        entry["key"]
        for entry, distance in zip(group, distances)
        if distance > threshold
    }


def _position_outlier_keys(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    for group in _group_entries(entries).values():
        remove.update(_position_outlier_keys_for_group(group, min_group_size=5))
    return remove


def _is_lower_roi_boundary_detection(entry: dict[str, Any]) -> bool:
    detect_roi = entry["event"].get("detect_roi") or {}
    xyxy = detect_roi.get("xyxy_px") or []
    if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
        return False
    try:
        roi_x1 = float(xyxy[0])
        roi_x2 = float(xyxy[2])
        frame_width = float(entry.get("frame_width") or 0.0)
        frame_height = float(entry.get("frame_height") or 0.0)
    except (TypeError, ValueError):
        return False
    if roi_x2 <= roi_x1 or frame_width <= 0 or frame_height <= 0:
        return False
    center_x, center_y = entry["center"]
    edge_margin = frame_width * POSITION_ROI_EDGE_MARGIN_RATIO
    return (
        float(center_y) >= frame_height * POSITION_LOWER_ROI_Y_RATIO
        and min(abs(float(center_x) - roi_x1), abs(float(center_x) - roi_x2)) <= edge_margin
    )


def _repeated_lower_roi_boundary_position_outlier_keys(
    entries: list[dict[str, Any]],
) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["work_group"]].append(entry)

    for group in groups.values():
        work_outliers = _position_outlier_keys_for_group(
            group,
            min_group_size=POSITION_WORK_MIN_GROUP_SIZE,
        )
        candidates = [
            entry
            for entry in group
            if entry["key"] in work_outliers and _is_lower_roi_boundary_detection(entry)
        ]
        for entry in candidates:
            matching_videos = {
                str(other["event"].get("video_name") or "")
                for other in candidates
                if _similar_work_detection(entry, other)
            }
            matching_videos.discard("")
            if len(matching_videos) >= POSITION_ROI_EDGE_MIN_VIDEO_COUNT:
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


def _large_lower_sea_region_keys(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        groups[entry["work_group"]].append(entry)

    for group in groups.values():
        areas = [float(entry["area"]) for entry in group if float(entry["area"]) > 0]
        if len(areas) < LARGE_LOWER_SEA_MIN_GROUP_SIZE:
            continue
        med_area = median(areas)
        if med_area <= 0:
            continue
        for entry in group:
            frame_height = float(entry.get("frame_height") or 720.0)
            if frame_height <= 0:
                frame_height = 720.0
            if (
                float(entry["area"]) > med_area * LARGE_LOWER_SEA_AREA_MEDIAN_RATIO
                and float(entry["bbox"][3]) >= frame_height * LARGE_LOWER_SEA_BOTTOM_Y_RATIO
                and float(entry["center"][1]) >= frame_height * LARGE_LOWER_SEA_CENTER_Y_RATIO
            ):
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


def _right_edge_keys(entries: list[dict[str, Any]]) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    for entry in entries:
        frame_width = float(entry.get("frame_width") or EDGE_DEFAULT_FRAME_WIDTH)
        if frame_width <= 0:
            frame_width = EDGE_DEFAULT_FRAME_WIDTH
        center_x = float(entry["center"][0])
        left_x = float(entry["bbox"][0])
        right_x = float(entry["bbox"][2])
        left_center_limit = frame_width * (1.0 - EDGE_CENTER_X_RATIO)
        right_center_limit = frame_width * EDGE_CENTER_X_RATIO
        left_edge_limit = frame_width * (1.0 - EDGE_SIDE_X_RATIO)
        right_edge_limit = frame_width * EDGE_SIDE_X_RATIO
        if (
            center_x <= left_center_limit
            or center_x >= right_center_limit
            or left_x <= left_edge_limit
            or right_x >= right_edge_limit
        ):
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
        expected_step = max(0.001, sample_steps.get(video_name, 1.0))
        max_time_gap = max(STATIC_POSITION_MAX_GAP_SEC, expected_step * 1.75)
        for entry in ordered:
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
            median_x = median([entry["center"][0] for entry in run])
            median_y = median([entry["center"][1] for entry in run])
            max_move = max(
                ((entry["center"][0] - median_x) ** 2 + (entry["center"][1] - median_y) ** 2) ** 0.5
                for entry in run
            )
            med_diag = median([entry["diag"] for entry in run])
            motion_threshold = max(
                STATIC_POSITION_MAX_CENTER_MOVE_PX,
                med_diag * STATIC_POSITION_MAX_CENTER_MOVE_DIAG_RATIO,
            )
            normal_static = (
                duration_sec >= STATIC_POSITION_MIN_DURATION_SEC
                or len(run) >= STATIC_POSITION_MIN_FRAMES
            ) and max_move <= motion_threshold
            max_size_change_ratio = (
                max(abs(float(entry["diag"]) - med_diag) / med_diag for entry in run)
                if med_diag > 0
                else float("inf")
            )
            strict_static = (
                len(run) >= STATIC_POSITION_STRICT_MIN_FRAMES
                and duration_sec >= STATIC_POSITION_STRICT_MIN_DURATION_SEC
                and max_move <= STATIC_POSITION_STRICT_MAX_CENTER_MOVE_PX
                and max_size_change_ratio <= STATIC_POSITION_STRICT_MAX_SIZE_CHANGE_RATIO
            )
            if normal_static or strict_static:
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
    return True


def _similar_work_detection(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if not _similar_temporal_detection(a, b):
        return False
    center_a = a["center"]
    center_b = b["center"]
    center_distance = (
        (float(center_a[0]) - float(center_b[0])) ** 2
        + (float(center_a[1]) - float(center_b[1])) ** 2
    ) ** 0.5
    smaller_diag = min(float(a.get("diag") or 0.0), float(b.get("diag") or 0.0))
    position_threshold = max(
        TEMPORAL_WORK_POSITION_MIN_PX,
        smaller_diag * TEMPORAL_WORK_POSITION_DIAG_RATIO,
    )
    return center_distance <= position_threshold


def _work_repeated_detection_keys(
    entries: list[dict[str, Any]],
    candidate_keys: set[tuple[int, int]],
    *,
    work_window_sec: float = TEMPORAL_WORK_WINDOW_SEC,
    local_window_sec: float = TEMPORAL_ISOLATION_WINDOW_SEC,
) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    if not candidate_keys or work_window_sec <= local_window_sec:
        return protected

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry.get("absolute_timestamp") is not None:
            groups[entry["work_group"]].append(entry)

    for group in groups.values():
        candidates = [entry for entry in group if entry["key"] in candidate_keys]
        for entry in candidates:
            entry_time = float(entry["absolute_timestamp"])
            for other in group:
                if other["key"] == entry["key"]:
                    continue
                time_gap = abs(float(other["absolute_timestamp"]) - entry_time)
                if time_gap <= local_window_sec or time_gap > work_window_sec:
                    continue
                if _similar_work_detection(entry, other):
                    protected.add(entry["key"])
                    break
    return protected


def _temporal_merge_protected_keys(
    entries: list[dict[str, Any]],
    *,
    merge_gap_sec: float,
) -> set[tuple[int, int]]:
    protected: set[tuple[int, int]] = set()
    gap_limit = max(0.0, float(merge_gap_sec))
    if gap_limit <= 0:
        return protected

    for group in _group_entries(entries).values():
        ordered = sorted(group, key=lambda entry: (entry["timestamp_sec"], entry["frame_index"]))
        if len(ordered) < 2:
            continue
        runs: list[list[dict[str, Any]]] = [[ordered[0]]]
        for entry in ordered[1:]:
            prev = runs[-1][-1]
            time_gap = float(entry["timestamp_sec"]) - float(prev["timestamp_sec"])
            if 0.0 <= time_gap <= gap_limit:
                runs[-1].append(entry)
            else:
                runs.append([entry])

        for run in runs:
            if len(run) >= 2:
                protected.update(entry["key"] for entry in run)
    return protected


def _temporal_isolated_keys(
    entries: list[dict[str, Any]],
    *,
    window_sec: float = TEMPORAL_ISOLATION_WINDOW_SEC,
    protect_tail_sec: float = 0.0,
    merge_protect_gap_sec: float = 0.0,
    protect_repeated_work: bool = True,
) -> set[tuple[int, int]]:
    remove: set[tuple[int, int]] = set()
    window = max(0.0, float(window_sec))
    if window <= 0:
        return remove
    protected_by_merge = _temporal_merge_protected_keys(
        entries,
        merge_gap_sec=merge_protect_gap_sec,
    )

    for group in _group_entries(entries).values():
        latest_timestamp = max((float(entry["timestamp_sec"]) for entry in group), default=0.0)
        def can_remove(entry: dict[str, Any]) -> bool:
            if entry["key"] in protected_by_merge:
                return False
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

    if protect_repeated_work:
        remove.difference_update(
            _work_repeated_detection_keys(
                entries,
                remove,
                work_window_sec=TEMPORAL_WORK_WINDOW_SEC,
                local_window_sec=window,
            )
        )
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
    remove_large_lower_sea_regions: bool = False,
    remove_tall_thin_boxes: bool = False,
    remove_right_edge_detections: bool = False,
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
    work_repeated_keys = _work_repeated_detection_keys(
        entries,
        {entry["key"] for entry in entries},
    )
    repeated_lower_roi_boundary_keys = (
        _repeated_lower_roi_boundary_position_outlier_keys(entries)
        if remove_position_outliers
        else set()
    )
    position_outlier_keys = (
        _position_outlier_keys(entries) | repeated_lower_roi_boundary_keys
        if remove_position_outliers
        else set()
    )
    remove_by_condition: dict[str, int] = {}
    protected_by_condition: dict[str, int] = {}
    protected_keys: set[tuple[int, int]] = set()
    remove_keys: set[tuple[int, int]] = set()
    for name, enabled, finder in (
        (
            "large_lower_sea_region",
            remove_large_lower_sea_regions,
            lambda: _large_lower_sea_region_keys(entries),
        ),
        ("position_outlier", remove_position_outliers, lambda: set(position_outlier_keys)),
        ("right_edge", remove_right_edge_detections, lambda: _right_edge_keys(entries)),
        ("size_outlier", remove_size_outliers, lambda: _size_outlier_keys(entries)),
        ("tall_thin_box", remove_tall_thin_boxes, lambda: _tall_thin_box_keys(entries)),
        ("static_short_track", remove_static_short_tracks, lambda: _static_short_track_keys(entries, timeline.get("videos", []))),
        (
            "temporal_isolated",
            remove_temporal_isolated,
            lambda: _temporal_isolated_keys(
                entries,
                protect_tail_sec=temporal_isolation_protect_tail_sec,
                merge_protect_gap_sec=float(max_gap_sec) if merge_segments else 0.0,
                protect_repeated_work=False,
            ),
        ),
        ("color_outlier", remove_color_outliers, lambda: _color_outlier_keys(entries, jobs_root)),
    ):
        if not enabled:
            remove_by_condition[name] = 0
            protected_by_condition[name] = 0
            continue
        keys = finder()
        protectable_keys = (
            keys - repeated_lower_roi_boundary_keys
            if name == "position_outlier"
            else keys
        )
        condition_protected = (
            protectable_keys & work_repeated_keys
            if name in WORK_REPEATED_PROTECTED_CONDITIONS
            else set()
        )
        protected_by_condition[name] = len(condition_protected)
        protected_keys.update(condition_protected)
        keys.difference_update(condition_protected)
        remove_by_condition[name] = len(keys - remove_keys)
        remove_keys.update(keys)

    removed_detections, removed_events = _remove_detections(timeline, remove_keys)
    retained_protected_keys = protected_keys - remove_keys
    filtered_segment_count = len(build_segments(timeline, merge_gap_sec=0))
    timeline["segment_merge_gap_sec"] = float(max_gap_sec) if merge_segments else 0.0
    after = len(build_segments(timeline))
    merged_segment_count = max(0, filtered_segment_count - after)
    postprocess = {
        "applied_at": _now_iso(),
        "merge_segments": bool(merge_segments),
        "segment_merge_gap_sec": float(max_gap_sec) if merge_segments else 0.0,
        "remove_position_outliers": bool(remove_position_outliers),
        "position_work_min_group_size": POSITION_WORK_MIN_GROUP_SIZE,
        "position_lower_roi_y_ratio": POSITION_LOWER_ROI_Y_RATIO,
        "position_roi_edge_margin_ratio": POSITION_ROI_EDGE_MARGIN_RATIO,
        "position_roi_edge_min_video_count": POSITION_ROI_EDGE_MIN_VIDEO_COUNT,
        "position_lower_roi_boundary_candidate_count": len(repeated_lower_roi_boundary_keys),
        "remove_size_outliers": bool(remove_size_outliers),
        "remove_large_lower_sea_regions": bool(remove_large_lower_sea_regions),
        "large_lower_sea_min_group_size": LARGE_LOWER_SEA_MIN_GROUP_SIZE,
        "large_lower_sea_area_median_ratio": LARGE_LOWER_SEA_AREA_MEDIAN_RATIO,
        "large_lower_sea_bottom_y_ratio": LARGE_LOWER_SEA_BOTTOM_Y_RATIO,
        "large_lower_sea_center_y_ratio": LARGE_LOWER_SEA_CENTER_Y_RATIO,
        "remove_tall_thin_boxes": bool(remove_tall_thin_boxes),
        "remove_right_edge_detections": bool(remove_right_edge_detections),
        "right_edge_center_x_ratio": EDGE_CENTER_X_RATIO,
        "right_edge_right_x_ratio": EDGE_SIDE_X_RATIO,
        "right_edge_default_frame_width": EDGE_DEFAULT_FRAME_WIDTH,
        "edge_center_x_ratio": EDGE_CENTER_X_RATIO,
        "edge_side_x_ratio": EDGE_SIDE_X_RATIO,
        "edge_default_frame_width": EDGE_DEFAULT_FRAME_WIDTH,
        "remove_static_short_tracks": bool(remove_static_short_tracks),
        "static_position_max_gap_sec": STATIC_POSITION_MAX_GAP_SEC,
        "static_position_min_duration_sec": STATIC_POSITION_MIN_DURATION_SEC,
        "static_position_min_frames": STATIC_POSITION_MIN_FRAMES,
        "static_position_max_center_move_px": STATIC_POSITION_MAX_CENTER_MOVE_PX,
        "static_position_max_center_move_diag_ratio": STATIC_POSITION_MAX_CENTER_MOVE_DIAG_RATIO,
        "static_position_strict_min_frames": STATIC_POSITION_STRICT_MIN_FRAMES,
        "static_position_strict_min_duration_sec": STATIC_POSITION_STRICT_MIN_DURATION_SEC,
        "static_position_strict_max_center_move_px": STATIC_POSITION_STRICT_MAX_CENTER_MOVE_PX,
        "static_position_strict_max_size_change_ratio": STATIC_POSITION_STRICT_MAX_SIZE_CHANGE_RATIO,
        "remove_temporal_isolated": bool(remove_temporal_isolated),
        "temporal_isolation_window_sec": TEMPORAL_ISOLATION_WINDOW_SEC,
        "temporal_short_burst_max_frames": TEMPORAL_SHORT_BURST_MAX_FRAMES,
        "temporal_short_burst_max_duration_sec": TEMPORAL_SHORT_BURST_MAX_DURATION_SEC,
        "temporal_work_window_sec": TEMPORAL_WORK_WINDOW_SEC,
        "temporal_work_position_min_px": TEMPORAL_WORK_POSITION_MIN_PX,
        "temporal_work_position_diag_ratio": TEMPORAL_WORK_POSITION_DIAG_RATIO,
        "work_repeated_detection_count": len(work_repeated_keys),
        "protected_candidate_count": len(protected_keys),
        "protected_detection_count": len(retained_protected_keys),
        "protected_by_condition": protected_by_condition,
        "temporal_isolation_protect_tail_sec": float(temporal_isolation_protect_tail_sec),
        "temporal_merge_protect_gap_sec": float(max_gap_sec) if merge_segments else 0.0,
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
        "protected_candidate_count": len(protected_keys),
        "protected_detection_count": len(retained_protected_keys),
        "protected_by_condition": protected_by_condition,
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
    event = {
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
    for key in ("sea_ratio", "sea_percent", "sea_area_px", "sea_method"):
        value = frame.get(key)
        if value is not None:
            event[key] = value
    detect_roi = frame.get("detect_roi")
    if detect_roi is not None:
        event["detect_roi"] = detect_roi
    for key in ("width", "height"):
        value = frame.get(key)
        if value is not None:
            event[key] = value
    return event


def _sea_frame_time(video_name: str, frame: dict[str, Any]) -> dict[str, Any]:
    timestamp = float(frame.get("timestamp_sec") or 0.0)
    absolute = absolute_frame_time(video_name, timestamp)
    return {
        "timestamp_sec": round(timestamp, 3),
        "absolute_time": absolute.isoformat() if absolute is not None else None,
        "time_label": format_absolute_time(absolute) or f"{timestamp:.3f}s",
    }


def _sea_analysis_from_manifest(video_name: str, manifest: dict[str, Any]) -> dict[str, Any]:
    frames = [
        frame
        for frame in manifest.get("frames", [])
        if isinstance(frame, dict)
        and any(
            frame.get(key) is not None
            for key in ("sea_quality", "sea_state", "sea_ratio", "sea_method")
        )
    ]
    enabled = bool(manifest.get("sea_ratio_enabled")) or bool(frames)
    if not enabled:
        return {"enabled": False, "sample_count": 0, "encounter_segments": [], "events": []}

    state_counts: dict[str, int] = defaultdict(int)
    methods: list[str] = []
    ratios: list[float] = []
    confidences: list[float] = []
    vessel_ratios: list[float] = []
    events: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None

    def close_segment(end_frame: dict[str, Any]) -> None:
        nonlocal current
        if current is None:
            return
        end_time = _sea_frame_time(video_name, end_frame)
        current.update(
            {
                "end_timestamp_sec": end_time["timestamp_sec"],
                "end_absolute_time": end_time["absolute_time"],
                "end_time": end_time["time_label"],
                "duration_sec": round(
                    max(0.0, float(end_time["timestamp_sec"]) - float(current["start_timestamp_sec"])),
                    3,
                ),
            }
        )
        segments.append(current)
        current = None

    last_encounter_frame: dict[str, Any] | None = None
    for frame in frames:
        state = str(frame.get("sea_state") or "unknown")
        quality = str(frame.get("sea_quality") or "unknown")
        if quality == "unknown":
            state = "unknown"
        state_counts[state] += 1

        method = str(frame.get("sea_method") or "").strip()
        if method and method not in methods:
            methods.append(method)
        if state != "unknown":
            for key, output in (
                ("sea_ratio", ratios),
                ("sea_confidence", confidences),
                ("vessel_ratio", vessel_ratios),
            ):
                try:
                    value = float(frame.get(key))
                except (TypeError, ValueError):
                    continue
                output.append(value)

        event_name = str(frame.get("sea_event") or "").strip()
        if event_name:
            events.append({"event": event_name, **_sea_frame_time(video_name, frame)})

        if state == "encounter":
            frame_time = _sea_frame_time(video_name, frame)
            if current is None:
                current = {
                    "start_timestamp_sec": frame_time["timestamp_sec"],
                    "start_absolute_time": frame_time["absolute_time"],
                    "start_time": frame_time["time_label"],
                    "sample_count": 0,
                    "min_sea_ratio": None,
                    "max_vessel_ratio": None,
                }
            current["sample_count"] += 1
            ratio = frame.get("sea_ratio")
            vessel = frame.get("vessel_ratio")
            if ratio is not None:
                ratio = float(ratio)
                previous = current.get("min_sea_ratio")
                current["min_sea_ratio"] = round(ratio if previous is None else min(previous, ratio), 4)
            if vessel is not None:
                vessel = float(vessel)
                previous = current.get("max_vessel_ratio")
                current["max_vessel_ratio"] = round(vessel if previous is None else max(previous, vessel), 4)
            last_encounter_frame = frame
        elif state != "unknown" and current is not None:
            close_segment(frame if event_name == "departure" else (last_encounter_frame or frame))

    if current is not None:
        close_segment(last_encounter_frame or frames[-1])

    unknown_count = int(state_counts.get("unknown", 0))
    sample_count = len(frames)
    valid_count = sample_count - unknown_count
    return {
        "enabled": True,
        "engine": manifest.get("sea_engine"),
        "methods": methods,
        "sample_count": sample_count,
        "valid_count": valid_count,
        "unknown_count": unknown_count,
        "unknown_fraction": round(unknown_count / sample_count, 4) if sample_count else 0.0,
        "state_counts": dict(state_counts),
        "avg_sea_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "min_sea_ratio": round(min(ratios), 4) if ratios else None,
        "max_sea_ratio": round(max(ratios), 4) if ratios else None,
        "avg_confidence": round(sum(confidences) / len(confidences), 4) if confidences else None,
        "max_vessel_ratio": round(max(vessel_ratios), 4) if vessel_ratios else None,
        "events": events,
        "encounter_segments": segments,
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
        "width": manifest.get("width"),
        "height": manifest.get("height"),
        "total_frames": manifest.get("total_frames"),
        "frames_processed": manifest.get("frames_processed", 0),
        "frames_with_detections": manifest.get("frames_with_detections", 0),
        "model": manifest.get("model"),
        "models": manifest.get("models") or [],
        "ensemble": bool(manifest.get("ensemble")),
        "object_detection_enabled": bool(manifest.get("object_detection_enabled", True)),
        "sea_only": bool(manifest.get("sea_only")),
        "detect_roi": manifest.get("detect_roi"),
        "dark_skip_enabled": bool(manifest.get("dark_skip_enabled")),
        "dark_video_assessment": manifest.get("dark_video_assessment"),
        "sea_ratio_summary": manifest.get("sea_ratio_summary"),
        "sea_analysis_interval_sec": manifest.get("sea_analysis_interval_sec"),
        "sea_analysis": _sea_analysis_from_manifest(video_name, manifest),
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
