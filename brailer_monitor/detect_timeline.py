"""Accumulated detection timeline across multiple video jobs."""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .video_time import absolute_frame_time, format_absolute_time, parse_video_start_time


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_timeline() -> dict[str, Any]:
    return {"updated_at": _now_iso(), "videos": [], "events": []}


def load_timeline(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_timeline()
    return json.loads(path.read_text(encoding="utf-8"))


def save_timeline(path: Path, timeline: dict[str, Any]) -> None:
    timeline["updated_at"] = _now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(timeline, indent=2, ensure_ascii=False), encoding="utf-8")


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
            expanded.append(
                _frame_to_dict(
                    event,
                    class_name=cls,
                    confidence=float(det.get("confidence", 0)),
                    area_px=detection_area_px(det),
                    bbox_xyxy=list(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
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


def _segment_from_frames(frames: list[dict[str, Any]]) -> dict[str, Any]:
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
    segment_id = f"{first['job_id']}:{first['class_name']}:{first.get('frame_index', 0)}"

    out: dict[str, Any] = {
        "segment_id": segment_id,
        "job_id": first["job_id"],
        "video_name": first["video_name"],
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
    return out


def build_segments(timeline: dict[str, Any]) -> list[dict[str, Any]]:
    videos_by_name = {video["video_name"]: video for video in timeline.get("videos", [])}
    class_frames: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for frame in expand_class_events(timeline.get("events", [])):
        key = (frame["video_name"], frame["class_name"])
        class_frames[key].append(frame)

    segments: list[dict[str, Any]] = []
    for (video_name, _class_name), frames in class_frames.items():
        stride = int(videos_by_name.get(video_name, {}).get("frame_stride") or 5)
        for group in merge_consecutive_frames(frames, frame_stride=stride):
            segments.append(_segment_from_frames(group))

    segments.sort(
        key=lambda segment: (
            segment.get("start_absolute_time") or "",
            segment.get("video_name") or "",
            segment.get("start_timestamp_sec") or 0,
        )
    )
    return segments


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
    videos_by_name = {video["video_name"]: video for video in timeline.get("videos", [])}
    segments = build_segments(timeline)
    segment = next((item for item in segments if item["segment_id"] == segment_id), None)
    if segment is None:
        raise KeyError(segment_id)

    video_name = segment["video_name"]
    class_name = segment["class_name"]
    stride = int(videos_by_name.get(video_name, {}).get("frame_stride") or 5)
    frames = [
        frame
        for frame in expand_class_events(timeline.get("events", []))
        if frame["video_name"] == video_name and frame["class_name"] == class_name
    ]
    groups = merge_consecutive_frames(frames, frame_stride=stride)
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


def merge_job_manifest(
    path: Path,
    *,
    job_id: str,
    video_name: str,
    manifest: dict[str, Any],
) -> int:
    """Append detection events from a completed job; return new event count."""
    timeline = load_timeline(path)
    video_start = parse_video_start_time(video_name)
    video_start_iso = video_start.isoformat() if video_start else None
    duration_sec = _video_duration_sec(manifest)

    added = 0
    for frame in manifest.get("frames", []):
        detections = frame.get("detections") or []
        if not detections:
            continue
        ts = float(frame.get("timestamp_sec", 0))
        abs_dt = absolute_frame_time(video_name, ts)
        timeline["events"].append(
            {
                "job_id": job_id,
                "video_name": video_name,
                "frame_index": frame.get("frame_index"),
                "timestamp_sec": ts,
                "video_start": video_start_iso,
                "absolute_time": abs_dt.isoformat() if abs_dt else None,
                "absolute_time_label": format_absolute_time(abs_dt),
                "preview_path": frame.get("preview_path"),
                "detections": detections,
            }
        )
        added += 1

    video_end = None
    if video_start is not None:
        video_end = (video_start + timedelta(seconds=duration_sec)).isoformat()

    timeline["videos"].append(
        {
            "job_id": job_id,
            "video_name": video_name,
            "video_start": video_start_iso,
            "video_end": video_end,
            "duration_sec": round(duration_sec, 3),
            "fps": manifest.get("fps"),
            "frame_stride": manifest.get("frame_stride"),
            "total_frames": manifest.get("total_frames"),
            "frames_processed": manifest.get("frames_processed", 0),
            "frames_with_detections": manifest.get("frames_with_detections", 0),
            "added_at": _now_iso(),
        }
    )

    timeline["events"].sort(
        key=lambda event: (
            event.get("absolute_time") or "",
            event.get("video_name") or "",
            event.get("frame_index") or 0,
        )
    )
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
        }
        for segment in segments
    ]
    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "video_count": len(timeline.get("videos", [])),
        "event_count": len(timeline.get("events", [])),
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
    }
