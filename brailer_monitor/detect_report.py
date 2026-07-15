"""Generate external reports from accumulated detection timeline results."""

from __future__ import annotations

import csv
import html
import json
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .detect_timeline import build_segment_frame_groups, load_timeline


@dataclass(frozen=True)
class DetectionSegmentReportRow:
    start_time: str
    end_time: str
    duration_sec: float
    sample_window_sec: float
    frame_count: int
    class_name: str
    best_match_pct: float
    best_confidence: float
    best_time: str
    best_segment_frame_number: int
    best_frame_index: int | None
    best_timestamp_sec: float | None
    bbox_x1: float | None
    bbox_y1: float | None
    bbox_x2: float | None
    bbox_y2: float | None
    area_px: int
    mask_area_px: int
    mask_width_px: int
    mask_height_px: int
    polygon_point_count: int
    avg_mask_area_px: float
    max_mask_area_px: int
    avg_mask_width_px: float
    max_mask_width_px: int
    avg_mask_height_px: float
    max_mask_height_px: int
    best_sea_ratio: float | None
    avg_sea_ratio: float | None
    min_sea_ratio: float | None
    max_sea_ratio: float | None
    video_name: str
    job_id: str
    preview_path: str | None
    preview_url: str | None
    video_url: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionTimelineFrame:
    time: str
    absolute_time: str
    class_name: str
    confidence: float
    match_pct: float
    video_name: str
    job_id: str
    frame_index: int | None
    timestamp_sec: float | None
    sea_ratio: float | None
    sea_percent: float | None
    preview_path: str | None
    preview_url: str | None
    is_representative: bool
    segment_frame_number: int
    segment_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionReport:
    generated_at: str
    source_summary: str
    model_summary: str
    model_count: int
    model_details: list[dict[str, str]]
    confidence_summary: str
    confidence_values: list[float]
    detect_roi_summary: str
    sea_analysis: dict[str, Any]
    sea_encounters: list[dict[str, Any]]
    postprocess: dict[str, Any]
    video_count: int
    dark_skip_enabled_video_count: int
    dark_skipped_video_count: int
    detection_frame_count: int
    segment_count: int
    class_counts: dict[str, int]
    duration_min_sec: float
    duration_max_sec: float
    duration_avg_sec: float
    rows: list[DetectionSegmentReportRow]
    timeline_frames: list[DetectionTimelineFrame]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
        payload["timeline_frames"] = [frame.to_dict() for frame in self.timeline_frames]
        return payload


REPORT_FIELDS = [
    "start_time",
    "end_time",
    "duration_sec",
    "sample_window_sec",
    "frame_count",
    "class_name",
    "best_match_pct",
    "best_confidence",
    "best_time",
    "best_segment_frame_number",
    "best_frame_index",
    "best_timestamp_sec",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area_px",
    "mask_area_px",
    "mask_width_px",
    "mask_height_px",
    "polygon_point_count",
    "avg_mask_area_px",
    "max_mask_area_px",
    "avg_mask_width_px",
    "max_mask_width_px",
    "avg_mask_height_px",
    "max_mask_height_px",
    "best_sea_ratio",
    "avg_sea_ratio",
    "min_sea_ratio",
    "max_sea_ratio",
    "video_name",
    "job_id",
    "preview_path",
    "preview_url",
    "video_url",
]


def _parse_report_time(value: Any) -> datetime | None:
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


def _positive_numbers(items: list[dict[str, Any]], key: str, *, fallback_key: str | None = None) -> list[float]:
    values: list[float] = []
    for item in items:
        raw = item.get(key)
        if raw is None and fallback_key is not None:
            raw = item.get(fallback_key)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if value > 0:
            values.append(value)
    return values


def _avg(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def _ratio_value(value: Any) -> float | None:
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= ratio <= 1.0:
        return None
    return ratio


def _ratio_values(items: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for item in items:
        ratio = _ratio_value(item.get(key))
        if ratio is not None:
            values.append(ratio)
    return values


def _avg_ratio(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def _source_summary(videos: list[dict[str, Any]], rows: list[DetectionSegmentReportRow]) -> str:
    names: list[str] = []
    for video in videos:
        name = str(video.get("video_name") or "").strip()
        if name and name not in names:
            names.append(name)
    if not names:
        for row in rows:
            name = row.video_name.strip()
            if name and name not in names:
                names.append(name)
    if not names:
        return "-"
    if len(names) == 1:
        return names[0]
    return f"{names[0]} 외 {len(names) - 1}개 영상"


def _append_model_detail(
    details: list[dict[str, str]],
    *,
    model_id: Any = None,
    name: Any = None,
    path: Any = None,
) -> None:
    clean_id = str(model_id or "").strip()
    clean_name = str(name or "").strip()
    clean_path = str(path or "").strip()
    if not clean_name and clean_path:
        clean_name = Path(clean_path).stem
    if not clean_name and clean_id:
        clean_name = clean_id
    if not clean_name and not clean_path:
        return
    if any(
        (clean_id and item.get("id") == clean_id)
        or (clean_path and item.get("path") == clean_path)
        or (not clean_id and not clean_path and item.get("name") == clean_name)
        for item in details
    ):
        return
    details.append(
        {
            "id": clean_id,
            "name": clean_name,
            "path": clean_path,
        }
    )


def _model_details(videos: list[dict[str, Any]], events: list[dict[str, Any]]) -> list[dict[str, str]]:
    details: list[dict[str, str]] = []
    for video in videos:
        for model in video.get("models") or []:
            if isinstance(model, dict):
                _append_model_detail(
                    details,
                    model_id=model.get("id"),
                    name=model.get("name"),
                    path=model.get("path"),
                )
        _append_model_detail(details, path=video.get("model"))

    for event in events:
        for det in event.get("detections") or []:
            if not isinstance(det, dict):
                continue
            _append_model_detail(
                details,
                model_id=det.get("model_id"),
                name=det.get("model_name"),
                path=det.get("model_path"),
            )
            names = det.get("ensemble_model_names") or []
            ids = det.get("ensemble_model_ids") or []
            for index, name in enumerate(names):
                model_id = ids[index] if index < len(ids) else None
                _append_model_detail(details, model_id=model_id, name=name)
    return details


def _model_summary(details: list[dict[str, str]], videos: list[dict[str, Any]] | None = None) -> str:
    names = [item["name"] for item in details if item.get("name")]
    if names:
        return ", ".join(names)
    if videos and all(bool(video.get("sea_only")) for video in videos):
        return "객체 탐지 안 함 (바다 영역만 분석)"
    return "-"


def _confidence_values(videos: list[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for video in videos:
        if video.get("sea_only"):
            continue
        try:
            value = float(video.get("confidence"))
        except (TypeError, ValueError):
            continue
        if value not in values:
            values.append(value)
    return sorted(values)


def _confidence_summary(values: list[float]) -> str:
    return ", ".join(f"{value:g}" for value in values) if values else "-"


def _roi_margin_percent(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number <= 1.0:
        number *= 100.0
    return number


def _detect_roi_label(roi: Any) -> str | None:
    if not isinstance(roi, dict):
        return None
    label = roi.get("label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    region = roi.get("region_percent")
    if isinstance(region, dict):
        try:
            return (
                f"x {float(region['x_min']):g}-{float(region['x_max']):g}%, "
                f"y {float(region['y_min']):g}-{float(region['y_max']):g}%"
            )
        except (KeyError, TypeError, ValueError):
            pass
    margins = roi.get("margins_percent") or roi.get("margins") or roi
    if not isinstance(margins, dict):
        return None
    top = _roi_margin_percent(margins.get("top"))
    right = _roi_margin_percent(margins.get("right"))
    bottom = _roi_margin_percent(margins.get("bottom"))
    left = _roi_margin_percent(margins.get("left"))
    if None in (top, right, bottom, left):
        return None
    return f"x {left:g}-{100.0 - right:g}%, y {top:g}-{100.0 - bottom:g}%"


def _detect_roi_summary(videos: list[dict[str, Any]]) -> str:
    labels: list[str] = []
    for video in videos:
        label = _detect_roi_label(video.get("detect_roi"))
        if label and label not in labels:
            labels.append(label)
    return ", ".join(labels) if labels else "-"


def _dark_skipped_video_count(videos: list[dict[str, Any]]) -> int:
    return sum(
        1
        for video in videos
        if video.get("skipped") and video.get("skip_reason") == "dark_video"
    )


def _dark_skip_enabled_video_count(videos: list[dict[str, Any]]) -> int:
    return sum(1 for video in videos if video.get("dark_skip_enabled"))


def _sea_report_summary(videos: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    enabled_videos = [
        video
        for video in videos
        if isinstance(video.get("sea_analysis"), dict) and video["sea_analysis"].get("enabled")
    ]
    methods: list[str] = []
    intervals: list[float] = []
    state_counts: Counter[str] = Counter()
    encounters: list[dict[str, Any]] = []
    sample_count = 0
    valid_count = 0
    unknown_count = 0
    encounter_events = 0
    departure_events = 0
    weighted_sea_ratio = 0.0
    weighted_sea_count = 0
    weighted_confidence = 0.0
    weighted_confidence_count = 0

    for video in enabled_videos:
        analysis = video["sea_analysis"]
        try:
            interval = float(video.get("sea_analysis_interval_sec"))
        except (TypeError, ValueError):
            interval = 5.0
        if interval not in intervals:
            intervals.append(interval)
        current_samples = int(analysis.get("sample_count") or 0)
        current_valid = int(analysis.get("valid_count") or 0)
        sample_count += current_samples
        valid_count += current_valid
        unknown_count += int(analysis.get("unknown_count") or 0)
        state_counts.update(analysis.get("state_counts") or {})
        for method in analysis.get("methods") or []:
            method = str(method)
            if method and method not in methods:
                methods.append(method)
        avg_ratio = _ratio_value(analysis.get("avg_sea_ratio"))
        if avg_ratio is not None and current_valid > 0:
            weighted_sea_ratio += avg_ratio * current_valid
            weighted_sea_count += current_valid
        avg_confidence = _ratio_value(analysis.get("avg_confidence"))
        if avg_confidence is not None and current_valid > 0:
            weighted_confidence += avg_confidence * current_valid
            weighted_confidence_count += current_valid
        for event in analysis.get("events") or []:
            if event.get("event") == "encounter_start":
                encounter_events += 1
            elif event.get("event") == "departure":
                departure_events += 1
        for segment in analysis.get("encounter_segments") or []:
            encounters.append(
                {
                    **dict(segment),
                    "job_id": str(video.get("job_id") or ""),
                    "video_name": str(video.get("video_name") or ""),
                }
            )

    encounters.sort(
        key=lambda item: (
            item.get("start_absolute_time") or "",
            item.get("video_name") or "",
            float(item.get("start_timestamp_sec") or 0.0),
        )
    )
    return (
        {
            "enabled_video_count": len(enabled_videos),
            "methods": methods,
            "interval_sec_values": sorted(intervals),
            "sample_count": sample_count,
            "valid_count": valid_count,
            "unknown_count": unknown_count,
            "unknown_fraction": round(unknown_count / sample_count, 4) if sample_count else 0.0,
            "state_counts": dict(state_counts),
            "avg_sea_ratio": (
                round(weighted_sea_ratio / weighted_sea_count, 4) if weighted_sea_count else None
            ),
            "avg_confidence": (
                round(weighted_confidence / weighted_confidence_count, 4)
                if weighted_confidence_count
                else None
            ),
            "encounter_count": len(encounters),
            "encounter_event_count": encounter_events,
            "departure_event_count": departure_events,
        },
        encounters,
    )


def _postprocess_summary_html(postprocess: dict[str, Any]) -> str:
    if not postprocess:
        return "<p>후처리: 적용 기록 없음</p>"

    enabled: list[str] = []
    if postprocess.get("merge_segments"):
        gap = float(postprocess.get("segment_merge_gap_sec") or 0.0)
        enabled.append(f"{gap:g}초 이내 구간 병합")
    if postprocess.get("remove_position_outliers"):
        enabled.append("위치 이상 제거")
    if postprocess.get("remove_size_outliers"):
        enabled.append("크기 이상 제거")
    if postprocess.get("remove_tall_thin_boxes"):
        enabled.append("세로형 빈 그물 제거")
    if postprocess.get("remove_right_edge_detections"):
        enabled.append("좌우 가장자리 제거")
    if postprocess.get("remove_static_short_tracks"):
        enabled.append("같은 위치 정지 제거")
    if postprocess.get("remove_temporal_isolated"):
        enabled.append("시간 고립/짧은 burst 제거")
    if postprocess.get("remove_color_outliers"):
        enabled.append("색상 이상 제거")
    label = ", ".join(enabled) if enabled else "선택된 후처리 없음"
    removed_by_condition = postprocess.get("removed_by_condition") or {}
    condition_labels = {
        "position_outlier": "위치",
        "size_outlier": "크기",
        "tall_thin_box": "세로형",
        "right_edge": "좌우가장자리",
        "static_short_track": "정지",
        "temporal_isolated": "시간고립",
        "color_outlier": "색상",
    }
    removed_parts = [
        f"{label_name} {int(removed_by_condition.get(key) or 0)}"
        for key, label_name in condition_labels.items()
        if key in removed_by_condition
    ]
    removed_detail = f" ({', '.join(removed_parts)})" if removed_parts else ""
    applied_at = postprocess.get("applied_at")
    applied = f" · 적용 시각 {html.escape(str(applied_at))}" if applied_at else ""
    return (
        "<p>후처리: "
        f"{html.escape(label)} · 구간 {int(postprocess.get('before_segment_count') or 0)}개"
        f" -> {int(postprocess.get('segment_count') or 0)}개"
        f" · 병합 {int(postprocess.get('merged_segment_count') or 0)}개"
        f" · 탐지 {int(postprocess.get('removed_detection_count') or 0)}개 제거"
        f" · 프레임 {int(postprocess.get('removed_event_count') or 0)}개 제거"
        f"{html.escape(removed_detail)}{applied}</p>"
    )


def build_detection_report(
    timeline_path: Path,
    *,
    asset_url_prefix: str = "/api/pipeline/detect",
) -> DetectionReport:
    timeline = load_timeline(timeline_path)
    videos = timeline.get("videos", []) or []
    events = timeline.get("events", []) or []
    videos_by_name = {video.get("video_name"): video for video in videos if video.get("video_name")}

    rows: list[DetectionSegmentReportRow] = []
    timeline_frames: list[DetectionTimelineFrame] = []
    for group in build_segment_frame_groups(timeline):
        first = group[0]
        last = group[-1]
        video_name = str(first.get("video_name") or "")
        class_name = str(first.get("class_name") or "")
        video = videos_by_name.get(video_name, {})
        stride = int(video.get("frame_stride") or 5)
        fps = float(video.get("fps") or 15.0)
        sample_step = stride / fps if fps > 0 else 0.0

        best = max(group, key=lambda item: float(item.get("confidence") or 0.0))
        best_segment_frame_number = next(
            (index for index, frame in enumerate(group, start=1) if frame is best),
            1,
        )
        bbox = best.get("bbox_xyxy")
        bbox_vals = [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else [None] * 4
        start_ts = float(first.get("timestamp_sec") or 0.0)
        end_ts = float(last.get("timestamp_sec") or start_ts)
        start_abs = _parse_report_time(first.get("absolute_time"))
        end_abs = _parse_report_time(last.get("absolute_time"))
        if start_abs is not None and end_abs is not None:
            duration = max(0.0, (end_abs - start_abs).total_seconds())
        else:
            duration = max(0.0, end_ts - start_ts)
        confidence = float(best.get("confidence") or 0.0)
        mask_areas = _positive_numbers(group, "mask_area_px", fallback_key="area_px")
        mask_widths = _positive_numbers(group, "mask_width_px")
        mask_heights = _positive_numbers(group, "mask_height_px")
        sea_ratios = _ratio_values(group, "sea_ratio")
        job_id = str(best.get("job_id") or "")
        preview_path = best.get("preview_path")
        preview_url = (
            f"{asset_url_prefix}/{job_id}/previews/{preview_path}"
            if job_id and preview_path
            else None
        )
        video_url = f"{asset_url_prefix}/{job_id}/video" if job_id else None

        segment_frame_count = len(group)
        for segment_frame_number, frame in enumerate(group, start=1):
            frame_job_id = str(frame.get("job_id") or "")
            frame_preview_path = frame.get("preview_path")
            frame_preview_url = (
                f"{asset_url_prefix}/{frame_job_id}/previews/{frame_preview_path}"
                if frame_job_id and frame_preview_path
                else None
            )
            frame_confidence = float(frame.get("confidence") or 0.0)
            frame_sea_ratio = _ratio_value(frame.get("sea_ratio"))
            abs_time = frame.get("absolute_time")
            if abs_time and not isinstance(abs_time, str):
                abs_time = abs_time.isoformat()
            timeline_frames.append(
                DetectionTimelineFrame(
                    time=str(frame.get("absolute_time_label") or ""),
                    absolute_time=str(abs_time or ""),
                    class_name=str(frame.get("class_name") or class_name),
                    confidence=round(frame_confidence, 4),
                    match_pct=round(frame_confidence * 100, 2),
                    video_name=str(frame.get("video_name") or video_name),
                    job_id=frame_job_id,
                    frame_index=frame.get("frame_index"),
                    timestamp_sec=frame.get("timestamp_sec"),
                    sea_ratio=round(frame_sea_ratio, 4) if frame_sea_ratio is not None else None,
                    sea_percent=round(frame_sea_ratio * 100.0, 2) if frame_sea_ratio is not None else None,
                    preview_path=frame_preview_path,
                    preview_url=frame_preview_url,
                    is_representative=frame is best,
                    segment_frame_number=segment_frame_number,
                    segment_frame_count=segment_frame_count,
                )
            )

        rows.append(
            DetectionSegmentReportRow(
                start_time=str(first.get("absolute_time_label") or ""),
                end_time=str(last.get("absolute_time_label") or ""),
                duration_sec=round(duration, 3),
                sample_window_sec=round(duration + sample_step, 3),
                frame_count=len(group),
                class_name=class_name,
                best_match_pct=round(confidence * 100, 2),
                best_confidence=round(confidence, 4),
                best_time=str(best.get("absolute_time_label") or ""),
                best_segment_frame_number=best_segment_frame_number,
                best_frame_index=best.get("frame_index"),
                best_timestamp_sec=best.get("timestamp_sec"),
                bbox_x1=bbox_vals[0],
                bbox_y1=bbox_vals[1],
                bbox_x2=bbox_vals[2],
                bbox_y2=bbox_vals[3],
                area_px=int(best.get("area_px") or 0),
                mask_area_px=int(best.get("mask_area_px") or best.get("area_px") or 0),
                mask_width_px=int(best.get("mask_width_px") or 0),
                mask_height_px=int(best.get("mask_height_px") or 0),
                polygon_point_count=int(best.get("polygon_point_count") or len(best.get("polygon_xy") or [])),
                avg_mask_area_px=_avg(mask_areas),
                max_mask_area_px=int(max(mask_areas, default=0)),
                avg_mask_width_px=_avg(mask_widths),
                max_mask_width_px=int(max(mask_widths, default=0)),
                avg_mask_height_px=_avg(mask_heights),
                max_mask_height_px=int(max(mask_heights, default=0)),
                best_sea_ratio=(
                    round(_ratio_value(best.get("sea_ratio")), 4)
                    if _ratio_value(best.get("sea_ratio")) is not None
                    else None
                ),
                avg_sea_ratio=_avg_ratio(sea_ratios),
                min_sea_ratio=round(min(sea_ratios), 4) if sea_ratios else None,
                max_sea_ratio=round(max(sea_ratios), 4) if sea_ratios else None,
                video_name=video_name,
                job_id=job_id,
                preview_path=preview_path,
                preview_url=preview_url,
                video_url=video_url,
            )
        )

    rows.sort(key=lambda row: (row.start_time, row.video_name, row.best_frame_index or 0))
    timeline_frames.sort(key=lambda frame: (frame.time, frame.video_name, frame.frame_index or 0))
    durations = [row.duration_sec for row in rows]
    classes = Counter(row.class_name for row in rows)
    model_details = _model_details(videos, events)
    confidence_values = _confidence_values(videos)
    sea_analysis, sea_encounters = _sea_report_summary(videos)
    return DetectionReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        source_summary=_source_summary(videos, rows),
        model_summary=_model_summary(model_details, videos),
        model_count=len(model_details),
        model_details=model_details,
        confidence_summary=_confidence_summary(confidence_values),
        confidence_values=confidence_values,
        detect_roi_summary=_detect_roi_summary(videos),
        sea_analysis=sea_analysis,
        sea_encounters=sea_encounters,
        postprocess=dict(timeline.get("postprocess") or {}),
        video_count=len(videos),
        dark_skip_enabled_video_count=_dark_skip_enabled_video_count(videos),
        dark_skipped_video_count=_dark_skipped_video_count(videos),
        detection_frame_count=len(events),
        segment_count=len(rows),
        class_counts=dict(classes),
        duration_min_sec=min(durations) if durations else 0.0,
        duration_max_sec=max(durations) if durations else 0.0,
        duration_avg_sec=round(sum(durations) / len(durations), 3) if durations else 0.0,
        rows=rows,
        timeline_frames=timeline_frames,
    )


def _bbox_label(row: DetectionSegmentReportRow) -> str:
    vals = [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2]
    if any(value is None for value in vals):
        return "-"
    return "[" + ", ".join(f"{float(value):.1f}" for value in vals) + "]"


def _mask_label(row: DetectionSegmentReportRow) -> str:
    if row.mask_area_px <= 0:
        return "-"
    size = f"{row.mask_width_px}x{row.mask_height_px}" if row.mask_width_px and row.mask_height_px else "-"
    return f"{row.mask_area_px} px / {size} / {row.polygon_point_count}점"


def _segment_mask_stats_label(row: DetectionSegmentReportRow) -> str:
    if row.avg_mask_area_px <= 0:
        return "-"
    width = f"{row.avg_mask_width_px:.1f}" if row.avg_mask_width_px > 0 else "-"
    height = f"{row.avg_mask_height_px:.1f}" if row.avg_mask_height_px > 0 else "-"
    return (
        f"avg {row.avg_mask_area_px:.1f} px / {width}x{height}"
        f"<br>max {row.max_mask_area_px} px / {row.max_mask_width_px}x{row.max_mask_height_px}"
    )


def _format_ratio_percent(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value * 100.0:.1f}%"


def _sea_ratio_label(row: DetectionSegmentReportRow) -> str:
    if row.avg_sea_ratio is None:
        return "-"
    parts = [f"avg {_format_ratio_percent(row.avg_sea_ratio)}"]
    if row.min_sea_ratio is not None and row.max_sea_ratio is not None:
        parts.append(f"range {_format_ratio_percent(row.min_sea_ratio)}-{_format_ratio_percent(row.max_sea_ratio)}")
    if row.best_sea_ratio is not None:
        parts.append(f"대표 {_format_ratio_percent(row.best_sea_ratio)}")
    return "<br>".join(parts)


def _render_sea_analysis(report: DetectionReport) -> str:
    summary = report.sea_analysis
    if not int(summary.get("enabled_video_count") or 0):
        return '<p class="muted">바다 영역 분석을 사용하지 않았습니다.</p>'
    methods = ", ".join(str(item) for item in summary.get("methods") or []) or "-"
    intervals = ", ".join(
        "모든 처리 프레임" if float(value) == 0.0 else f"{float(value):g}초"
        for value in summary.get("interval_sec_values") or []
    ) or "-"
    avg_ratio = _format_ratio_percent(_ratio_value(summary.get("avg_sea_ratio")))
    confidence = _format_ratio_percent(_ratio_value(summary.get("avg_confidence")))
    unknown = _format_ratio_percent(_ratio_value(summary.get("unknown_fraction")))
    rows = []
    for item in report.sea_encounters:
        min_sea = _format_ratio_percent(_ratio_value(item.get("min_sea_ratio")))
        max_vessel = _format_ratio_percent(_ratio_value(item.get("max_vessel_ratio")))
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(item.get('start_time') or '-'))}</td>"
            f"<td>{html.escape(str(item.get('end_time') or '-'))}</td>"
            f"<td>{float(item.get('duration_sec') or 0.0):.1f}s</td>"
            f"<td>{int(item.get('sample_count') or 0)}</td>"
            f"<td>{min_sea}</td>"
            f"<td>{max_vessel}</td>"
            f"<td>{html.escape(str(item.get('video_name') or '-'))}</td>"
            "</tr>"
        )
    body = "".join(rows) or '<tr><td colspan="7">확정된 조우 구간이 없습니다.</td></tr>'
    return (
        f"<p>분석 영상 {int(summary.get('enabled_video_count') or 0)}개 · "
        f"샘플 {int(summary.get('sample_count') or 0)}개 · 평균 바다 {avg_ratio} · "
        f"평균 신뢰도 {confidence} · 판정 불가 {unknown} · 엔진 {html.escape(methods)}</p>"
        f"<p>바다 분석 간격: {html.escape(intervals)}</p>"
        '<table><thead><tr><th>조우 시작</th><th>이탈/종료</th><th>시간</th><th>샘플</th>'
        f'<th>최소 바다</th><th>최대 선박</th><th>영상</th></tr></thead><tbody>{body}</tbody></table>'
    )


def _link_or_text(label: str, href: str | None) -> str:
    escaped_label = html.escape(label or "-")
    if not href:
        return escaped_label
    return f'<a href="{html.escape(href)}" target="_blank" rel="noreferrer">{escaped_label}</a>'


def _preview_title(row: DetectionSegmentReportRow) -> str:
    sea = ""
    if row.best_sea_ratio is not None:
        sea = f" · 바다 {_format_ratio_percent(row.best_sea_ratio)}"
    return (
        f"{row.class_name} · {row.best_match_pct:.2f}% · "
        f"연속구간 프레임 {row.best_segment_frame_number}/{row.frame_count} · "
        f"{row.start_time} - {row.end_time}{sea} · {row.video_name}"
    )


PreviewKey = tuple[str, int | None, str | None, str]


def _preview_key(
    job_id: str,
    frame_index: int | None,
    preview_url: str | None,
    class_name: str,
) -> PreviewKey:
    return (job_id, frame_index, preview_url, class_name)


def _row_preview_key(row: DetectionSegmentReportRow) -> PreviewKey:
    return _preview_key(row.job_id, row.best_frame_index, row.preview_url, row.class_name)


def _frame_preview_key(frame: DetectionTimelineFrame) -> PreviewKey:
    return _preview_key(frame.job_id, frame.frame_index, frame.preview_url, frame.class_name)


def _preview_thumbnail(row: DetectionSegmentReportRow, preview_index: int | None = None) -> str:
    label = html.escape(row.best_time or "대표 프레임")
    if not row.preview_url:
        return label
    href = html.escape(row.preview_url)
    alt = html.escape(f"{row.class_name} 대표 프레임 {row.best_time}".strip())
    index_attr = f' data-preview-index="{preview_index}"' if preview_index is not None else ""
    return (
        f'<a class="preview-thumb" href="{href}" data-preview-url="{href}" '
        f'data-preview-title="{html.escape(_preview_title(row))}"{index_attr}>'
        f'<img src="{href}" alt="{alt}" loading="lazy" />'
        f'<span>{label}</span>'
        "</a>"
    )


def _timeline_color(class_name: str) -> str:
    palette = [
        "#2563eb",
        "#16a34a",
        "#dc2626",
        "#9333ea",
        "#c2410c",
        "#0891b2",
        "#be123c",
        "#4f46e5",
    ]
    index = sum(ord(ch) for ch in class_name or "-") % len(palette)
    return palette[index]


def _format_timeline_label(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _timeline_data_time(dt: datetime) -> str:
    return dt.isoformat()


def _render_timeline_overview(report: DetectionReport, preview_indices: dict[PreviewKey, int]) -> str:
    frame_entries: list[tuple[DetectionTimelineFrame, datetime]] = []
    for frame in report.timeline_frames:
        event_time = _parse_report_time(frame.time) or _parse_report_time(frame.absolute_time)
        if event_time is None:
            continue
        frame_entries.append((frame, event_time))

    entries: list[tuple[DetectionSegmentReportRow, datetime, datetime]] = []
    for row in report.rows:
        start = _parse_report_time(row.start_time)
        end = _parse_report_time(row.end_time) or start
        if start is None or end is None:
            continue
        if end < start:
            end = start
        entries.append((row, start, end))

    if not entries and not frame_entries:
        return '<p class="muted">표시할 시간 정보가 있는 탐지 구간이 없습니다.</p>'

    starts = [start for _row, start, _end in entries] + [time for _frame, time in frame_entries]
    ends = [end for _row, _start, end in entries] + [time for _frame, time in frame_entries]
    range_start = min(starts)
    range_end = max(ends)
    if range_end <= range_start:
        range_end = range_start
    total_sec = max((range_end - range_start).total_seconds(), 1.0)

    frame_markers: list[str] = []
    for frame, event_time in frame_entries:
        left = max(0.0, min(100.0, ((event_time - range_start).total_seconds() / total_sec) * 100.0))
        sea_label = ""
        if frame.sea_ratio is not None:
            sea_label = f" · 바다 {_format_ratio_percent(frame.sea_ratio)}"
        title = (
            f"{frame.class_name} · {frame.match_pct:.2f}% · "
            f"연속구간 프레임 {frame.segment_frame_number}/{frame.segment_frame_count} · "
            f"{frame.time}{sea_label} · {frame.video_name}"
        )
        preview_index = preview_indices.get(_frame_preview_key(frame))
        index_attr = f' data-preview-index="{preview_index}"' if preview_index is not None else ""
        marker_class = "timeline-frame-marker representative" if frame.is_representative else "timeline-frame-marker"
        marker = (
            f'<a class="{marker_class}" href="{html.escape(frame.preview_url)}" '
            f'data-preview-url="{html.escape(frame.preview_url)}" '
            f'data-preview-title="{html.escape(title)}"{index_attr} title="{html.escape(title)}" '
            f'style="left:{left:.4f}%"></a>'
            if frame.preview_url
            else f'<div class="{marker_class}" title="{html.escape(title)}" style="left:{left:.4f}%"></div>'
        )
        frame_markers.append(marker)

    markers: list[str] = []
    for row, start, end in entries:
        left = max(0.0, min(100.0, ((start - range_start).total_seconds() / total_sec) * 100.0))
        width = max(0.002, min(100.0 - left, ((end - start).total_seconds() / total_sec) * 100.0))
        color = _timeline_color(row.class_name)
        title = _preview_title(row)
        preview_index = preview_indices.get(_row_preview_key(row))
        index_attr = f' data-preview-index="{preview_index}"' if preview_index is not None else ""
        label = f"{html.escape(row.class_name)} {row.best_match_pct:.0f}%"
        content = f'<span>{label}</span>'
        marker = (
            f'<a class="timeline-marker" href="{html.escape(row.preview_url)}" '
            f'data-preview-url="{html.escape(row.preview_url)}" '
            f'data-preview-title="{html.escape(title)}"{index_attr} title="{html.escape(title)}" '
            f'style="left:{left:.4f}%;width:{width:.4f}%;background:{color}">{content}</a>'
            if row.preview_url
            else f'<div class="timeline-marker" title="{html.escape(title)}" '
            f'style="left:{left:.4f}%;width:{width:.4f}%;background:{color}">{content}</div>'
        )
        markers.append(marker)

    class_legend = " ".join(
        f'<span class="legend-item"><i style="background:{_timeline_color(name)}"></i>{html.escape(name)} ({count})</span>'
        for name, count in sorted(report.class_counts.items())
    )
    return (
        '<div class="timeline-overview" data-timeline '
        f'data-range-start="{html.escape(_timeline_data_time(range_start))}" '
        f'data-range-end="{html.escape(_timeline_data_time(range_end))}">'
        '<div class="timeline-toolbar">'
        '<button type="button" data-timeline-zoom="out" title="축소">-</button>'
        '<button type="button" data-timeline-zoom="reset" title="배율 초기화">100%</button>'
        '<button type="button" data-timeline-zoom="in" title="확대">+</button>'
        '<span class="timeline-zoom-label" data-timeline-zoom-label>100%</span>'
        '<span class="timeline-visible-range" data-timeline-visible-range></span>'
        '</div>'
        '<div class="timeline-scale">'
        f'<span>{html.escape(_format_timeline_label(range_start))}</span>'
        f'<span>{html.escape(_format_timeline_label(range_end))}</span>'
        '</div>'
        f'<div class="timeline-viewport" tabindex="0"><div class="timeline-bar" data-timeline-track>{"".join(frame_markers)}{"".join(markers)}</div></div>'
        f'<div class="timeline-legend">{class_legend or "-"}</div>'
        '<p class="muted">얇은 amber 라인은 탐지된 개별 프레임, 진한 막대는 구간 대표 프레임입니다. 확대 후 좌우로 스크롤할 수 있고, Ctrl/Command+휠로 배율을 바꿀 수 있습니다.</p>'
        '</div>'
    )


def _report_preview_modal() -> str:
    return """
  <div class="report-preview-modal hidden" data-report-preview-modal>
    <div class="report-preview-backdrop" data-report-preview-close></div>
    <div class="report-preview-dialog" role="dialog" aria-modal="true" aria-label="대표 프레임">
      <button type="button" class="report-preview-close" data-report-preview-close aria-label="닫기">×</button>
      <button type="button" class="report-preview-nav report-preview-prev" data-report-preview-prev aria-label="이전 탐지 프레임">‹</button>
      <img data-report-preview-image alt="대표 프레임" />
      <button type="button" class="report-preview-nav report-preview-next" data-report-preview-next aria-label="다음 탐지 프레임">›</button>
      <div class="report-preview-caption" data-report-preview-caption></div>
    </div>
  </div>
"""


def _report_interactivity_script() -> str:
    return """<script>
(function () {
  const zoomLevels = [1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 80, 120, 160, 200];

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function formatTimelineDate(date) {
    const pad = (value) => String(value).padStart(2, "0");
    return [
      date.getFullYear(),
      "-",
      pad(date.getMonth() + 1),
      "-",
      pad(date.getDate()),
      " ",
      pad(date.getHours()),
      ":",
      pad(date.getMinutes()),
      ":",
      pad(date.getSeconds()),
    ].join("");
  }

  document.querySelectorAll("[data-timeline]").forEach((root) => {
    const viewport = root.querySelector(".timeline-viewport");
    const track = root.querySelector("[data-timeline-track]");
    const label = root.querySelector("[data-timeline-zoom-label]");
    const visibleRange = root.querySelector("[data-timeline-visible-range]");
    const rangeStart = new Date(root.getAttribute("data-range-start") || "");
    const rangeEnd = new Date(root.getAttribute("data-range-end") || "");
    if (!viewport || !track) return;

    let zoomIndex = 0;

    function updateVisibleRange() {
      if (!visibleRange || Number.isNaN(rangeStart.getTime()) || Number.isNaN(rangeEnd.getTime())) return;
      const trackWidth = Math.max(track.scrollWidth || track.getBoundingClientRect().width, 1);
      const startRatio = clamp(viewport.scrollLeft / trackWidth, 0, 1);
      const endRatio = clamp((viewport.scrollLeft + viewport.clientWidth) / trackWidth, 0, 1);
      const totalMs = Math.max(rangeEnd.getTime() - rangeStart.getTime(), 0);
      const visibleStart = new Date(rangeStart.getTime() + totalMs * startRatio);
      const visibleEnd = new Date(rangeStart.getTime() + totalMs * endRatio);
      visibleRange.textContent = `현재 화면: ${formatTimelineDate(visibleStart)} - ${formatTimelineDate(visibleEnd)}`;
    }

    function setZoom(nextIndex, anchorRatio) {
      const oldWidth = Math.max(track.getBoundingClientRect().width, 1);
      const defaultAnchor = (viewport.scrollLeft + viewport.clientWidth / 2) / oldWidth;
      const anchor = clamp(anchorRatio == null ? defaultAnchor : anchorRatio, 0, 1);
      zoomIndex = clamp(nextIndex, 0, zoomLevels.length - 1);
      const zoom = zoomLevels[zoomIndex];
      track.style.width = `${zoom * 100}%`;
      if (label) label.textContent = `${Math.round(zoom * 100)}%`;

      root.querySelectorAll("[data-timeline-zoom]").forEach((button) => {
        const action = button.getAttribute("data-timeline-zoom");
        button.disabled =
          (action === "out" && zoomIndex === 0) ||
          (action === "in" && zoomIndex === zoomLevels.length - 1);
      });

      requestAnimationFrame(() => {
        const newWidth = Math.max(track.getBoundingClientRect().width, 1);
        viewport.scrollLeft = anchor * newWidth - viewport.clientWidth / 2;
        updateVisibleRange();
      });
    }

    root.addEventListener("click", (event) => {
      const button = event.target.closest("[data-timeline-zoom]");
      if (!button) return;
      const action = button.getAttribute("data-timeline-zoom");
      if (action === "in") setZoom(zoomIndex + 1);
      if (action === "out") setZoom(zoomIndex - 1);
      if (action === "reset") setZoom(0, 0);
    });

    viewport.addEventListener(
      "wheel",
      (event) => {
        if (!event.ctrlKey && !event.metaKey) return;
        event.preventDefault();
        const rect = viewport.getBoundingClientRect();
        const trackWidth = Math.max(track.getBoundingClientRect().width, 1);
        const anchor = (viewport.scrollLeft + event.clientX - rect.left) / trackWidth;
        setZoom(zoomIndex + (event.deltaY > 0 ? -1 : 1), anchor);
      },
      { passive: false },
    );
    viewport.addEventListener("scroll", updateVisibleRange);
    window.addEventListener("resize", updateVisibleRange);

    setZoom(0, 0);
  });

  const modal = document.querySelector("[data-report-preview-modal]");
  const modalImage = document.querySelector("[data-report-preview-image]");
  const modalCaption = document.querySelector("[data-report-preview-caption]");
  const previewPrev = document.querySelector("[data-report-preview-prev]");
  const previewNext = document.querySelector("[data-report-preview-next]");
  const previewItemsByIndex = new Map();
  let currentPreviewIndex = null;

  document.querySelectorAll("[data-preview-url][data-preview-index]").forEach((item) => {
    const index = Number(item.getAttribute("data-preview-index"));
    if (!Number.isFinite(index) || previewItemsByIndex.has(index)) return;
    previewItemsByIndex.set(index, {
      url: item.getAttribute("data-preview-url") || "",
      title: item.getAttribute("data-preview-title") || "",
    });
  });
  const previewIndices = Array.from(previewItemsByIndex.keys()).sort((a, b) => a - b);

  function previewIndexPosition(index) {
    return previewIndices.indexOf(index);
  }

  function updatePreviewNav() {
    const position = currentPreviewIndex == null ? -1 : previewIndexPosition(currentPreviewIndex);
    if (previewPrev) previewPrev.disabled = position <= 0;
    if (previewNext) previewNext.disabled = position < 0 || position >= previewIndices.length - 1;
  }

  function clearTimelinePreviewHighlight() {
    document.querySelectorAll("[data-timeline-track] .timeline-preview-active").forEach((item) => {
      item.classList.remove("timeline-preview-active");
    });
  }

  function updateTimelinePreviewHighlight(index) {
    clearTimelinePreviewHighlight();
    if (!Number.isFinite(index)) return;
    const activeItems = document.querySelectorAll(`[data-timeline-track] [data-preview-index="${index}"]`);
    activeItems.forEach((item) => item.classList.add("timeline-preview-active"));
    const firstItem = activeItems[0];
    if (!firstItem) return;
    const viewport = firstItem.closest(".timeline-viewport");
    if (!viewport) return;
    const viewportRect = viewport.getBoundingClientRect();
    const itemRect = firstItem.getBoundingClientRect();
    const margin = Math.min(160, Math.max(48, viewport.clientWidth * 0.18));
    if (itemRect.left >= viewportRect.left + margin && itemRect.right <= viewportRect.right - margin) return;
    const itemCenter = itemRect.left + itemRect.width / 2 - viewportRect.left + viewport.scrollLeft;
    viewport.scrollTo({
      left: Math.max(0, itemCenter - viewport.clientWidth / 2),
      behavior: "smooth",
    });
  }

  function openPreviewModal(item, index) {
    if (!modal || !modalImage || !item.url) return;
    modalImage.src = item.url;
    if (modalCaption) modalCaption.textContent = item.title || "";
    currentPreviewIndex = Number.isFinite(index) ? index : null;
    updatePreviewNav();
    updateTimelinePreviewHighlight(currentPreviewIndex);
    modal.classList.remove("hidden");
  }

  function showPreviewAtOffset(offset) {
    if (currentPreviewIndex == null) return;
    const position = previewIndexPosition(currentPreviewIndex);
    const nextIndex = previewIndices[position + offset];
    if (nextIndex == null) return;
    openPreviewModal(previewItemsByIndex.get(nextIndex), nextIndex);
  }

  function closePreviewModal() {
    if (!modal || !modalImage) return;
    modal.classList.add("hidden");
    modalImage.removeAttribute("src");
    if (modalCaption) modalCaption.textContent = "";
    currentPreviewIndex = null;
    updatePreviewNav();
    clearTimelinePreviewHighlight();
  }

  document.addEventListener("click", (event) => {
    const prevButton = event.target.closest("[data-report-preview-prev]");
    if (prevButton) {
      showPreviewAtOffset(-1);
      return;
    }

    const nextButton = event.target.closest("[data-report-preview-next]");
    if (nextButton) {
      showPreviewAtOffset(1);
      return;
    }

    const preview = event.target.closest("[data-preview-url]");
    if (preview && modal && modalImage) {
      event.preventDefault();
      const index = Number(preview.getAttribute("data-preview-index"));
      openPreviewModal(
        {
          url: preview.getAttribute("data-preview-url") || "",
          title: preview.getAttribute("data-preview-title") || "",
        },
        index,
      );
      return;
    }

    if (event.target.closest("[data-report-preview-close]")) {
      closePreviewModal();
    }
  });

  document.addEventListener("keydown", (event) => {
    const isOpen = modal && !modal.classList.contains("hidden");
    if (event.key === "Escape") closePreviewModal();
    if (!isOpen) return;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      showPreviewAtOffset(-1);
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      showPreviewAtOffset(1);
    }
  });
})();
</script>"""


def _render_table_rows(
    rows: list[DetectionSegmentReportRow],
    preview_indices: dict[PreviewKey, int],
) -> str:
    rendered = []
    for row in rows:
        preview_index = preview_indices.get(_row_preview_key(row))
        rendered.append(
            "<tr>"
            f"<td>{html.escape(row.start_time)} - {html.escape(row.end_time)}</td>"
            f"<td>{row.duration_sec:.3f}s</td>"
            f"<td>{row.sample_window_sec:.3f}s</td>"
            f"<td>{row.frame_count}</td>"
            f"<td>{html.escape(row.class_name)}</td>"
            f"<td>{row.best_match_pct:.2f}%</td>"
            f"<td>{_preview_thumbnail(row, preview_index)}</td>"
            f"<td>{html.escape(_bbox_label(row))}</td>"
            f"<td>{html.escape(_mask_label(row))}</td>"
            f"<td>{_segment_mask_stats_label(row)}</td>"
            f"<td>{_sea_ratio_label(row)}</td>"
            f"<td>{_link_or_text(row.video_name, row.video_url)}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def render_detection_report_html(report: DetectionReport) -> str:
    top_duration = sorted(report.rows, key=lambda row: (row.duration_sec, row.best_confidence), reverse=True)[:20]
    top_confidence = sorted(report.rows, key=lambda row: (row.best_confidence, row.duration_sec), reverse=True)[:20]
    class_counts = ", ".join(f"{html.escape(name)}: {count}" for name, count in sorted(report.class_counts.items()))
    preview_indices = {
        _frame_preview_key(frame): index
        for index, frame in enumerate(report.timeline_frames)
        if frame.preview_url
    }
    timeline_overview = _render_timeline_overview(report, preview_indices)
    preview_modal = _report_preview_modal()
    interactivity_script = _report_interactivity_script()
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>탐지 결과 요약 리포트</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    .timeline-overview {{ margin: 18px 0 28px; }}
    .timeline-toolbar {{ display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }}
    .timeline-toolbar button {{ min-width: 38px; height: 30px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; color: #111827; cursor: pointer; font-weight: 700; }}
    .timeline-toolbar button:hover:not(:disabled) {{ background: #f3f4f6; }}
    .timeline-toolbar button:disabled {{ color: #9ca3af; cursor: default; }}
    .timeline-zoom-label {{ color: #374151; font-size: 12px; margin-left: 4px; min-width: 42px; }}
    .timeline-visible-range {{ color: #374151; font-size: 12px; margin-left: 8px; }}
    .timeline-scale {{ display: flex; justify-content: space-between; gap: 16px; color: #6b7280; font-size: 12px; margin-bottom: 6px; }}
    .timeline-viewport {{ border: 1px solid #d1d5db; border-radius: 8px; overflow-x: auto; overflow-y: hidden; background: #f9fafb; }}
    .timeline-viewport:focus {{ outline: 2px solid #93c5fd; outline-offset: 2px; }}
    .timeline-bar {{ position: relative; min-width: 100%; height: 88px; background: #f9fafb; overflow: hidden; }}
    .timeline-bar::before {{ content: ""; position: absolute; left: 0; right: 0; top: 28px; border-top: 1px solid #d1d5db; }}
    .timeline-bar::after {{ content: ""; position: absolute; left: 0; right: 0; top: 66px; border-top: 1px solid #e5e7eb; }}
    .timeline-frame-marker {{ position: absolute; top: 56px; width: 2px; height: 24px; min-width: 2px; background: #f59e0b; border-radius: 1px; opacity: 0.62; cursor: pointer; z-index: 1; }}
    .timeline-frame-marker:hover {{ opacity: 1; width: 4px; }}
    .timeline-frame-marker.representative {{ top: 7px; height: 40px; background: #0f766e; opacity: 0.85; z-index: 3; }}
    .timeline-frame-marker.representative::after {{ content: ""; position: absolute; left: 0; top: 49px; width: 2px; height: 24px; background: #064e3b; border-radius: 1px; }}
    .timeline-frame-marker.representative:hover::after {{ width: 4px; }}
    .timeline-frame-marker.timeline-preview-active {{ width: 6px; background: #dc2626; opacity: 1; box-shadow: 0 0 0 2px rgba(220,38,38,0.22); z-index: 5; }}
    .timeline-frame-marker.representative.timeline-preview-active::after {{ width: 6px; background: #dc2626; box-shadow: 0 0 0 2px rgba(220,38,38,0.22); }}
    .timeline-marker {{ position: absolute; top: 10px; height: 34px; min-width: 5px; border-radius: 4px; color: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.22); overflow: hidden; cursor: pointer; z-index: 2; }}
    .timeline-marker.timeline-preview-active {{ outline: 2px solid #111827; outline-offset: 2px; box-shadow: 0 0 0 4px rgba(220,38,38,0.22); z-index: 4; }}
    .timeline-marker span {{ display: block; padding: 9px 6px 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; font-size: 11px; font-weight: 700; }}
    .timeline-legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 8px; font-size: 12px; color: #374151; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 5px; }}
    .legend-item i {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
    .report-preview-modal.hidden {{ display: none; }}
    .report-preview-modal {{ position: fixed; inset: 0; z-index: 999; display: grid; place-items: center; padding: 24px; }}
    .report-preview-backdrop {{ position: absolute; inset: 0; background: rgba(17, 24, 39, 0.72); }}
    .report-preview-dialog {{ position: relative; width: min(1100px, 94vw); max-height: 92vh; background: #fff; border-radius: 8px; box-shadow: 0 18px 50px rgba(0, 0, 0, 0.32); padding: 14px; display: grid; gap: 10px; }}
    .report-preview-dialog img {{ width: 100%; max-height: 78vh; object-fit: contain; background: #111827; border-radius: 6px; }}
    .report-preview-close {{ position: absolute; top: 8px; right: 8px; width: 34px; height: 34px; border: 1px solid #d1d5db; border-radius: 6px; background: #fff; color: #111827; cursor: pointer; font-size: 22px; line-height: 1; }}
    .report-preview-nav {{ position: absolute; top: 50%; transform: translateY(-50%); width: 42px; height: 54px; border: 1px solid rgba(255,255,255,0.42); border-radius: 7px; background: rgba(17, 24, 39, 0.72); color: #fff; cursor: pointer; font-size: 34px; line-height: 1; }}
    .report-preview-nav:disabled {{ opacity: 0.28; cursor: default; }}
    .report-preview-prev {{ left: 22px; }}
    .report-preview-next {{ right: 22px; }}
    .report-preview-caption {{ color: #374151; font-size: 13px; line-height: 1.35; padding-right: 42px; }}
    .preview-thumb {{ display: grid; gap: 5px; width: 144px; color: #2563eb; }}
    .preview-thumb img {{ width: 144px; aspect-ratio: 16 / 9; object-fit: cover; border: 1px solid #d1d5db; border-radius: 6px; background: #f3f4f6; }}
    .preview-thumb span {{ font-size: 12px; line-height: 1.2; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    a {{ color: #2563eb; text-decoration: none; }}
    a:hover {{ text-decoration: underline; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>탐지 결과 요약 리포트</h1>
  <p class="muted">생성 시각: {html.escape(report.generated_at)}</p>
  <div class="cards">
    <div class="card"><div class="value">{report.video_count}</div><div>분석 영상</div></div>
    <div class="card"><div class="value">{report.dark_skipped_video_count}</div><div>어두워 건너뜀</div></div>
    <div class="card"><div class="value">{report.detection_frame_count}</div><div>탐지 프레임</div></div>
    <div class="card"><div class="value">{report.segment_count}</div><div>연속 탐지 구간</div></div>
    <div class="card"><div class="value">{len(report.sea_encounters)}</div><div>조우 구간</div></div>
    <div class="card"><div class="value">{report.duration_max_sec:.3f}s</div><div>최장 연속 시간</div></div>
  </div>
  <p>소스: {html.escape(report.source_summary)}</p>
  <p>사용 모델: {html.escape(report.model_summary)}</p>
  <p>Confidence ratio: {html.escape(report.confidence_summary)}</p>
  <p>탐지 영역: {html.escape(report.detect_roi_summary)}</p>
  {_postprocess_summary_html(report.postprocess)}
  <p>어두운 영상 건너뛰기: 전체 {report.video_count}개 중 {report.dark_skipped_video_count}개 건너뜀 · 검사 적용 {report.dark_skip_enabled_video_count}개</p>
  <p>클래스별 구간 수: {class_counts or "-"}</p>
  <p>평균 연속 시간: {report.duration_avg_sec:.3f}s · 최소: {report.duration_min_sec:.3f}s · 최대: {report.duration_max_sec:.3f}s</p>
  <p class="muted">대표 프레임 링크는 bbox와 mask가 표시된 preview 이미지를 엽니다. 영상 링크는 해당 탐지 job의 원본 영상을 엽니다.</p>

  <h2>바다 영역 및 조우 분석</h2>
  {_render_sea_analysis(report)}

  <h2>전체 타임라인</h2>
  {timeline_overview}

  <h2>연속 시간이 긴 대표 구간</h2>
  <table>
    <thead><tr><th>구간</th><th>연속 시간</th><th>샘플 창</th><th>프레임</th><th>클래스</th><th>최고 일치율</th><th>대표 프레임</th><th>bbox</th><th>대표 mask</th><th>구간 mask 통계</th><th>바다 비율</th><th>영상</th></tr></thead>
    <tbody>{_render_table_rows(top_duration, preview_indices)}</tbody>
  </table>

  <h2>일치율이 높은 대표 구간</h2>
  <table>
    <thead><tr><th>구간</th><th>연속 시간</th><th>샘플 창</th><th>프레임</th><th>클래스</th><th>최고 일치율</th><th>대표 프레임</th><th>bbox</th><th>대표 mask</th><th>구간 mask 통계</th><th>바다 비율</th><th>영상</th></tr></thead>
    <tbody>{_render_table_rows(top_confidence, preview_indices)}</tbody>
  </table>

  <h2>전체 연속 탐지 구간</h2>
  <table>
    <thead><tr><th>구간</th><th>연속 시간</th><th>샘플 창</th><th>프레임</th><th>클래스</th><th>최고 일치율</th><th>대표 프레임</th><th>bbox</th><th>대표 mask</th><th>구간 mask 통계</th><th>바다 비율</th><th>영상</th></tr></thead>
    <tbody>{_render_table_rows(report.rows, preview_indices)}</tbody>
  </table>
  {preview_modal}
  {interactivity_script}
</body>
</html>
"""


def write_detection_report_bundle(
    timeline_path: Path,
    output_dir: Path,
    *,
    asset_url_prefix: str = "/api/pipeline/detect",
) -> dict[str, Any]:
    report = build_detection_report(timeline_path, asset_url_prefix=asset_url_prefix)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = f"detection-report-{stamp}"
    html_path = output_dir / f"{base}.html"
    csv_path = output_dir / f"{base}.csv"
    json_path = output_dir / f"{base}.json"

    html_path.write_text(render_detection_report_html(report), encoding="utf-8")
    json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        for row in report.rows:
            writer.writerow(row.to_dict())

    return {
        "report": report.to_dict(),
        "output_dir": str(output_dir.resolve()),
        "files": {
            "html": html_path.name,
            "csv": csv_path.name,
            "json": json_path.name,
        },
    }
