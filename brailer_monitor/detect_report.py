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

from .detect_timeline import expand_class_events, load_timeline, merge_consecutive_frames


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
    best_frame_index: int | None
    best_timestamp_sec: float | None
    bbox_x1: float | None
    bbox_y1: float | None
    bbox_x2: float | None
    bbox_y2: float | None
    area_px: int
    video_name: str
    job_id: str
    preview_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DetectionReport:
    generated_at: str
    video_count: int
    detection_frame_count: int
    segment_count: int
    class_counts: dict[str, int]
    duration_min_sec: float
    duration_max_sec: float
    duration_avg_sec: float
    rows: list[DetectionSegmentReportRow]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["rows"] = [row.to_dict() for row in self.rows]
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
    "best_frame_index",
    "best_timestamp_sec",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area_px",
    "video_name",
    "job_id",
    "preview_path",
]


def build_detection_report(timeline_path: Path) -> DetectionReport:
    timeline = load_timeline(timeline_path)
    videos = timeline.get("videos", []) or []
    events = timeline.get("events", []) or []
    videos_by_name = {video.get("video_name"): video for video in videos if video.get("video_name")}
    frames_by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for frame in expand_class_events(events):
        key = (str(frame["video_name"]), str(frame["class_name"]))
        frames_by_key.setdefault(key, []).append(frame)

    rows: list[DetectionSegmentReportRow] = []
    for (video_name, class_name), frames in frames_by_key.items():
        video = videos_by_name.get(video_name, {})
        stride = int(video.get("frame_stride") or 5)
        fps = float(video.get("fps") or 15.0)
        sample_step = stride / fps if fps > 0 else 0.0

        for group in merge_consecutive_frames(frames, frame_stride=stride):
            first = group[0]
            last = group[-1]
            best = max(group, key=lambda item: float(item.get("confidence") or 0.0))
            bbox = best.get("bbox_xyxy")
            bbox_vals = [float(v) for v in bbox] if isinstance(bbox, list) and len(bbox) == 4 else [None] * 4
            start_ts = float(first.get("timestamp_sec") or 0.0)
            end_ts = float(last.get("timestamp_sec") or start_ts)
            duration = max(0.0, end_ts - start_ts)
            confidence = float(best.get("confidence") or 0.0)

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
                    best_frame_index=best.get("frame_index"),
                    best_timestamp_sec=best.get("timestamp_sec"),
                    bbox_x1=bbox_vals[0],
                    bbox_y1=bbox_vals[1],
                    bbox_x2=bbox_vals[2],
                    bbox_y2=bbox_vals[3],
                    area_px=int(best.get("area_px") or 0),
                    video_name=video_name,
                    job_id=str(best.get("job_id") or ""),
                    preview_path=best.get("preview_path"),
                )
            )

    rows.sort(key=lambda row: (row.start_time, row.video_name, row.best_frame_index or 0))
    durations = [row.duration_sec for row in rows]
    classes = Counter(row.class_name for row in rows)
    return DetectionReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        video_count=len(videos),
        detection_frame_count=len(events),
        segment_count=len(rows),
        class_counts=dict(classes),
        duration_min_sec=min(durations) if durations else 0.0,
        duration_max_sec=max(durations) if durations else 0.0,
        duration_avg_sec=round(sum(durations) / len(durations), 3) if durations else 0.0,
        rows=rows,
    )


def _bbox_label(row: DetectionSegmentReportRow) -> str:
    vals = [row.bbox_x1, row.bbox_y1, row.bbox_x2, row.bbox_y2]
    if any(value is None for value in vals):
        return "-"
    return "[" + ", ".join(f"{float(value):.1f}" for value in vals) + "]"


def _render_table_rows(rows: list[DetectionSegmentReportRow]) -> str:
    rendered = []
    for row in rows:
        rendered.append(
            "<tr>"
            f"<td>{html.escape(row.start_time)} - {html.escape(row.end_time)}</td>"
            f"<td>{row.duration_sec:.3f}s</td>"
            f"<td>{row.sample_window_sec:.3f}s</td>"
            f"<td>{row.frame_count}</td>"
            f"<td>{html.escape(row.class_name)}</td>"
            f"<td>{row.best_match_pct:.2f}%</td>"
            f"<td>{html.escape(row.best_time)}</td>"
            f"<td>{html.escape(_bbox_label(row))}</td>"
            f"<td>{row.area_px}</td>"
            f"<td>{html.escape(row.video_name)}</td>"
            "</tr>"
        )
    return "\n".join(rendered)


def render_detection_report_html(report: DetectionReport) -> str:
    top_duration = sorted(report.rows, key=lambda row: (row.duration_sec, row.best_confidence), reverse=True)[:20]
    top_confidence = sorted(report.rows, key=lambda row: (row.best_confidence, row.duration_sec), reverse=True)[:20]
    class_counts = ", ".join(f"{html.escape(name)}: {count}" for name, count in sorted(report.class_counts.items()))
    return f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <title>탐지 결과 요약 리포트</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #111827; }}
    h1 {{ margin-bottom: 8px; }}
    .muted {{ color: #6b7280; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 20px 0; }}
    .card {{ border: 1px solid #d1d5db; border-radius: 8px; padding: 12px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 7px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f4f6; }}
    code {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
  </style>
</head>
<body>
  <h1>탐지 결과 요약 리포트</h1>
  <p class="muted">생성 시각: {html.escape(report.generated_at)}</p>
  <div class="cards">
    <div class="card"><div class="value">{report.video_count}</div><div>분석 영상</div></div>
    <div class="card"><div class="value">{report.detection_frame_count}</div><div>탐지 프레임</div></div>
    <div class="card"><div class="value">{report.segment_count}</div><div>연속 탐지 구간</div></div>
    <div class="card"><div class="value">{report.duration_max_sec:.3f}s</div><div>최장 연속 시간</div></div>
  </div>
  <p>클래스별 구간 수: {class_counts or "-"}</p>
  <p>평균 연속 시간: {report.duration_avg_sec:.3f}s · 최소: {report.duration_min_sec:.3f}s · 최대: {report.duration_max_sec:.3f}s</p>

  <h2>연속 시간이 긴 대표 구간</h2>
  <table>
    <thead><tr><th>구간</th><th>연속 시간</th><th>샘플 창</th><th>프레임</th><th>클래스</th><th>최고 일치율</th><th>대표 프레임</th><th>bbox</th><th>면적</th><th>영상</th></tr></thead>
    <tbody>{_render_table_rows(top_duration)}</tbody>
  </table>

  <h2>일치율이 높은 대표 구간</h2>
  <table>
    <thead><tr><th>구간</th><th>연속 시간</th><th>샘플 창</th><th>프레임</th><th>클래스</th><th>최고 일치율</th><th>대표 프레임</th><th>bbox</th><th>면적</th><th>영상</th></tr></thead>
    <tbody>{_render_table_rows(top_confidence)}</tbody>
  </table>

  <h2>전체 연속 탐지 구간</h2>
  <table>
    <thead><tr><th>구간</th><th>연속 시간</th><th>샘플 창</th><th>프레임</th><th>클래스</th><th>최고 일치율</th><th>대표 프레임</th><th>bbox</th><th>면적</th><th>영상</th></tr></thead>
    <tbody>{_render_table_rows(report.rows)}</tbody>
  </table>
</body>
</html>
"""


def write_detection_report_bundle(timeline_path: Path, output_dir: Path) -> dict[str, Any]:
    report = build_detection_report(timeline_path)
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
        "files": {
            "html": html_path.name,
            "csv": csv_path.name,
            "json": json_path.name,
        },
    }
