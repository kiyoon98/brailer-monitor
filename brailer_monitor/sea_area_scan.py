"""Standalone sea-area scanning for recorded videos and live streams."""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2

from .sea_area_analysis import SeaAreaAnalyzer
from .video_time import absolute_frame_time


SeaSampleHandler = Callable[[dict[str, Any]], None]


def _validate_scan_options(
    frame_stride: int,
    max_samples: int | None,
    duration_sec: float | None = None,
) -> None:
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if max_samples is not None and max_samples < 1:
        raise ValueError("max_samples must be at least 1")
    if duration_sec is not None and duration_sec <= 0:
        raise ValueError("duration_sec must be greater than 0")


def _open_capture(source: str) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(source)
    if capture.isOpened():
        return capture
    capture.release()
    raise RuntimeError(f"Cannot open video source: {source}")


def _capture_fps(capture: cv2.VideoCapture, fallback: float = 15.0) -> float:
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    return fps if fps > 0 else fallback


def _make_sample(
    *,
    source_type: str,
    source: str,
    source_name: str,
    frame_index: int,
    timestamp_sec: float,
    frame: Any,
    analyzer: SeaAreaAnalyzer,
) -> dict[str, Any]:
    absolute_time = absolute_frame_time(source_name, timestamp_sec)
    analysis_timestamp = (
        absolute_time.timestamp() if source_type != "stream" and absolute_time is not None else timestamp_sec
    )
    started = time.perf_counter()
    stats = analyzer.analyze(frame, timestamp_sec=analysis_timestamp)
    processing_ms = (time.perf_counter() - started) * 1000.0
    height, width = frame.shape[:2]
    if source_type == "stream":
        absolute_time_label = datetime.now().astimezone().isoformat(timespec="milliseconds")
    else:
        absolute_time_label = absolute_time.isoformat(sep=" ", timespec="milliseconds") if absolute_time else None
    return {
        "type": "frame",
        "source_type": source_type,
        "source": source,
        "source_name": source_name,
        "frame_index": int(frame_index),
        "timestamp_sec": round(float(timestamp_sec), 3),
        "absolute_time": absolute_time_label,
        "width": int(width),
        "height": int(height),
        "processing_ms": round(processing_ms, 1),
        **stats,
    }


class _SeaSummary:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum: float | None = None
        self.maximum: float | None = None
        self.states: dict[str, int] = {}
        self.events: list[dict[str, Any]] = []
        self.confidence_total = 0.0
        self.confidence_count = 0

    def add(self, sample: dict[str, Any]) -> None:
        state = str(sample.get("sea_state") or "unknown")
        ratio_value = sample.get("sea_ratio")
        ratio = float(ratio_value) if ratio_value is not None else 0.0
        self.count += 1
        if state != "unknown" and ratio_value is not None:
            self.total += ratio
            self.minimum = ratio if self.minimum is None else min(self.minimum, ratio)
            self.maximum = ratio if self.maximum is None else max(self.maximum, ratio)
        self.states[state] = self.states.get(state, 0) + 1
        if sample.get("sea_event"):
            self.events.append(
                {
                    "event": sample["sea_event"],
                    "frame_index": sample["frame_index"],
                    "timestamp_sec": sample["timestamp_sec"],
                }
            )
        if state != "unknown" and sample.get("sea_confidence") is not None:
            self.confidence_total += float(sample["sea_confidence"])
            self.confidence_count += 1

    def to_dict(self) -> dict[str, Any]:
        valid_count = self.count - self.states.get("unknown", 0)
        average = self.total / valid_count if valid_count else None
        return {
            "samples": self.count,
            "avg_sea_ratio": round(average, 4) if average is not None else None,
            "avg_sea_percent": round(average * 100.0, 2) if average is not None else None,
            "min_sea_ratio": round(self.minimum, 4) if self.minimum is not None else None,
            "max_sea_ratio": round(self.maximum, 4) if self.maximum is not None else None,
            "state_counts": self.states,
            "unknown_count": self.states.get("unknown", 0),
            "avg_sea_confidence": (
                round(self.confidence_total / self.confidence_count, 4) if self.confidence_count else None
            ),
            "events": self.events,
        }


def scan_recorded_sea_area(
    source: str | Path,
    *,
    source_name: str | None = None,
    frame_stride: int = 30,
    max_samples: int | None = None,
    on_sample: SeaSampleHandler | None = None,
    analyzer: SeaAreaAnalyzer | None = None,
    sea_engine: str = "hybrid",
    device: str | int = "cpu",
) -> dict[str, Any]:
    """Sample a recorded video and emit sea-area statistics per sampled frame."""
    _validate_scan_options(frame_stride, max_samples)
    source_text = str(source)
    display_name = source_name or Path(source_text).name or source_text
    capture = _open_capture(source_text)
    analyzer = analyzer or SeaAreaAnalyzer(device=device, engine=sea_engine)
    summary = _SeaSummary()
    fps = _capture_fps(capture)
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_index = 0
    frames_visited = 0
    end_reason = "eof"

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            frames_visited += 1
            sample = _make_sample(
                source_type="storage",
                source=source_text,
                source_name=display_name,
                frame_index=frame_index,
                timestamp_sec=frame_index / fps,
                frame=frame,
                analyzer=analyzer,
            )
            summary.add(sample)
            if on_sample is not None:
                on_sample(sample)
            if max_samples is not None and summary.count >= max_samples:
                end_reason = "max_samples"
                break

            reached_end = False
            skipped = 0
            for _ in range(frame_stride - 1):
                if not capture.grab():
                    reached_end = True
                    break
                skipped += 1
            frames_visited += skipped
            frame_index += skipped + 1
            if reached_end:
                break
    finally:
        capture.release()

    return {
        "type": "summary",
        "source_type": "storage",
        "source": source_text,
        "source_name": display_name,
        "fps": round(fps, 3),
        "total_frames": total_frames,
        "frame_stride": frame_stride,
        "frames_visited": frames_visited,
        "end_reason": end_reason,
        **summary.to_dict(),
    }


def scan_stream_sea_area(
    stream_url: str,
    *,
    frame_stride: int = 30,
    max_samples: int | None = None,
    duration_sec: float | None = None,
    on_sample: SeaSampleHandler | None = None,
    reconnect_after_failures: int = 15,
    reconnect_delay_sec: float = 1.0,
    analyzer: SeaAreaAnalyzer | None = None,
    sea_engine: str = "hybrid",
    device: str | int = "cpu",
) -> dict[str, Any]:
    """Continuously sample a live stream, reconnecting after repeated read failures."""
    _validate_scan_options(frame_stride, max_samples, duration_sec)
    reconnect_after_failures = max(1, int(reconnect_after_failures))
    capture = _open_capture(stream_url)
    analyzer = analyzer or SeaAreaAnalyzer(device=device, engine=sea_engine)
    summary = _SeaSummary()
    fps = _capture_fps(capture)
    frame_index = 0
    successful_frames = 0
    failed_reads = 0
    reconnects = 0
    started = time.monotonic()
    end_reason = "stopped"

    try:
        while True:
            elapsed = time.monotonic() - started
            if duration_sec is not None and elapsed >= duration_sec:
                end_reason = "duration"
                break

            ok, frame = capture.read()
            if not ok or frame is None:
                failed_reads += 1
                if failed_reads < reconnect_after_failures:
                    time.sleep(0.2)
                    continue

                capture.release()
                while True:
                    if duration_sec is not None and time.monotonic() - started >= duration_sec:
                        end_reason = "duration"
                        break
                    try:
                        capture = _open_capture(stream_url)
                        fps = _capture_fps(capture, fallback=fps)
                        reconnects += 1
                        failed_reads = 0
                        break
                    except RuntimeError:
                        time.sleep(max(0.1, reconnect_delay_sec))
                if end_reason == "duration":
                    break
                continue

            failed_reads = 0
            successful_frames += 1
            if frame_index % frame_stride == 0:
                sample = _make_sample(
                    source_type="stream",
                    source=stream_url,
                    source_name=stream_url,
                    frame_index=frame_index,
                    timestamp_sec=time.monotonic() - started,
                    frame=frame,
                    analyzer=analyzer,
                )
                summary.add(sample)
                if on_sample is not None:
                    on_sample(sample)
                if max_samples is not None and summary.count >= max_samples:
                    end_reason = "max_samples"
                    break
            frame_index += 1
    finally:
        capture.release()

    return {
        "type": "summary",
        "source_type": "stream",
        "source": stream_url,
        "source_name": stream_url,
        "fps": round(fps, 3),
        "total_frames": 0,
        "frame_stride": frame_stride,
        "frames_visited": successful_frames,
        "duration_sec": round(time.monotonic() - started, 3),
        "reconnects": reconnects,
        "end_reason": end_reason,
        **summary.to_dict(),
    }


def format_sea_sample(sample: dict[str, Any]) -> str:
    """Format one sample as a compact, human-readable progress line."""
    timestamp = float(sample.get("timestamp_sec") or 0.0)
    hours = int(timestamp // 3600)
    minutes = int((timestamp % 3600) // 60)
    seconds = timestamp % 60
    relative_time = f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
    absolute_time = sample.get("absolute_time") or "-"
    roi = sample.get("sea_roi_xyxy")
    sea_percent = sample.get("sea_percent")
    sea_label = f"{float(sea_percent):.2f}%" if sea_percent is not None else "--"
    confidence = sample.get("sea_confidence")
    confidence_label = f"{float(confidence):.3f}" if confidence is not None else "--"
    vessel_increase = sample.get("vessel_increase_ratio")
    increase_label = f"+{float(vessel_increase) * 100.0:.2f}%" if vessel_increase is not None else "--"
    return (
        f"[{sample.get('source_name')}] "
        f"frame={int(sample.get('frame_index') or 0)} "
        f"time={relative_time} absolute={absolute_time} "
        f"sea={sea_label} "
        f"state={sample.get('sea_state') or '-'} "
        f"confidence={confidence_label} "
        f"vessel={float(sample.get('vessel_ratio') or 0.0) * 100.0:.2f}% "
        f"vessel_increase={increase_label} "
        f"area_px={int(sample.get('sea_area_px') or 0)} "
        f"horizon_y={sample.get('sea_horizon_y')} roi={roi} "
        f"processing_ms={float(sample.get('processing_ms') or 0.0):.1f}"
    )


def format_sea_summary(summary: dict[str, Any]) -> str:
    """Format a source summary for terminal output."""
    average = summary.get("avg_sea_percent")
    min_ratio = summary.get("min_sea_ratio")
    max_ratio = summary.get("max_sea_ratio")
    minimum = float(min_ratio) * 100.0 if min_ratio is not None else 0.0
    maximum = float(max_ratio) * 100.0 if max_ratio is not None else 0.0
    return (
        f"SUMMARY [{summary.get('source_name')}] "
        f"samples={int(summary.get('samples') or 0)} "
        f"avg={float(average or 0.0):.2f}% min={minimum:.2f}% max={maximum:.2f}% "
        f"frames_visited={int(summary.get('frames_visited') or 0)} "
        f"end={summary.get('end_reason')}"
    )
