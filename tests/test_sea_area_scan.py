"""Tests for standalone sea-area scanning."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from brailer_monitor.cli import _parse_storage_hour, build_parser
from brailer_monitor.sea_area_scan import (
    format_sea_sample,
    scan_recorded_sea_area,
    scan_stream_sea_area,
)


class _FakeCapture:
    def __init__(self, frame_count: int, *, fps: float = 10.0) -> None:
        self.frames = [np.full((8, 12, 3), index, dtype=np.uint8) for index in range(frame_count)]
        self.fps = fps
        self.position = 0
        self.released = False

    def isOpened(self) -> bool:
        return not self.released

    def read(self):
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    def grab(self) -> bool:
        if self.position >= len(self.frames):
            return False
        self.position += 1
        return True

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float(len(self.frames))
        return 0.0

    def release(self) -> None:
        self.released = True


class _FakeAnalyzer:
    def analyze(self, frame: np.ndarray, *, timestamp_sec: float = 0.0) -> dict[str, object]:
        value = int(frame[0, 0, 0])
        ratio = value / 10.0
        return {
            "sea_ratio": ratio,
            "sea_percent": ratio * 100.0,
            "sea_area_px": value * 10,
            "sea_method": "test",
            "sea_horizon_y": value,
            "sea_roi_xyxy": [0, 0, 12, 8],
            "sea_candidate_area_px": value * 12,
            "sea_state": "open_sea",
            "sea_confidence": 0.9,
            "vessel_ratio": 0.0,
        }


class SeaAreaScanTests(unittest.TestCase):
    @patch("brailer_monitor.sea_area_scan._open_capture")
    def test_recorded_scan_uses_frame_stride(self, open_capture) -> None:
        capture = _FakeCapture(5, fps=10.0)
        open_capture.return_value = capture
        samples: list[dict[str, object]] = []

        summary = scan_recorded_sea_area(
            "video.mp4",
            source_name="JJR-102283_stream04_260128_030016.mp4",
            frame_stride=2,
            on_sample=samples.append,
            analyzer=_FakeAnalyzer(),
        )

        self.assertEqual([sample["frame_index"] for sample in samples], [0, 2, 4])
        self.assertEqual([sample["timestamp_sec"] for sample in samples], [0.0, 0.2, 0.4])
        self.assertEqual(samples[1]["absolute_time"], "2026-01-28 03:00:00.200")
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["frames_visited"], 5)
        self.assertEqual(summary["avg_sea_percent"], 20.0)
        self.assertTrue(capture.released)

    @patch("brailer_monitor.sea_area_scan._open_capture")
    def test_stream_scan_stops_at_max_samples(self, open_capture) -> None:
        capture = _FakeCapture(8, fps=10.0)
        open_capture.return_value = capture
        samples: list[dict[str, object]] = []

        summary = scan_stream_sea_area(
            "http://example.invalid/live.m3u8",
            frame_stride=2,
            max_samples=3,
            on_sample=samples.append,
            analyzer=_FakeAnalyzer(),
        )

        self.assertEqual([sample["frame_index"] for sample in samples], [0, 2, 4])
        self.assertEqual(summary["samples"], 3)
        self.assertEqual(summary["end_reason"], "max_samples")
        self.assertEqual(summary["reconnects"], 0)
        self.assertTrue(capture.released)

    def test_sample_formatter_contains_frame_metrics(self) -> None:
        line = format_sea_sample(
            {
                "source_name": "sample.mp4",
                "frame_index": 30,
                "timestamp_sec": 2.0,
                "absolute_time": "2026-01-28 03:00:02.000",
                "sea_percent": 42.5,
                "sea_area_px": 100,
                "sea_horizon_y": 12,
                "sea_roi_xyxy": [0, 10, 100, 80],
                "processing_ms": 5.5,
            }
        )

        self.assertIn("frame=30", line)
        self.assertIn("sea=42.50%", line)
        self.assertIn("processing_ms=5.5", line)

    def test_cli_selects_storage_or_stream(self) -> None:
        parser = build_parser()
        storage = parser.parse_args(
            ["sea-area", "storage", "--url", "http://example.invalid/video.mp4", "--frame-stride", "15"]
        )
        stream = parser.parse_args(["sea-area", "stream", "--max-samples", "2"])

        self.assertEqual(storage.sea_source, "storage")
        self.assertEqual(storage.frame_stride, 15)
        self.assertEqual(stream.sea_source, "stream")
        self.assertEqual(stream.url, "http://127.0.0.1:8081/live_04.m3u8")
        self.assertEqual(stream.device, "cpu")

    def test_storage_hour_accepts_full_or_short_date(self) -> None:
        self.assertEqual(_parse_storage_hour("2026-01-28T03", default_year=2025).year, 2026)
        self.assertEqual(_parse_storage_hour("01-28T03", default_year=2026).year, 2026)


if __name__ == "__main__":
    unittest.main()
