"""Tests for accumulated detection timeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brailer_monitor.detect_timeline import list_timeline, merge_job_manifest, reset_timeline


class DetectTimelineTests(unittest.TestCase):
    def test_merge_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "frames_processed": 2,
                "frames_with_detections": 1,
                "frames": [
                    {"frame_index": 10, "timestamp_sec": 1.0, "detections": [], "preview_path": None},
                    {
                        "frame_index": 20,
                        "timestamp_sec": 30.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.9}],
                        "preview_path": "frame_000020.jpg",
                    },
                ],
            }
            added = merge_job_manifest(
                path,
                job_id="abc123",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )
            self.assertEqual(added, 1)
            result = list_timeline(path)
            self.assertEqual(result["total"], 1)
            event = result["events"][0]
            self.assertEqual(event["absolute_time_label"], "2026-02-01 04:00:30")
            reset_timeline(path)
            self.assertEqual(list_timeline(path)["total"], 0)


if __name__ == "__main__":
    unittest.main()
