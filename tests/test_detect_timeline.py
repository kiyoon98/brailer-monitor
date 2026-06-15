"""Tests for accumulated detection timeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brailer_monitor.detect_timeline import (
    build_segments,
    detection_area_px,
    get_segment_frames,
    list_timeline,
    merge_job_manifest,
    reset_timeline,
    timeline_range,
)


class DetectTimelineTests(unittest.TestCase):
    def test_merge_and_list(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 30.0,
                "total_frames": 900,
                "frame_stride": 5,
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
            segment = result["segments"][0]
            self.assertEqual(segment["start_absolute_time_label"], "2026-02-01 04:00:30")
            self.assertEqual(result["range_start_label"], "2026-02-01 04:00:00")
            self.assertEqual(result["range_end_label"], "2026-02-01 04:00:30")
            video = result["videos"][0]
            self.assertEqual(video["duration_sec"], 30.0)
            reset_timeline(path)
            self.assertEqual(list_timeline(path)["total"], 0)

    def test_merge_consecutive_frames_into_one_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 1.0,
                "total_frames": 100,
                "frame_stride": 5,
                "frames_processed": 3,
                "frames_with_detections": 3,
                "frames": [
                    {
                        "frame_index": 10,
                        "timestamp_sec": 10.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.85}],
                        "preview_path": "frame_000015.jpg",
                    },
                    {
                        "frame_index": 40,
                        "timestamp_sec": 40.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.9}],
                        "preview_path": "frame_000040.jpg",
                    },
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )
            segments = build_segments(__import__("json").loads(path.read_text(encoding="utf-8")))
            self.assertEqual(len(segments), 2)
            self.assertEqual(segments[0]["frame_count"], 2)
            self.assertEqual(segments[1]["frame_count"], 1)

            detail = get_segment_frames(path, segments[0]["segment_id"])
            self.assertEqual(len(detail["frames"]), 2)

    def test_timeline_range_from_video_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            for idx, name in enumerate(
                [
                    "JJR-102283_stream04_260201_040016.mp4",
                    "JJR-102283_stream04_260201_050016.mp4",
                ]
            ):
                merge_job_manifest(
                    path,
                    job_id=f"job{idx}",
                    video_name=name,
                    manifest={
                        "fps": 1.0,
                        "total_frames": 3600,
                        "frame_stride": 5,
                        "frames_processed": 0,
                        "frames_with_detections": 0,
                        "frames": [],
                    },
                )
            summary = timeline_range(__import__("json").loads(path.read_text(encoding="utf-8")))
            self.assertEqual(summary["range_start_label"], "2026-02-01 04:00:00")
            self.assertEqual(summary["range_end_label"], "2026-02-01 06:00:00")

    def test_detection_area_and_segment_stats(self) -> None:
        self.assertEqual(
            detection_area_px({"bbox_xyxy": [10, 20, 110, 70]}),
            5000,
        )
        self.assertEqual(
            detection_area_px({"area_px": 12340, "bbox_xyxy": [0, 0, 1, 1]}),
            12340,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 1.0,
                "total_frames": 100,
                "frame_stride": 5,
                "frames_processed": 2,
                "frames_with_detections": 2,
                "frames": [
                    {
                        "frame_index": 10,
                        "timestamp_sec": 10.0,
                        "detections": [
                            {
                                "class_name": "brailer",
                                "confidence": 0.8,
                                "bbox_xyxy": [0, 0, 100, 50],
                            }
                        ],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [
                            {
                                "class_name": "brailer",
                                "confidence": 0.9,
                                "area_px": 12000,
                            }
                        ],
                        "preview_path": "frame_000015.jpg",
                    },
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )
            result = list_timeline(path)
            segment = result["segments"][0]
            self.assertEqual(segment["max_confidence"], 0.9)
            self.assertEqual(segment["max_area_px"], 12000)

            detail = get_segment_frames(path, segment["segment_id"])
            areas = {int(f["area_px"]) for f in detail["frames"]}
            self.assertEqual(areas, {5000, 12000})


if __name__ == "__main__":
    unittest.main()
