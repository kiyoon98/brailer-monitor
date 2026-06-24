"""Tests for external detection report generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brailer_monitor.detect_report import build_detection_report, write_detection_report_bundle
from brailer_monitor.detect_timeline import merge_job_manifest


class DetectReportTests(unittest.TestCase):
    def test_report_uses_highest_confidence_frame_per_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            timeline = Path(tmp) / "timeline.json"
            merge_job_manifest(
                timeline,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={
                    "fps": 10.0,
                    "total_frames": 100,
                    "frame_stride": 5,
                    "frames_processed": 3,
                    "frames_with_detections": 3,
                    "frames": [
                        {
                            "frame_index": 10,
                            "timestamp_sec": 1.0,
                            "detections": [
                                {
                                    "class_name": "brailer",
                                    "confidence": 0.7,
                                    "bbox_xyxy": [1, 2, 11, 12],
                                    "area_px": 100,
                                    "mask_area_px": 100,
                                    "mask_width_px": 10,
                                    "mask_height_px": 10,
                                }
                            ],
                            "preview_path": "frame_000010.jpg",
                        },
                        {
                            "frame_index": 15,
                            "timestamp_sec": 1.5,
                            "detections": [
                                {
                                    "class_name": "brailer",
                                    "confidence": 0.95,
                                    "bbox_xyxy": [3, 4, 13, 14],
                                    "area_px": 110,
                                    "mask_area_px": 110,
                                    "mask_width_px": 11,
                                    "mask_height_px": 10,
                                }
                            ],
                            "preview_path": "frame_000015.jpg",
                        },
                    ],
                },
            )

            report = build_detection_report(timeline)

            self.assertEqual(report.segment_count, 1)
            row = report.rows[0]
            self.assertEqual(row.best_match_pct, 95.0)
            self.assertEqual(row.best_frame_index, 15)
            self.assertEqual(row.bbox_x1, 3.0)
            self.assertEqual(row.duration_sec, 0.5)
            self.assertEqual(row.sample_window_sec, 1.0)
            self.assertEqual(row.mask_area_px, 110)
            self.assertEqual(row.avg_mask_area_px, 105.0)
            self.assertEqual(row.max_mask_area_px, 110)
            self.assertEqual(row.avg_mask_width_px, 10.5)
            self.assertEqual(row.max_mask_width_px, 11)
            self.assertEqual(row.avg_mask_height_px, 10.0)
            self.assertEqual(row.preview_url, "/api/pipeline/detect/job1/previews/frame_000015.jpg")
            self.assertEqual(row.video_url, "/api/pipeline/detect/job1/video")

    def test_write_report_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            timeline = root / "timeline.json"
            merge_job_manifest(
                timeline,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={
                    "fps": 1.0,
                    "total_frames": 10,
                    "frame_stride": 1,
                    "frames_processed": 1,
                    "frames_with_detections": 1,
                    "frames": [
                        {
                            "frame_index": 1,
                            "timestamp_sec": 1.0,
                            "detections": [{"class_name": "brailer", "confidence": 0.8}],
                            "preview_path": None,
                        }
                    ],
                },
            )

            result = write_detection_report_bundle(timeline, root / "reports")

            files = result["files"]
            self.assertTrue((root / "reports" / files["html"]).exists())
            self.assertTrue((root / "reports" / files["csv"]).exists())
            json_path = root / "reports" / files["json"]
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["segment_count"], 1)
            self.assertEqual(result["output_dir"], str((root / "reports").resolve()))
            html_text = (root / "reports" / files["html"]).read_text(encoding="utf-8")
            self.assertIn("/api/pipeline/detect/job1/video", html_text)


if __name__ == "__main__":
    unittest.main()
