"""Tests for external detection report generation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brailer_monitor.detect_report import build_detection_report, write_detection_report_bundle
from brailer_monitor.detect_timeline import compact_timeline_segments, merge_job_manifest


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
                    "models": [
                        {
                            "id": "model-a",
                            "name": "win_01280525",
                            "path": "/models/library/model-a/weights.pt",
                        }
                    ],
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
            self.assertEqual(report.source_summary, "JJR-102283_stream04_260201_040016.mp4")
            self.assertEqual(report.model_summary, "win_01280525")
            self.assertEqual(report.model_count, 1)
            self.assertEqual(report.model_details[0]["id"], "model-a")
            row = report.rows[0]
            self.assertEqual(row.best_match_pct, 95.0)
            self.assertEqual(row.best_frame_index, 15)
            self.assertEqual(row.best_segment_frame_number, 2)
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
            saved_report = build_detection_report(timeline, asset_url_prefix="/saved/result/jobs")
            self.assertEqual(saved_report.rows[0].preview_url, "/saved/result/jobs/job1/previews/frame_000015.jpg")
            self.assertEqual(saved_report.rows[0].video_url, "/saved/result/jobs/job1/video")
            self.assertEqual(len(report.timeline_frames), 2)
            self.assertEqual(report.timeline_frames[0].preview_url, "/api/pipeline/detect/job1/previews/frame_000010.jpg")
            self.assertEqual(report.timeline_frames[0].segment_frame_number, 1)
            self.assertEqual(report.timeline_frames[0].segment_frame_count, 2)
            self.assertFalse(report.timeline_frames[0].is_representative)
            self.assertEqual(report.timeline_frames[1].segment_frame_number, 2)
            self.assertTrue(report.timeline_frames[1].is_representative)

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
                    "models": [
                        {"id": "model-a", "name": "model A", "path": "/models/a.pt"},
                        {"id": "model-b", "name": "model B", "path": "/models/b.pt"},
                    ],
                    "ensemble": True,
                    "frames": [
                        {
                            "frame_index": 1,
                            "timestamp_sec": 1.0,
                            "detections": [
                                {
                                    "class_name": "brailer",
                                    "confidence": 0.8,
                                    "ensemble_model_ids": ["model-a", "model-b"],
                                    "ensemble_model_names": ["model A", "model B"],
                                }
                            ],
                            "preview_path": "frame_000001.jpg",
                        }
                    ],
                },
            )
            merge_job_manifest(
                timeline,
                job_id="dark-job",
                video_name="JJR-102283_stream04_260201_041016.mp4",
                manifest={
                    "fps": 1.0,
                    "total_frames": 10,
                    "frame_stride": 1,
                    "frames_processed": 0,
                    "frames_with_detections": 0,
                    "dark_skip_enabled": True,
                    "skipped": True,
                    "skip_reason": "dark_video",
                    "dark_video_assessment": {
                        "sample_count": 5,
                        "dark_sample_count": 5,
                        "all_samples_dark": True,
                    },
                    "frames": [],
                },
            )
            compact_timeline_segments(
                timeline,
                max_gap_sec=8,
                merge_segments=True,
                remove_size_outliers=True,
                remove_tall_thin_boxes=True,
                remove_color_outliers=True,
            )

            result = write_detection_report_bundle(timeline, root / "reports")

            files = result["files"]
            self.assertTrue((root / "reports" / files["html"]).exists())
            self.assertTrue((root / "reports" / files["csv"]).exists())
            json_path = root / "reports" / files["json"]
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["segment_count"], 1)
            self.assertEqual(payload["source_summary"], "JJR-102283_stream04_260201_040016.mp4 외 1개 영상")
            self.assertEqual(payload["model_summary"], "model A, model B")
            self.assertEqual(payload["model_count"], 2)
            self.assertEqual(payload["video_count"], 2)
            self.assertEqual(payload["dark_skip_enabled_video_count"], 1)
            self.assertEqual(payload["dark_skipped_video_count"], 1)
            self.assertTrue(payload["postprocess"]["merge_segments"])
            self.assertTrue(payload["postprocess"]["remove_size_outliers"])
            self.assertTrue(payload["postprocess"]["remove_tall_thin_boxes"])
            self.assertTrue(payload["postprocess"]["remove_color_outliers"])
            self.assertEqual(len(payload["timeline_frames"]), 1)
            self.assertEqual(payload["timeline_frames"][0]["segment_frame_number"], 1)
            self.assertEqual(payload["timeline_frames"][0]["segment_frame_count"], 1)
            self.assertEqual(result["output_dir"], str((root / "reports").resolve()))
            html_text = (root / "reports" / files["html"]).read_text(encoding="utf-8")
            self.assertIn("/api/pipeline/detect/job1/video", html_text)
            self.assertIn("전체 타임라인", html_text)
            self.assertIn("timeline-marker", html_text)
            self.assertIn("timeline-frame-marker", html_text)
            self.assertIn("timeline-preview-active", html_text)
            self.assertIn("data-timeline-zoom", html_text)
            self.assertIn("data-timeline-visible-range", html_text)
            self.assertIn("data-range-start=", html_text)
            self.assertIn("현재 화면:", html_text)
            self.assertIn("연속구간 프레임 1/1", html_text)
            self.assertIn("소스: JJR-102283_stream04_260201_040016.mp4 외 1개 영상", html_text)
            self.assertIn("사용 모델: model A, model B", html_text)
            self.assertIn("후처리: 8초 이내 구간 병합, 크기 이상 제거, 세로형 빈 그물 제거, 색상 이상 제거", html_text)
            self.assertIn("탐지 0개 제거", html_text)
            self.assertIn("어두운 영상 건너뛰기: 전체 2개 중 1개 건너뜀", html_text)
            self.assertIn("Ctrl/Command+휠", html_text)
            self.assertIn("zoomLevels", html_text)
            self.assertIn('data-preview-url="/api/pipeline/detect/job1/previews/frame_000001.jpg"', html_text)
            self.assertIn('data-preview-index="0"', html_text)
            self.assertIn("report-preview-modal", html_text)
            self.assertIn("updateTimelinePreviewHighlight", html_text)
            self.assertIn("viewport.scrollTo", html_text)
            self.assertIn("data-report-preview-prev", html_text)
            self.assertIn("data-report-preview-next", html_text)
            self.assertIn("ArrowRight", html_text)
            self.assertIn("width:0.0020%", html_text)
            self.assertIn("/api/pipeline/detect/job1/previews/frame_000001.jpg", html_text)
            self.assertIn('class="preview-thumb"', html_text)
            self.assertIn('<img src="/api/pipeline/detect/job1/previews/frame_000001.jpg"', html_text)


if __name__ == "__main__":
    unittest.main()
