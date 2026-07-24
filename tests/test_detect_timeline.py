"""Tests for accumulated detection timeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brailer_monitor.detect_timeline import (
    _is_uniform_bright_color,
    _sparse_static_work_outlier_keys,
    build_segments,
    compact_timeline_segments,
    detection_area_px,
    get_segment_frames,
    load_timeline,
    list_timeline,
    merge_frame_detection,
    merge_job_manifest,
    reset_timeline,
    timeline_range,
)


class DetectTimelineTests(unittest.TestCase):
    def test_sparse_static_work_outlier_requires_work_level_position_anomaly(self) -> None:
        entries = []
        for index in range(10):
            entries.append(
                {
                    "key": (index, 0),
                    "work_group": ("camera", "brailer"),
                    "center": (100.0 + index % 2, 100.0),
                    "width": 40.0,
                    "height": 40.0,
                    "diag": 56.6,
                    "absolute_timestamp": float(index),
                    "event": {"video_name": f"camera_260314_000{index:02d}.mp4"},
                }
            )
        expected = set()
        for index in range(6):
            key = (10 + index, 0)
            expected.add(key)
            entries.append(
                {
                    "key": key,
                    "work_group": ("camera", "brailer"),
                    "center": (1018.0 + index % 2, 444.0 + index % 3),
                    "width": 120.0,
                    "height": 154.0,
                    "diag": 195.2,
                    "absolute_timestamp": float(index * 60),
                    "event": {"video_name": f"camera_260314_01{index // 2:02d}04.mp4"},
                }
            )

        self.assertEqual(_sparse_static_work_outlier_keys(entries), expected)

    def test_uniform_bright_color_accepts_adjusted_boundary(self) -> None:
        base = {"mean_gray": 184.0, "median_gray": 206.0}
        self.assertTrue(
            _is_uniform_bright_color({**base, "bright_uniform_ratio": 0.7393})
        )
        self.assertFalse(
            _is_uniform_bright_color({**base, "bright_uniform_ratio": 0.7199})
        )

    def test_manifest_preserves_full_sea_analysis_and_encounter_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            frames = [
                {
                    "frame_index": 0,
                    "timestamp_sec": 0.0,
                    "detections": [],
                    "sea_ratio": 0.8,
                    "sea_quality": "good",
                    "sea_state": "open_sea",
                    "sea_confidence": 0.9,
                    "vessel_ratio": 0.0,
                    "sea_method": "hybrid-test",
                },
                {
                    "frame_index": 10,
                    "timestamp_sec": 10.0,
                    "detections": [],
                    "sea_ratio": 0.5,
                    "sea_quality": "good",
                    "sea_state": "encounter",
                    "sea_event": "encounter_start",
                    "sea_confidence": 0.8,
                    "vessel_ratio": 0.02,
                    "sea_method": "hybrid-test",
                },
                {
                    "frame_index": 20,
                    "timestamp_sec": 20.0,
                    "detections": [],
                    "sea_ratio": 0.45,
                    "sea_quality": "good",
                    "sea_state": "encounter",
                    "sea_confidence": 0.75,
                    "vessel_ratio": 0.03,
                    "sea_method": "hybrid-test",
                },
                {
                    "frame_index": 50,
                    "timestamp_sec": 50.0,
                    "detections": [],
                    "sea_ratio": 0.79,
                    "sea_quality": "good",
                    "sea_state": "open_sea",
                    "sea_event": "departure",
                    "sea_confidence": 0.88,
                    "vessel_ratio": 0.0,
                    "sea_method": "hybrid-test",
                },
            ]
            merge_job_manifest(
                path,
                job_id="job-sea",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={
                    "sea_ratio_enabled": True,
                    "sea_only": True,
                    "object_detection_enabled": False,
                    "sea_engine": "hybrid",
                    "sea_analysis_interval_sec": 5.0,
                    "fps": 1.0,
                    "total_frames": 60,
                    "frames_processed": len(frames),
                    "frames": frames,
                },
            )

            result = list_timeline(path)
            analysis = result["videos"][0]["sea_analysis"]

            self.assertEqual(analysis["sample_count"], 4)
            self.assertTrue(result["videos"][0]["sea_only"])
            self.assertFalse(result["videos"][0]["object_detection_enabled"])
            self.assertEqual(result["videos"][0]["sea_analysis_interval_sec"], 5.0)
            self.assertEqual(analysis["unknown_count"], 0)
            self.assertEqual(analysis["methods"], ["hybrid-test"])
            self.assertEqual(len(analysis["encounter_segments"]), 1)
            encounter = analysis["encounter_segments"][0]
            self.assertEqual(encounter["start_time"], "2026-02-01 04:00:10")
            self.assertEqual(encounter["end_time"], "2026-02-01 04:00:50")
            self.assertEqual(encounter["duration_sec"], 40.0)
            self.assertEqual(encounter["min_sea_ratio"], 0.45)
            self.assertEqual(encounter["max_vessel_ratio"], 0.03)

    def test_load_timeline_ignores_stale_trailing_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            path.write_text(
                '{"updated_at": "now", "videos": [], "events": []}  }\\n  ]\\n}',
                encoding="utf-8",
            )

            timeline = load_timeline(path)

            self.assertEqual(timeline["videos"], [])
            self.assertEqual(timeline["events"], [])

    def test_load_timeline_treats_empty_file_as_empty_timeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            path.write_text("", encoding="utf-8")

            timeline = load_timeline(path)

            self.assertEqual(timeline["videos"], [])
            self.assertEqual(timeline["events"], [])

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
                        "sea_ratio": 0.25,
                        "sea_percent": 25.0,
                        "sea_area_px": 100,
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
            self.assertEqual(segment["avg_sea_ratio"], 0.25)
            self.assertEqual(segment["avg_sea_percent"], 25.0)
            self.assertEqual(result["range_start_label"], "2026-02-01 04:00:00")
            self.assertEqual(result["range_end_label"], "2026-02-01 04:00:30")
            video = result["videos"][0]
            self.assertEqual(video["duration_sec"], 30.0)
            reset_timeline(path)
            self.assertEqual(list_timeline(path)["total"], 0)

    def test_merge_job_manifest_preserves_dark_skip_video_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            merge_job_manifest(
                path,
                job_id="dark-job",
                video_name="JJR-102283_stream04_260201_041016.mp4",
                manifest={
                    "fps": 30.0,
                    "total_frames": 900,
                    "frame_stride": 5,
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

            timeline = load_timeline(path)
            self.assertEqual(len(timeline["videos"]), 1)
            video = timeline["videos"][0]
            self.assertTrue(video["dark_skip_enabled"])
            self.assertTrue(video["skipped"])
            self.assertEqual(video["skip_reason"], "dark_video")
            self.assertEqual(video["dark_video_assessment"]["dark_sample_count"], 5)
            self.assertEqual(len(timeline["events"]), 0)

    def test_incremental_frame_merge_is_replaced_by_final_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            frame = {
                "frame_index": 20,
                "timestamp_sec": 30.0,
                "detections": [{"class_name": "brailer", "confidence": 0.9}],
                "preview_path": "frame_000020.jpg",
            }
            manifest_meta = {
                "fps": 30.0,
                "total_frames": 900,
                "frame_stride": 5,
                "frames_processed": 1,
                "frames_with_detections": 1,
            }

            added = merge_frame_detection(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                frame=frame,
                manifest=manifest_meta,
            )
            duplicate = merge_frame_detection(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                frame=frame,
                manifest=manifest_meta,
            )

            self.assertEqual(added, 1)
            self.assertEqual(duplicate, 0)
            timeline = load_timeline(path)
            self.assertEqual(len(timeline["events"]), 1)
            self.assertEqual(len(timeline["videos"]), 1)

            manifest = {
                **manifest_meta,
                "frames": [
                    {"frame_index": 10, "timestamp_sec": 15.0, "detections": [], "preview_path": None},
                    frame,
                ],
            }
            final_added = merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
                replace_job=True,
            )

            timeline = load_timeline(path)
            self.assertEqual(final_added, 1)
            self.assertEqual(len(timeline["events"]), 1)
            self.assertEqual(len(timeline["videos"]), 1)

    def test_reprocessing_same_video_replaces_previous_video_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            video_name = "JJR-102283_stream04_260201_040016.mp4"
            first_manifest = {
                "fps": 1.0,
                "total_frames": 100,
                "frame_stride": 5,
                "frames_processed": 1,
                "frames_with_detections": 1,
                "frames": [
                    {
                        "frame_index": 10,
                        "timestamp_sec": 10.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8}],
                        "preview_path": "frame_000010.jpg",
                    }
                ],
            }
            second_manifest = {
                **first_manifest,
                "frames": [
                    {
                        "frame_index": 20,
                        "timestamp_sec": 20.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.9}],
                        "preview_path": "frame_000020.jpg",
                    }
                ],
            }

            merge_job_manifest(path, job_id="old", video_name=video_name, manifest=first_manifest)
            merge_job_manifest(
                path,
                job_id="new",
                video_name=video_name,
                manifest=second_manifest,
                replace_video=True,
            )

            timeline = load_timeline(path)
            self.assertEqual(len(timeline["events"]), 1)
            self.assertEqual(timeline["events"][0]["job_id"], "new")
            self.assertEqual(timeline["events"][0]["frame_index"], 20)
            self.assertEqual(len(timeline["videos"]), 1)
            self.assertEqual(timeline["videos"][0]["job_id"], "new")

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

    def test_compact_segments_merges_gaps_up_to_ten_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 1.0,
                "total_frames": 100,
                "frame_stride": 1,
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
                        "frame_index": 14,
                        "timestamp_sec": 14.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.9}],
                        "preview_path": "frame_000014.jpg",
                    },
                    {
                        "frame_index": 20,
                        "timestamp_sec": 20.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.7}],
                        "preview_path": "frame_000020.jpg",
                    },
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )
            self.assertEqual(len(build_segments(__import__("json").loads(path.read_text(encoding="utf-8")))), 3)

            result = compact_timeline_segments(path, max_gap_sec=8)
            self.assertEqual(result["before_segment_count"], 3)
            self.assertEqual(result["segment_count"], 1)
            self.assertEqual(result["merged_segment_count"], 2)
            self.assertTrue(result["postprocess"]["merge_segments"])
            self.assertEqual(result["postprocess"]["segment_merge_gap_sec"], 8.0)
            self.assertEqual(result["postprocess"]["merged_segment_count"], 2)

            listed = list_timeline(path)
            self.assertEqual(listed["total"], 1)
            self.assertEqual(listed["segment_merge_gap_sec"], 8.0)
            self.assertEqual(listed["segments"][0]["frame_count"], 3)
            timeline = load_timeline(path)
            self.assertEqual(timeline["postprocess"]["segment_count"], 1)

    def test_postprocess_removes_size_outlier_detections(self) -> None:
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
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [0, 0, 10, 10]}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [0, 0, 10, 10]}],
                        "preview_path": "frame_000015.jpg",
                    },
                    {
                        "frame_index": 20,
                        "timestamp_sec": 20.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [0, 0, 30, 30]}],
                        "preview_path": "frame_000020.jpg",
                    },
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_size_outliers=True,
            )

            self.assertEqual(result["removed_detection_count"], 1)
            self.assertEqual(result["removed_event_count"], 1)
            self.assertEqual(result["event_count"], 2)
            self.assertEqual(result["removed_by_condition"]["size_outlier"], 1)

    def test_postprocess_removes_tall_thin_box_detections(self) -> None:
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
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [0, 0, 20, 20]}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [0, 0, 20, 60]}],
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

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_tall_thin_boxes=True,
            )

            self.assertEqual(result["removed_detection_count"], 1)
            self.assertEqual(result["removed_event_count"], 1)
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["removed_by_condition"]["tall_thin_box"], 1)

    def test_postprocess_removes_repeated_large_lower_sea_regions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            normal_frames = [
                {
                    "frame_index": index,
                    "timestamp_sec": float(index),
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": [100, 100, 140, 140],
                        }
                    ],
                    "preview_path": f"frame_{index:06d}.jpg",
                }
                for index in range(10, 100, 10)
            ]
            normal_frames.append(
                {
                    "frame_index": 100,
                    "timestamp_sec": 100.0,
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": [700, 470, 1000, 720],
                        }
                    ],
                    "preview_path": "frame_000100.jpg",
                }
            )
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={
                    "fps": 1.0,
                    "width": 1280,
                    "height": 720,
                    "total_frames": 120,
                    "frame_stride": 5,
                    "frames": normal_frames,
                },
            )
            merge_job_manifest(
                path,
                job_id="job2",
                video_name="JJR-102283_stream04_260201_060016.mp4",
                manifest={
                    "fps": 1.0,
                    "width": 1280,
                    "height": 720,
                    "total_frames": 120,
                    "frame_stride": 5,
                    "frames": [
                        {
                            "frame_index": 10,
                            "timestamp_sec": 10.0,
                            "detections": [
                                {
                                    "class_name": "brailer",
                                    "confidence": 0.8,
                                    "bbox_xyxy": [704, 468, 1004, 718],
                                }
                            ],
                            "preview_path": "frame_000010.jpg",
                        }
                    ],
                },
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_large_lower_sea_regions=True,
                remove_position_outliers=True,
                remove_size_outliers=True,
            )

            self.assertEqual(result["removed_detection_count"], 2)
            self.assertEqual(result["event_count"], 9)
            self.assertEqual(result["removed_by_condition"]["large_lower_sea_region"], 2)
            self.assertEqual(result["protected_by_condition"]["large_lower_sea_region"], 0)
            self.assertEqual(result["postprocess"]["large_lower_sea_area_median_ratio"], 4.0)

    def test_postprocess_removes_side_edge_detections(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 1.0,
                "width": 1280,
                "height": 720,
                "total_frames": 100,
                "frame_stride": 5,
                "frames_processed": 3,
                "frames_with_detections": 3,
                "frames": [
                    {
                        "frame_index": 10,
                        "timestamp_sec": 10.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [500, 100, 650, 250]}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [1125, 155, 1268, 330]}],
                        "preview_path": "frame_000015.jpg",
                    },
                    {
                        "frame_index": 20,
                        "timestamp_sec": 20.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [10, 155, 150, 330]}],
                        "preview_path": "frame_000020.jpg",
                    },
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_right_edge_detections=True,
            )

            self.assertEqual(result["removed_detection_count"], 2)
            self.assertEqual(result["removed_event_count"], 2)
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["removed_by_condition"]["right_edge"], 2)
            self.assertTrue(result["postprocess"]["remove_right_edge_detections"])
            self.assertEqual(result["postprocess"]["right_edge_center_x_ratio"], 0.85)
            self.assertEqual(result["postprocess"]["edge_side_x_ratio"], 0.985)

    def test_postprocess_removes_static_three_to_four_second_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 1.0,
                "total_frames": 100,
                "frame_stride": 1,
                "frames_processed": 4,
                "frames_with_detections": 4,
                "frames": [
                    {
                        "frame_index": idx,
                        "timestamp_sec": float(idx),
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [100, 100, 140, 140]}],
                        "preview_path": f"frame_{idx:06d}.jpg",
                    }
                    for idx in (10, 11, 12, 13)
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_static_short_tracks=True,
            )

            self.assertEqual(result["removed_detection_count"], 4)
            self.assertEqual(result["removed_event_count"], 4)
            self.assertEqual(result["event_count"], 0)
            self.assertEqual(result["removed_by_condition"]["static_short_track"], 4)

    def test_postprocess_removes_short_nearly_frozen_track(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 15.0,
                "total_frames": 100,
                "frame_stride": 5,
                "frames_processed": 3,
                "frames_with_detections": 3,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": frame_index / 15.0,
                        "detections": [
                            {
                                "class_name": "brailer",
                                "confidence": 0.8,
                                "bbox_xyxy": [100 + drift, 100, 180 + drift, 170],
                            }
                        ],
                        "preview_path": f"frame_{frame_index:06d}.jpg",
                    }
                    for frame_index, drift in ((10, 0.0), (20, 0.5), (30, -0.5))
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="LAKE_AURORA_stream01_260307_230304.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_static_short_tracks=True,
            )

            self.assertEqual(result["removed_detection_count"], 3)
            self.assertEqual(result["event_count"], 0)
            self.assertEqual(result["removed_by_condition"]["static_short_track"], 3)
            self.assertEqual(result["postprocess"]["static_position_strict_min_frames"], 3)

    def test_postprocess_preserves_short_track_with_visible_motion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 15.0,
                "total_frames": 100,
                "frame_stride": 5,
                "frames_processed": 3,
                "frames_with_detections": 3,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "timestamp_sec": frame_index / 15.0,
                        "detections": [
                            {
                                "class_name": "brailer",
                                "confidence": 0.8,
                                "bbox_xyxy": [100 + drift, 100, 180 + drift, 170],
                            }
                        ],
                        "preview_path": f"frame_{frame_index:06d}.jpg",
                    }
                    for frame_index, drift in ((10, 0.0), (20, 8.0), (30, 16.0))
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="LAKE_AURORA_stream01_260307_230304.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_static_short_tracks=True,
            )

            self.assertEqual(result["removed_detection_count"], 0)
            self.assertEqual(result["event_count"], 3)

    def test_postprocess_removes_same_position_runs_outside_three_to_four_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            first_run = [75, 80, 90, 95, 110, 115]
            second_run = [
                400,
                405,
                410,
                415,
                420,
                425,
                430,
                435,
                440,
                445,
                460,
                465,
                470,
                530,
                625,
                695,
                720,
                735,
                740,
                805,
                810,
                815,
                830,
                835,
                840,
                845,
                850,
                855,
                860,
                880,
                925,
                1015,
                1045,
            ]

            def frame_payload(frame_index: int) -> dict:
                drift = ((frame_index // 5) % 9) - 4
                cx = 690 + drift * 2.0
                return {
                    "frame_index": frame_index,
                    "timestamp_sec": round(frame_index / 15.0, 3),
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": [cx - 50.0, 570.0, cx + 50.0, 720.0],
                        }
                    ],
                    "preview_path": f"frame_{frame_index:06d}.jpg",
                }

            frames = [frame_payload(idx) for idx in first_run + second_run]
            manifest = {
                "fps": 15.0,
                "total_frames": 1200,
                "frame_stride": 5,
                "frames_processed": len(frames),
                "frames_with_detections": len(frames),
                "frames": frames,
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260128_030016.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_static_short_tracks=True,
            )

            self.assertEqual(result["removed_detection_count"], len(frames))
            self.assertEqual(result["removed_event_count"], len(frames))
            self.assertEqual(result["event_count"], 0)
            self.assertEqual(result["removed_by_condition"]["static_short_track"], len(frames))
            self.assertEqual(result["postprocess"]["static_position_min_frames"], 6)
            self.assertEqual(result["postprocess"]["static_position_max_gap_sec"], 8.0)

    def test_postprocess_removes_temporally_isolated_detections(self) -> None:
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
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [100, 100, 140, 140]}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [106, 102, 146, 142]}],
                        "preview_path": "frame_000015.jpg",
                    },
                    {
                        "frame_index": 40,
                        "timestamp_sec": 40.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [300, 300, 340, 340]}],
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

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_temporal_isolated=True,
            )

            self.assertEqual(result["removed_detection_count"], 1)
            self.assertEqual(result["removed_event_count"], 1)
            self.assertEqual(result["event_count"], 2)
            self.assertEqual(result["removed_by_condition"]["temporal_isolated"], 1)
            self.assertTrue(result["postprocess"]["remove_temporal_isolated"])
            self.assertEqual(result["postprocess"]["temporal_isolation_window_sec"], 10.0)

    def test_postprocess_removes_short_three_frame_temporal_burst(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 10.0,
                "total_frames": 100,
                "frame_stride": 3,
                "frames_processed": 3,
                "frames_with_detections": 3,
                "frames": [
                    {
                        "frame_index": idx,
                        "timestamp_sec": ts,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [100, 100, 140, 140]}],
                        "preview_path": f"frame_{idx:06d}.jpg",
                    }
                    for idx, ts in ((10, 1.0), (13, 1.3), (16, 1.6))
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_temporal_isolated=True,
            )

            self.assertEqual(result["removed_detection_count"], 3)
            self.assertEqual(result["removed_event_count"], 3)
            self.assertEqual(result["event_count"], 0)
            self.assertEqual(result["removed_by_condition"]["temporal_isolated"], 3)
            self.assertEqual(result["postprocess"]["temporal_short_burst_max_frames"], 3)
            self.assertEqual(result["postprocess"]["temporal_short_burst_max_duration_sec"], 1.0)

    def test_temporal_isolation_preserves_similar_position_in_same_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            for job_id, video_name, bbox in (
                ("job1", "JJR-102283_stream04_260201_040016.mp4", [100, 100, 140, 140]),
                ("job2", "JJR-102283_stream04_260201_060016.mp4", [106, 102, 146, 142]),
            ):
                merge_job_manifest(
                    path,
                    job_id=job_id,
                    video_name=video_name,
                    manifest={
                        "fps": 1.0,
                        "total_frames": 100,
                        "frame_stride": 5,
                        "frames": [
                            {
                                "frame_index": 10,
                                "timestamp_sec": 10.0,
                                "detections": [
                                    {"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": bbox}
                                ],
                                "preview_path": "frame_000010.jpg",
                            }
                        ],
                    },
                )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_temporal_isolated=True,
            )

            self.assertEqual(result["removed_detection_count"], 0)
            self.assertEqual(result["event_count"], 2)
            self.assertEqual(result["postprocess"]["temporal_work_window_sec"], 12 * 60 * 60)

    def test_temporal_isolation_does_not_use_different_work_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            for job_id, video_name, bbox in (
                ("job1", "JJR-102283_stream04_260201_040016.mp4", [100, 100, 140, 140]),
                ("job2", "JJR-102283_stream04_260201_060016.mp4", [700, 350, 740, 390]),
            ):
                merge_job_manifest(
                    path,
                    job_id=job_id,
                    video_name=video_name,
                    manifest={
                        "fps": 1.0,
                        "total_frames": 100,
                        "frame_stride": 5,
                        "frames": [
                            {
                                "frame_index": 10,
                                "timestamp_sec": 10.0,
                                "detections": [
                                    {"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": bbox}
                                ],
                                "preview_path": "frame_000010.jpg",
                            }
                        ],
                    },
                )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_temporal_isolated=True,
            )

            self.assertEqual(result["removed_detection_count"], 2)
            self.assertEqual(result["event_count"], 0)

    def test_repeated_work_detection_is_protected_from_position_and_size_outliers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            first_frames = [
                {
                    "frame_index": index,
                    "timestamp_sec": float(index),
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": bbox,
                        }
                    ],
                    "preview_path": f"frame_{index:06d}.jpg",
                }
                for index, bbox in (
                    (10, [700, 350, 740, 390]),
                    (20, [702, 351, 742, 391]),
                    (30, [698, 349, 738, 389]),
                    (40, [701, 352, 741, 392]),
                    (50, [100, 100, 180, 180]),
                )
            ]
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={"fps": 1.0, "total_frames": 100, "frame_stride": 5, "frames": first_frames},
            )
            merge_job_manifest(
                path,
                job_id="job2",
                video_name="JJR-102283_stream04_260201_060016.mp4",
                manifest={
                    "fps": 1.0,
                    "total_frames": 100,
                    "frame_stride": 5,
                    "frames": [
                        {
                            "frame_index": 10,
                            "timestamp_sec": 10.0,
                            "detections": [
                                {
                                    "class_name": "brailer",
                                    "confidence": 0.8,
                                    "bbox_xyxy": [104, 102, 184, 182],
                                }
                            ],
                            "preview_path": "frame_000010.jpg",
                        }
                    ],
                },
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_position_outliers=True,
                remove_size_outliers=True,
            )

            timeline = load_timeline(path)
            kept_frames = {(event["video_name"], event["frame_index"]) for event in timeline["events"]}
            self.assertIn(("JJR-102283_stream04_260201_040016.mp4", 50), kept_frames)
            self.assertIn(("JJR-102283_stream04_260201_060016.mp4", 10), kept_frames)
            self.assertEqual(result["protected_detection_count"], 1)
            self.assertEqual(result["protected_by_condition"]["position_outlier"], 1)
            self.assertEqual(result["protected_by_condition"]["size_outlier"], 1)

    def test_position_outlier_removes_repeated_lower_roi_boundary_hotspot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            detect_roi = {"xyxy_px": [192, 0, 1088, 720]}
            normal_frames = [
                {
                    "frame_index": index,
                    "timestamp_sec": float(index),
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": [700, 120, 760, 200],
                        }
                    ],
                    "preview_path": f"frame_{index:06d}.jpg",
                    "detect_roi": detect_roi,
                }
                for index in range(10, 110, 10)
            ]
            normal_frames.append(
                {
                    "frame_index": 120,
                    "timestamp_sec": 120.0,
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.52,
                            "bbox_xyxy": [1044, 528, 1112, 578],
                        }
                    ],
                    "preview_path": "frame_000120.jpg",
                    "detect_roi": detect_roi,
                }
            )
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={"fps": 1.0, "width": 1280, "height": 720, "frames": normal_frames},
            )
            for job_id, video_name in (
                ("job2", "JJR-102283_stream04_260201_060016.mp4"),
                ("job3", "JJR-102283_stream04_260201_080016.mp4"),
            ):
                merge_job_manifest(
                    path,
                    job_id=job_id,
                    video_name=video_name,
                    manifest={
                        "fps": 1.0,
                        "width": 1280,
                        "height": 720,
                        "frames": [
                            {
                                "frame_index": 10,
                                "timestamp_sec": 10.0,
                                "detections": [
                                    {
                                        "class_name": "brailer",
                                        "confidence": 0.52,
                                        "bbox_xyxy": [1046, 529, 1110, 577],
                                    }
                                ],
                                "preview_path": "frame_000010.jpg",
                                "detect_roi": detect_roi,
                            }
                        ],
                    },
                )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_position_outliers=True,
            )

            self.assertEqual(result["removed_detection_count"], 3)
            self.assertEqual(result["event_count"], 10)
            self.assertEqual(result["removed_by_condition"]["position_outlier"], 3)
            self.assertEqual(result["protected_by_condition"]["position_outlier"], 0)
            self.assertEqual(result["postprocess"]["position_lower_roi_boundary_candidate_count"], 3)
            self.assertEqual(result["postprocess"]["position_lower_roi_y_ratio"], 0.7)
            self.assertEqual(result["postprocess"]["position_roi_edge_margin_ratio"], 0.02)
            self.assertEqual(result["postprocess"]["position_roi_edge_min_video_count"], 3)

    def test_position_outlier_removes_repeated_deep_lower_hotspot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            normal_frames = [
                {
                    "frame_index": index,
                    "timestamp_sec": float(index),
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": [700, 120, 760, 200],
                        }
                    ],
                    "preview_path": f"frame_{index:06d}.jpg",
                }
                for index in range(10, 110, 10)
            ]
            normal_frames.append(
                {
                    "frame_index": 120,
                    "timestamp_sec": 120.0,
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.66,
                            "bbox_xyxy": [590, 558, 650, 674],
                        }
                    ],
                    "preview_path": "frame_000120.jpg",
                }
            )
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={"fps": 1.0, "width": 1280, "height": 720, "frames": normal_frames},
            )
            for job_id, video_name, bbox in (
                ("job2", "JJR-102283_stream04_260201_060016.mp4", [592, 560, 652, 676]),
                ("job3", "JJR-102283_stream04_260201_080016.mp4", [588, 556, 648, 672]),
            ):
                merge_job_manifest(
                    path,
                    job_id=job_id,
                    video_name=video_name,
                    manifest={
                        "fps": 1.0,
                        "width": 1280,
                        "height": 720,
                        "frames": [
                            {
                                "frame_index": 10,
                                "timestamp_sec": 10.0,
                                "detections": [
                                    {
                                        "class_name": "brailer",
                                        "confidence": 0.66,
                                        "bbox_xyxy": bbox,
                                    }
                                ],
                                "preview_path": "frame_000010.jpg",
                            }
                        ],
                    },
                )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_position_outliers=True,
            )

            self.assertEqual(result["removed_detection_count"], 3)
            self.assertEqual(result["event_count"], 10)
            self.assertEqual(result["removed_by_condition"]["position_outlier"], 3)
            self.assertEqual(result["protected_by_condition"]["position_outlier"], 0)
            self.assertEqual(result["postprocess"]["position_lower_roi_boundary_candidate_count"], 0)
            self.assertEqual(result["postprocess"]["position_deep_lower_candidate_count"], 3)
            self.assertEqual(result["postprocess"]["position_deep_lower_y_ratio"], 0.8)

    def test_position_outlier_preserves_lower_roi_boundary_without_three_videos(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            detect_roi = {"xyxy_px": [192, 0, 1088, 720]}
            frames = [
                {
                    "frame_index": index,
                    "timestamp_sec": float(index),
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.8,
                            "bbox_xyxy": [700, 120, 760, 200],
                        }
                    ],
                    "preview_path": f"frame_{index:06d}.jpg",
                    "detect_roi": detect_roi,
                }
                for index in range(10, 110, 10)
            ]
            frames.append(
                {
                    "frame_index": 120,
                    "timestamp_sec": 120.0,
                    "detections": [
                        {
                            "class_name": "brailer",
                            "confidence": 0.52,
                            "bbox_xyxy": [1044, 528, 1112, 578],
                        }
                    ],
                    "preview_path": "frame_000120.jpg",
                    "detect_roi": detect_roi,
                }
            )
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest={"fps": 1.0, "width": 1280, "height": 720, "frames": frames},
            )
            merge_job_manifest(
                path,
                job_id="job2",
                video_name="JJR-102283_stream04_260201_060016.mp4",
                manifest={
                    "fps": 1.0,
                    "width": 1280,
                    "height": 720,
                    "frames": [
                        {
                            "frame_index": 10,
                            "timestamp_sec": 10.0,
                            "detections": [
                                {
                                    "class_name": "brailer",
                                    "confidence": 0.52,
                                    "bbox_xyxy": [1046, 529, 1110, 577],
                                }
                            ],
                            "preview_path": "frame_000010.jpg",
                            "detect_roi": detect_roi,
                        }
                    ],
                },
            )

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_position_outliers=True,
            )

            self.assertEqual(result["removed_detection_count"], 0)
            self.assertEqual(result["event_count"], 12)
            self.assertEqual(result["postprocess"]["position_lower_roi_boundary_candidate_count"], 0)
            self.assertEqual(result["protected_by_condition"]["position_outlier"], 1)

    def test_repeated_work_detection_is_protected_from_color_outlier(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            for job_id, video_name, bbox in (
                ("job1", "JJR-102283_stream04_260201_040016.mp4", [100, 100, 140, 140]),
                ("job2", "JJR-102283_stream04_260201_060016.mp4", [104, 102, 144, 142]),
            ):
                merge_job_manifest(
                    path,
                    job_id=job_id,
                    video_name=video_name,
                    manifest={
                        "fps": 1.0,
                        "total_frames": 100,
                        "frame_stride": 5,
                        "frames": [
                            {
                                "frame_index": 10,
                                "timestamp_sec": 10.0,
                                "detections": [
                                    {"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": bbox}
                                ],
                                "preview_path": "frame_000010.jpg",
                            }
                        ],
                    },
                )

            with patch(
                "brailer_monitor.detect_timeline._color_outlier_key_sets",
                return_value=({(0, 0), (1, 0)}, set()),
            ):
                result = compact_timeline_segments(
                    path,
                    merge_segments=False,
                    remove_color_outliers=True,
                    jobs_root=Path(tmp),
                )

            self.assertEqual(result["removed_detection_count"], 0)
            self.assertEqual(result["event_count"], 2)
            self.assertEqual(result["protected_detection_count"], 2)
            self.assertEqual(result["protected_by_condition"]["color_outlier"], 2)

    def test_uniform_bright_color_is_not_protected_as_repeated_work(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            for job_id, video_name, bbox in (
                ("job1", "JJR-102283_stream04_260201_040016.mp4", [100, 100, 140, 140]),
                ("job2", "JJR-102283_stream04_260201_060016.mp4", [104, 102, 144, 142]),
            ):
                merge_job_manifest(
                    path,
                    job_id=job_id,
                    video_name=video_name,
                    manifest={
                        "fps": 1.0,
                        "total_frames": 100,
                        "frames": [
                            {
                                "frame_index": 10,
                                "timestamp_sec": 10.0,
                                "detections": [
                                    {"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": bbox}
                                ],
                                "preview_path": "frame_000010.jpg",
                            }
                        ],
                    },
                )

            bright_keys = {(0, 0), (1, 0)}
            with patch(
                "brailer_monitor.detect_timeline._color_outlier_key_sets",
                return_value=(set(bright_keys), set(bright_keys)),
            ):
                result = compact_timeline_segments(
                    path,
                    merge_segments=False,
                    remove_color_outliers=True,
                    jobs_root=Path(tmp),
                )

            self.assertEqual(result["removed_detection_count"], 2)
            self.assertEqual(result["event_count"], 0)
            self.assertEqual(result["protected_by_condition"]["color_outlier"], 0)
            self.assertEqual(result["postprocess"]["uniform_bright_color_candidate_count"], 2)

    def test_temporal_isolation_preserves_frames_merged_by_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "timeline.json"
            manifest = {
                "fps": 10.0,
                "total_frames": 100,
                "frame_stride": 3,
                "frames_processed": 3,
                "frames_with_detections": 3,
                "frames": [
                    {
                        "frame_index": idx,
                        "timestamp_sec": ts,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [100, 100, 140, 140]}],
                        "preview_path": f"frame_{idx:06d}.jpg",
                    }
                    for idx, ts in ((10, 1.0), (13, 1.3), (16, 1.6))
                ],
            }
            merge_job_manifest(
                path,
                job_id="job1",
                video_name="JJR-102283_stream04_260201_040016.mp4",
                manifest=manifest,
            )

            result = compact_timeline_segments(
                path,
                max_gap_sec=8,
                merge_segments=True,
                remove_temporal_isolated=True,
            )

            self.assertEqual(result["removed_detection_count"], 0)
            self.assertEqual(result["event_count"], 3)
            self.assertEqual(result["segment_count"], 1)
            self.assertEqual(result["postprocess"]["temporal_merge_protect_gap_sec"], 8.0)

    def test_temporal_similarity_ignores_center_distance(self) -> None:
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
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [100, 100, 140, 140]}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 15,
                        "timestamp_sec": 15.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [700, 350, 740, 390]}],
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

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_temporal_isolated=True,
            )

            self.assertEqual(result["removed_detection_count"], 0)
            self.assertEqual(result["event_count"], 2)

    def test_postprocess_preserves_temporal_tail_when_detection_is_active(self) -> None:
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
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [100, 100, 140, 140]}],
                        "preview_path": "frame_000010.jpg",
                    },
                    {
                        "frame_index": 40,
                        "timestamp_sec": 40.0,
                        "detections": [{"class_name": "brailer", "confidence": 0.8, "bbox_xyxy": [300, 300, 340, 340]}],
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

            result = compact_timeline_segments(
                path,
                merge_segments=False,
                remove_temporal_isolated=True,
                temporal_isolation_protect_tail_sec=10.0,
            )

            self.assertEqual(result["removed_detection_count"], 1)
            self.assertEqual(result["event_count"], 1)
            self.assertEqual(result["postprocess"]["temporal_isolation_protect_tail_sec"], 10.0)
            timeline = load_timeline(path)
            self.assertEqual(timeline["events"][0]["frame_index"], 40)

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
            detection_area_px({"area_px": 0, "bbox_xyxy": [0, 0, 100, 50]}),
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
