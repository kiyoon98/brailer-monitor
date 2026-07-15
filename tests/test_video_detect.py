"""Tests for video detection result serialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from brailer_monitor.detector import Detection
from brailer_monitor.video_detect import (
    _detection_to_dict,
    assess_video_darkness,
    detection_roi_for_frame,
    detect_video,
    estimate_sea_ratio,
    filter_detections_by_roi,
    merge_ensemble_detections,
    normalize_sea_analysis_interval,
    sea_analysis_due,
    sea_mask_for_frame,
)


class VideoDetectTests(unittest.TestCase):
    def _write_test_video(self, path: Path, frames: list[np.ndarray], *, fps: float = 5.0) -> None:
        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        self.assertTrue(writer.isOpened())
        for frame in frames:
            writer.write(frame)
        writer.release()

    def test_detect_video_skips_clearly_dark_video_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "dark.mp4"
            frames = [np.full((32, 32, 3), 8, dtype=np.uint8) for _ in range(6)]
            self._write_test_video(video_path, frames)

            manifest = detect_video(
                video_path,
                root / "missing-model.pt",
                output_dir=root / "out",
                skip_dark_video=True,
            )

            self.assertTrue(manifest["skipped"])
            self.assertEqual(manifest["skip_reason"], "dark_video")
            self.assertEqual(manifest["frames_processed"], 0)
            self.assertEqual(manifest["frames"], [])
            self.assertTrue((root / "out" / "detections.json").exists())

    def test_detect_video_samples_sea_area_on_five_second_default_interval(self) -> None:
        class FakeSeaAnalyzer:
            def __init__(self, **_kwargs) -> None:
                self.calls: list[float] = []

            def analyze(self, _frame: np.ndarray, *, timestamp_sec: float) -> dict[str, object]:
                self.calls.append(timestamp_sec)
                return {
                    "sea_ratio": 0.5,
                    "sea_percent": 50.0,
                    "sea_area_px": 512,
                    "sea_method": "test",
                    "sea_quality": "good",
                    "sea_state": "open_sea",
                    "sea_confidence": 0.9,
                    "vessel_ratio": 0.0,
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sample.mp4"
            frames = [np.full((32, 32, 3), 120, dtype=np.uint8) for _ in range(61)]
            self._write_test_video(video_path, frames, fps=10.0)
            analyzer = FakeSeaAnalyzer()
            model_manifest = [{"path": str(root / "model.pt"), "name": "test"}]

            with (
                patch("brailer_monitor.video_detect._build_detectors", return_value=([], model_manifest)),
                patch("brailer_monitor.video_detect._predict_ensemble", return_value=[]),
                patch("brailer_monitor.sea_area_analysis.SeaAreaAnalyzer", return_value=analyzer),
            ):
                manifest = detect_video(
                    video_path,
                    root / "model.pt",
                    output_dir=root / "out",
                    frame_stride=1,
                    calculate_sea_ratio=True,
                    use_sam=False,
                    save_previews=False,
                )

            sampled = [frame for frame in manifest["frames"] if frame.get("sea_method") == "test"]
            self.assertEqual(len(sampled), 2)
            self.assertEqual([frame["timestamp_sec"] for frame in sampled], [0.0, 5.0])
            self.assertEqual(manifest["sea_analysis_interval_sec"], 5.0)

    def test_sea_only_skips_object_detector_and_sam_without_model(self) -> None:
        class FakeSeaAnalyzer:
            def __init__(self, **_kwargs) -> None:
                pass

            def analyze(self, _frame: np.ndarray, *, timestamp_sec: float) -> dict[str, object]:
                return {
                    "sea_ratio": 0.4,
                    "sea_percent": 40.0,
                    "sea_area_px": 410,
                    "sea_method": "test",
                    "sea_quality": "good",
                    "sea_state": "open_sea",
                    "sea_confidence": 0.8,
                    "vessel_ratio": 0.1,
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "sea-only.mp4"
            self._write_test_video(
                video_path,
                [np.full((32, 32, 3), 120, dtype=np.uint8) for _ in range(3)],
            )

            with (
                patch("brailer_monitor.video_detect._build_detectors") as build_detectors,
                patch("brailer_monitor.video_detect._predict_ensemble") as predict_ensemble,
                patch("brailer_monitor.video_detect.SamBoxSegmenter") as sam_segmenter,
                patch("brailer_monitor.sea_area_analysis.SeaAreaAnalyzer", FakeSeaAnalyzer),
            ):
                manifest = detect_video(
                    video_path,
                    None,
                    output_dir=root / "out",
                    sea_only=True,
                    sea_analysis_interval_sec=0,
                    save_previews=False,
                )

            build_detectors.assert_not_called()
            predict_ensemble.assert_not_called()
            sam_segmenter.assert_not_called()
            self.assertTrue(manifest["sea_only"])
            self.assertFalse(manifest["object_detection_enabled"])
            self.assertTrue(manifest["sea_ratio_enabled"])
            self.assertIsNone(manifest["model"])
            self.assertEqual(manifest["models"], [])
            self.assertEqual(manifest["frames_processed"], 3)
            self.assertEqual(manifest["frames_with_detections"], 0)
            self.assertTrue(all(frame["detections"] == [] for frame in manifest["frames"]))
            self.assertTrue(all(frame.get("sea_method") == "test" for frame in manifest["frames"]))

    def test_zero_sea_interval_means_every_processed_frame(self) -> None:
        self.assertTrue(sea_analysis_due(0.0, None, 0.0))
        self.assertTrue(sea_analysis_due(0.1, 0.0, 0.0))
        self.assertFalse(sea_analysis_due(4.999, 0.0, 5.0))
        self.assertTrue(sea_analysis_due(5.0, 0.0, 5.0))
        self.assertEqual(normalize_sea_analysis_interval(300), 300.0)
        with self.assertRaises(ValueError):
            normalize_sea_analysis_interval(300.1)

    def test_assess_video_darkness_requires_all_samples_dark(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            video_path = root / "mixed.mp4"
            dark = np.full((32, 32, 3), 8, dtype=np.uint8)
            bright = np.full((32, 32, 3), 160, dtype=np.uint8)
            self._write_test_video(video_path, [dark, dark, bright, dark, dark])

            cap = cv2.VideoCapture(str(video_path))
            try:
                assessment = assess_video_darkness(cap, 5)
            finally:
                cap.release()

            self.assertEqual(assessment["sample_count"], 5)
            self.assertFalse(assessment["all_samples_dark"])

    def test_estimate_sea_ratio_uses_blue_cyan_frame_fraction(self) -> None:
        blue_sea = np.full((20, 20, 3), (180, 110, 30), dtype=np.uint8)
        deck = np.full((20, 20, 3), (40, 150, 80), dtype=np.uint8)
        mixed = np.concatenate([blue_sea[:, :10], deck[:, 10:]], axis=1)

        sea_stats = estimate_sea_ratio(blue_sea)
        deck_stats = estimate_sea_ratio(deck)
        mixed_stats = estimate_sea_ratio(mixed)

        self.assertGreater(sea_stats["sea_ratio"], 0.95)
        self.assertLess(deck_stats["sea_ratio"], 0.05)
        self.assertAlmostEqual(mixed_stats["sea_ratio"], 0.5, delta=0.05)
        self.assertEqual(sea_stats["sea_method"], "hsv_lab_grabcut_v3")

    def test_estimate_sea_ratio_excludes_isolated_deck_or_hull_fragments(self) -> None:
        frame = np.full((100, 100, 3), (40, 150, 80), dtype=np.uint8)
        sea_color = np.array((180, 110, 30), dtype=np.uint8)
        frame[10:70, 35:65] = sea_color
        frame[76:95, 8:24] = sea_color

        sea_mask, stats = sea_mask_for_frame(frame)

        self.assertGreater(stats["sea_ratio"], 0.16)
        self.assertLess(stats["sea_ratio"], 0.2)
        self.assertTrue(sea_mask[20, 45])
        self.assertFalse(sea_mask[85, 12])
        self.assertGreater(stats["sea_candidate_area_px"], stats["sea_area_px"])

    def test_merge_ensemble_detections_uses_highest_confidence_and_model_labels(self) -> None:
        low = Detection(
            bbox_xyxy=(0.0, 0.0, 100.0, 100.0),
            confidence=0.6,
            class_id=0,
            class_name="brailer",
            model_id="model_a",
            model_name="A",
        )
        high = Detection(
            bbox_xyxy=(5.0, 5.0, 105.0, 105.0),
            confidence=0.9,
            class_id=0,
            class_name="brailer",
            model_id="model_b",
            model_name="B",
        )

        merged = merge_ensemble_detections([low, high])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.9)
        self.assertEqual(merged[0].model_id, "model_b")
        self.assertEqual(merged[0].model_name, "B")
        self.assertEqual(merged[0].ensemble_model_ids, ("model_b", "model_a"))
        self.assertEqual(merged[0].ensemble_model_names, ("B", "A"))

    def test_merge_ensemble_detections_keeps_largest_bbox_with_highest_confidence(self) -> None:
        large_low = Detection(
            bbox_xyxy=(0.0, 0.0, 130.0, 130.0),
            confidence=0.7,
            class_id=0,
            class_name="brailer",
            model_id="model_a",
            model_name="A",
        )
        small_high = Detection(
            bbox_xyxy=(10.0, 10.0, 115.0, 115.0),
            confidence=0.95,
            class_id=0,
            class_name="brailer",
            model_id="model_b",
            model_name="B",
        )

        merged = merge_ensemble_detections([large_low, small_high])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox_xyxy, large_low.bbox_xyxy)
        self.assertEqual(merged[0].confidence, 0.95)
        self.assertEqual(merged[0].model_id, "model_b")
        self.assertEqual(merged[0].model_name, "B")
        self.assertEqual(merged[0].ensemble_model_ids, ("model_b", "model_a"))
        self.assertEqual(merged[0].ensemble_model_names, ("B", "A"))

    def test_merge_ensemble_detections_expands_overlap_cluster(self) -> None:
        first = Detection((0.0, 0.0, 100.0, 100.0), 0.95, 0, "brailer", model_id="a", model_name="A")
        middle = Detection((20.0, 0.0, 120.0, 100.0), 0.9, 0, "brailer", model_id="b", model_name="B")
        last = Detection((40.0, 0.0, 140.0, 100.0), 0.85, 0, "brailer", model_id="c", model_name="C")

        merged = merge_ensemble_detections([first, middle, last])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].confidence, 0.95)
        self.assertEqual(merged[0].ensemble_model_ids, ("a", "b", "c"))
        self.assertEqual(merged[0].ensemble_model_names, ("A", "B", "C"))

    def test_merge_ensemble_detections_merges_contained_boxes_and_brailer_aliases(self) -> None:
        large = Detection(
            bbox_xyxy=(0.0, 0.0, 200.0, 200.0),
            confidence=0.88,
            class_id=0,
            class_name="brailer",
            model_id="model_a",
            model_name="A",
        )
        contained = Detection(
            bbox_xyxy=(60.0, 60.0, 120.0, 120.0),
            confidence=0.92,
            class_id=1,
            class_name="brailers",
            model_id="model_b",
            model_name="B",
        )

        merged = merge_ensemble_detections([large, contained])

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].bbox_xyxy, large.bbox_xyxy)
        self.assertEqual(merged[0].confidence, 0.92)
        self.assertEqual(merged[0].class_name, "brailer")
        self.assertEqual(merged[0].model_id, "model_b")
        self.assertEqual(merged[0].ensemble_model_ids, ("model_b", "model_a"))

    def test_merge_ensemble_detections_keeps_different_classes_separate(self) -> None:
        detections = [
            Detection((0.0, 0.0, 100.0, 100.0), 0.9, 0, "brailer", model_id="a", model_name="A"),
            Detection((5.0, 5.0, 105.0, 105.0), 0.8, 1, "other", model_id="b", model_name="B"),
        ]

        merged = merge_ensemble_detections(detections)

        self.assertEqual(len(merged), 2)

    def test_filter_detections_by_roi_uses_bbox_center(self) -> None:
        roi = detection_roi_for_frame(1000, 500, {"top": 0.15, "right": 0.15, "bottom": 0.15, "left": 0.15})
        inside = Detection((150.0, 75.0, 250.0, 175.0), 0.9, 0, "brailer")
        left_edge = Detection((0.0, 100.0, 100.0, 200.0), 0.9, 0, "brailer")
        bottom_edge = Detection((400.0, 440.0, 500.0, 490.0), 0.9, 0, "brailer")

        filtered = filter_detections_by_roi([inside, left_edge, bottom_edge], roi)

        self.assertEqual(filtered, [inside])
        self.assertEqual(roi["xyxy_px"], [150, 75, 850, 425])
        self.assertEqual(roi["label"], "x 15-85%, y 15-85%")

    def test_detection_to_dict_includes_mask_polygon_and_size(self) -> None:
        mask = np.zeros((20, 30), dtype=np.float32)
        mask[5:15, 7:22] = 1.0
        mask[0:4, 0:6] = 1.0
        det = Detection(
            bbox_xyxy=(7.0, 5.0, 22.0, 15.0),
            confidence=0.9,
            class_id=0,
            class_name="brailer",
            mask=mask,
        )

        payload = _detection_to_dict(det, frame_w=30, frame_h=20, sam_mask=mask)

        # The mask has extra positive pixels outside the bbox; they must be clipped out.
        self.assertEqual(payload["mask_area_px"], 150)
        self.assertEqual(payload["area_px"], 150)
        self.assertEqual(payload["mask_width_px"], 15)
        self.assertEqual(payload["mask_height_px"], 10)
        self.assertEqual(payload["segmentation_source"], "sam2")
        self.assertGreaterEqual(len(payload["polygon_xy"]), 4)
        self.assertEqual(payload["yolo_mask_area_px"], 150)
        self.assertEqual(payload["yolo_mask_width_px"], 15)
        self.assertEqual(payload["yolo_mask_height_px"], 10)
        self.assertGreaterEqual(len(payload["yolo_polygon_xy"]), 4)
        for x, y in payload["polygon_xy"]:
            self.assertGreaterEqual(x, 7)
            self.assertLessEqual(x, 21)
            self.assertGreaterEqual(y, 5)
            self.assertLessEqual(y, 14)


if __name__ == "__main__":
    unittest.main()
