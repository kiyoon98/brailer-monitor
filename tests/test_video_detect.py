"""Tests for video detection result serialization."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from brailer_monitor.detector import Detection
from brailer_monitor.video_detect import _detection_to_dict, assess_video_darkness, detect_video, merge_ensemble_detections


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
