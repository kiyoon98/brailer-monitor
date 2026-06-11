"""Unit tests for event model, calibration, and aggregation."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brailer_monitor.aggregation import summarize
from brailer_monitor.calibration import (
    CameraCalibration,
    StandardCapacityConfig,
    load_calibration,
    load_capacity_config,
)
from brailer_monitor.events import (
    BrailerEvent,
    ReviewStatus,
    load_events_json,
    save_events_csv,
    save_events_json,
)
from brailer_monitor.volume_estimator import VolumeEstimator


class CalibrationTests(unittest.TestCase):
    def test_length_cm(self) -> None:
        cal = CameraCalibration(camera_id="cam-1", cm_per_pixel=2.0)
        self.assertEqual(cal.length_cm(100), 200.0)

    def test_load_calibration_file(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "calibration.json"
        cal = load_calibration(path)
        self.assertEqual(cal.camera_id, "cam-brailer-01")
        self.assertIsNotNone(cal.transfer_zone)

    def test_combine_weights(self) -> None:
        cfg = StandardCapacityConfig()
        est, conf = cfg.combine_weights(800.0, 1500.0, 0.9, has_geometry=True)
        self.assertGreater(est, 800.0)
        self.assertLess(est, 1500.0)
        self.assertGreater(conf, 0.0)


class EventTests(unittest.TestCase):
    def test_roundtrip_json(self) -> None:
        event = BrailerEvent(
            timestamp="PT00H01M00.000S",
            camera_id="cam-1",
            track_id="trk-0001",
            fill_ratio=0.8,
            volume_m3=1.2,
            weight_kg_geom=700.0,
            weight_kg_std=1500.0,
            weight_kg_est=1000.0,
            confidence=0.75,
            video_clip_ref="clip.mp4",
            review_status=ReviewStatus.ACCEPTED,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.json"
            save_events_json([event], path)
            loaded = load_events_json(path)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].track_id, "trk-0001")
        self.assertEqual(loaded[0].review_status, ReviewStatus.ACCEPTED)

    def test_csv_export(self) -> None:
        event = BrailerEvent(
            timestamp="PT00H01M00.000S",
            camera_id="cam-1",
            track_id="trk-0001",
            weight_kg_est=1000.0,
            confidence=0.75,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.csv"
            save_events_csv([event], path)
            text = path.read_text(encoding="utf-8")
        self.assertIn("trk-0001", text)
        self.assertIn("brailer_transfer", text)


class AggregationTests(unittest.TestCase):
    def test_summarize(self) -> None:
        events = [
            BrailerEvent(
                timestamp="t1",
                camera_id="cam-1",
                track_id="trk-1",
                weight_kg_geom=800.0,
                weight_kg_std=1500.0,
                weight_kg_est=1100.0,
                confidence=0.8,
                video_clip_ref="a.mp4",
                review_status=ReviewStatus.ACCEPTED,
            ),
            BrailerEvent(
                timestamp="t2",
                camera_id="cam-1",
                track_id="trk-2",
                weight_kg_geom=600.0,
                weight_kg_std=1500.0,
                weight_kg_est=900.0,
                confidence=0.5,
                video_clip_ref="a.mp4",
                review_status=ReviewStatus.PENDING,
            ),
            BrailerEvent(
                timestamp="t3",
                camera_id="cam-1",
                track_id="trk-3",
                weight_kg_est=0.0,
                confidence=0.3,
                review_status=ReviewStatus.EXCLUDED,
            ),
        ]
        summary = summarize(events, review_confidence_threshold=0.65)
        self.assertEqual(summary.transfer_count, 2)
        self.assertEqual(summary.excluded_count, 1)
        self.assertEqual(summary.pending_review_count, 1)
        self.assertAlmostEqual(summary.total_weight_kg_est, 2000.0)


class VolumeEstimatorTests(unittest.TestCase):
    def test_bbox_fallback(self) -> None:
        import numpy as np

        from brailer_monitor.detector import Detection

        cal = CameraCalibration(camera_id="cam-1", cm_per_pixel=2.5)
        estimator = VolumeEstimator(cal)
        det = Detection(
            bbox_xyxy=(100.0, 100.0, 300.0, 400.0),
            confidence=0.9,
            class_id=0,
            class_name="brailer_loaded",
        )
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        result = estimator.estimate(frame, det)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreater(result.weight_kg_geom, 0.0)
        self.assertGreater(result.fill_ratio, 0.0)


class SampleFileTests(unittest.TestCase):
    def test_sample_events_valid(self) -> None:
        path = Path(__file__).resolve().parents[1] / "examples" / "sample_events.json"
        events = load_events_json(path)
        self.assertEqual(len(events), 2)

    def test_summarize_sample(self) -> None:
        path = Path(__file__).resolve().parents[1] / "examples" / "sample_events.json"
        events = load_events_json(path)
        capacity_path = Path(__file__).resolve().parents[1] / "config" / "standard_capacity.json"
        capacity = load_capacity_config(capacity_path)
        summary = summarize(events, capacity.review_confidence_threshold)
        payload = summary.to_dict()
        json.dumps(payload)
        self.assertEqual(summary.transfer_count, 2)


if __name__ == "__main__":
    unittest.main()
