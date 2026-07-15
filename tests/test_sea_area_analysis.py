"""Tests for hybrid sea segmentation and encounter state tracking."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from brailer_monitor.sea_area_analysis import (
    HYBRID_METHOD,
    LEGACY_METHOD,
    SeaAreaAnalyzer,
    SeaEncounterTracker,
)


class _FakeSegmenter:
    def predict(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        height, width = frame.shape[:2]
        sea = np.zeros((height, width), dtype=np.float32)
        sea[height // 2 :, :] = 0.9
        sky = np.zeros((height, width), dtype=np.float32)
        sky[: height // 2, :] = 0.9
        vessel = np.zeros((height, width), dtype=np.float32)
        vessel[height // 2 : height // 2 + 10, width // 2 : width // 2 + 10] = 0.9
        return {
            "sea": sea,
            "sky": sky,
            "vessel": vessel,
            "max_probability": np.full((height, width), 0.9, dtype=np.float32),
        }


class _FailingSegmenter:
    def predict(self, _frame: np.ndarray) -> dict[str, np.ndarray]:
        raise RuntimeError("model unavailable")


def _legacy_result(frame: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[height // 2 :, :] = 1
    return mask, {
        "sea_ratio": 0.5,
        "sea_percent": 50.0,
        "sea_area_px": height * width // 2,
        "sea_method": LEGACY_METHOD,
        "sea_horizon_y": height // 2,
        "sea_roi_xyxy": [0, height // 2, width, height],
        "sea_candidate_area_px": height * width // 2,
    }


class SeaAreaAnalysisTests(unittest.TestCase):
    def test_tracker_confirms_encounter_and_departure_by_duration(self) -> None:
        config = {
            "baseline_window_sec": 60.0,
            "baseline_min_duration_sec": 1.0,
            "baseline_min_samples": 2,
            "encounter_vessel_ratio": 0.005,
            "encounter_sea_drop_ratio": 0.2,
            "encounter_duration_sec": 2.0,
            "departure_vessel_ratio": 0.001,
            "departure_sea_drop_ratio": 0.1,
            "departure_duration_sec": 3.0,
        }
        tracker = SeaEncounterTracker(config)
        open_sea = {"sea_ratio": 0.8, "vessel_ratio": 0.0, "sea_quality": "good"}
        encounter = {"sea_ratio": 0.5, "vessel_ratio": 0.01, "sea_quality": "good"}

        tracker.update(open_sea, 0.0)
        tracker.update(open_sea, 1.0)
        self.assertEqual(tracker.update(open_sea, 2.0)["sea_state"], "open_sea")
        tracker.update(encounter, 3.0)
        tracker.update(encounter, 4.0)
        started = tracker.update(encounter, 5.0)
        self.assertEqual(started["sea_state"], "encounter")
        self.assertEqual(started["sea_event"], "encounter_start")

        tracker.update(open_sea, 6.0)
        tracker.update(open_sea, 7.0)
        tracker.update(open_sea, 8.0)
        departed = tracker.update(open_sea, 9.0)
        self.assertEqual(departed["sea_state"], "open_sea")
        self.assertEqual(departed["sea_event"], "departure")

    def test_tracker_uses_vessel_increase_over_camera_baseline(self) -> None:
        config = {
            "baseline_window_sec": 60.0,
            "baseline_min_duration_sec": 1.0,
            "baseline_min_samples": 2,
            "encounter_vessel_ratio": 0.005,
            "encounter_sea_drop_ratio": 0.2,
            "encounter_duration_sec": 2.0,
            "departure_vessel_ratio": 0.001,
            "departure_sea_drop_ratio": 0.1,
            "departure_duration_sec": 3.0,
        }
        tracker = SeaEncounterTracker(config)
        own_ship = {"sea_ratio": 0.4, "vessel_ratio": 0.7, "sea_quality": "good"}

        tracker.update(own_ship, 0.0)
        tracker.update(own_ship, 1.0)
        result = tracker.update(own_ship, 2.0)
        result = tracker.update(own_ship, 10.0)

        self.assertEqual(result["sea_state"], "open_sea")
        self.assertEqual(result["vessel_baseline_ratio"], 0.7)
        self.assertEqual(result["vessel_increase_ratio"], 0.0)
        self.assertIsNone(result["sea_event"])

    def test_hybrid_analyzer_combines_semantic_and_legacy_masks(self) -> None:
        analyzer = SeaAreaAnalyzer(engine="hybrid", device="cpu")
        analyzer.segmenter = _FakeSegmenter()
        frame = np.full((100, 100, 3), 150, dtype=np.uint8)

        with patch.object(SeaAreaAnalyzer, "_legacy", side_effect=_legacy_result):
            result = analyzer.analyze(frame, timestamp_sec=0.0)

        self.assertEqual(result["sea_method"], HYBRID_METHOD)
        self.assertEqual(result["sea_quality"], "good")
        self.assertGreater(result["sea_ratio"], 0.9)
        self.assertGreater(result["vessel_ratio"], 0.01)
        self.assertEqual(result["sea_state"], "calibrating")

    def test_semantic_failure_falls_back_to_legacy(self) -> None:
        analyzer = SeaAreaAnalyzer(engine="hybrid", device="cpu")
        analyzer.segmenter = _FailingSegmenter()
        frame = np.full((80, 120, 3), 150, dtype=np.uint8)

        with (
            patch.object(SeaAreaAnalyzer, "_legacy", side_effect=_legacy_result),
            patch("torch.cuda.empty_cache") as empty_cache,
        ):
            result = analyzer.analyze(frame, timestamp_sec=0.0)

        self.assertEqual(result["sea_method"], LEGACY_METHOD)
        self.assertEqual(result["sea_ratio"], 0.5)
        self.assertIn("model unavailable", result["sea_fallback_reason"])
        empty_cache.assert_not_called()

    def test_tracker_state_is_restored_between_video_workers(self) -> None:
        config = {
            "baseline_window_sec": 60.0,
            "baseline_min_duration_sec": 1.0,
            "baseline_min_samples": 2,
            "encounter_vessel_ratio": 0.005,
            "encounter_sea_drop_ratio": 0.2,
            "encounter_duration_sec": 10.0,
            "departure_vessel_ratio": 0.001,
            "departure_sea_drop_ratio": 0.1,
            "departure_duration_sec": 30.0,
        }
        with tempfile.TemporaryDirectory() as tmp:
            state_path = Path(tmp) / "sea-state.json"
            first = SeaEncounterTracker(config, state_path=state_path)
            first.update({"sea_ratio": 0.8, "vessel_ratio": 0.0, "sea_quality": "good"}, 0.0)
            first.update({"sea_ratio": 0.8, "vessel_ratio": 0.0, "sea_quality": "good"}, 1.0)

            restored = SeaEncounterTracker(config, state_path=state_path)

            self.assertEqual(len(restored.baseline_samples), 2)
            self.assertEqual(restored.last_timestamp, 1.0)


if __name__ == "__main__":
    unittest.main()
