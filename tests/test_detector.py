"""Tests for YOLO detector task fallback behavior."""

from __future__ import annotations

import unittest

import numpy as np

from brailer_monitor.detector import BrailerDetector


class _EmptyBoxes:
    def __len__(self) -> int:
        return 0


class _Result:
    boxes = _EmptyBoxes()
    names = {}


class _FakeModel:
    def __init__(self, task: str, *, fail_segment: bool = False):
        self.task = task
        self.fail_segment = fail_segment
        self.predictor = None
        self.tasks: list[str] = []
        self.predictors: list[object] = []

    def predict(self, **kwargs):
        task = kwargs.get("task")
        self.tasks.append(task)
        self.predictors.append(kwargs.get("predictor"))
        if self.fail_segment and task == "segment":
            raise RuntimeError("mat1 and mat2 shapes cannot be multiplied (300x0 and 32x6656)")
        return [_Result()]


class DetectorTaskFallbackTests(unittest.TestCase):
    def test_detect_model_is_not_forced_through_segment_predictor(self) -> None:
        detector = BrailerDetector("unused.pt", use_segmentation=True, device="cpu")
        model = _FakeModel("detect")
        detector._model = model
        detector._resolved_path = detector.model_path

        result = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertEqual(result, [])
        self.assertEqual(model.tasks, ["detect"])
        self.assertIsNotNone(model.predictors[0])
        self.assertFalse(detector.use_segmentation)

    def test_empty_mask_coeff_error_falls_back_to_detect_mode(self) -> None:
        detector = BrailerDetector("unused.pt", use_segmentation=True, device="cpu")
        model = _FakeModel("segment", fail_segment=True)
        detect_model = _FakeModel("detect")
        detector._model = model
        detector._detect_model = detect_model
        detector._resolved_path = detector.model_path

        result = detector.predict(np.zeros((16, 16, 3), dtype=np.uint8))

        self.assertEqual(result, [])
        self.assertEqual(model.tasks, ["segment"])
        self.assertEqual(detect_model.tasks, ["detect"])
        self.assertIsNone(model.predictors[0])
        self.assertIsNotNone(detect_model.predictors[0])
        self.assertFalse(detector.use_segmentation)


if __name__ == "__main__":
    unittest.main()
