"""Tests for video analysis pipeline helpers."""

from __future__ import annotations

import unittest

from brailer_monitor.pipeline import _is_brailer_detection


class PipelineDetectionFilterTests(unittest.TestCase):
    def test_accepts_named_brailer_classes(self) -> None:
        self.assertTrue(_is_brailer_detection("brailer", 7))
        self.assertTrue(_is_brailer_detection("brailer_loaded", 0))
        self.assertTrue(_is_brailer_detection("Brailer Loaded", 3))

    def test_accepts_unnamed_custom_class_zero(self) -> None:
        self.assertTrue(_is_brailer_detection("class_0", 0))
        self.assertTrue(_is_brailer_detection("", 0))

    def test_rejects_pretrained_or_unrelated_classes(self) -> None:
        self.assertFalse(_is_brailer_detection("person", 0))
        self.assertFalse(_is_brailer_detection("boat", 8))
        self.assertFalse(_is_brailer_detection("tuna", 0))


if __name__ == "__main__":
    unittest.main()
