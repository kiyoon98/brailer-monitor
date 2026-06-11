"""Tests for YOLO seg label parsing."""

from __future__ import annotations

import unittest
from pathlib import Path

from brailer_monitor.label_format import load_frame_label, parse_yolo_seg_line


class LabelFormatTests(unittest.TestCase):
    def test_parse_polygon(self) -> None:
        line = "0 0.5 0.1 0.6 0.2 0.55 0.3 0.45 0.25"
        label = parse_yolo_seg_line(line)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertEqual(label.class_name, "brailer_loaded")
        self.assertEqual(len(label.polygon_norm), 4)
        bbox = label.bbox_px(1280, 720)
        self.assertGreater(bbox[2], bbox[0])

    def test_load_real_label(self) -> None:
        path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "dataset"
            / "staging"
            / "labels"
            / "lake_win_00060s_f00900.txt"
        )
        if not path.exists():
            self.skipTest("label file missing")
        label = load_frame_label(path)
        self.assertIsNotNone(label)
        assert label is not None
        self.assertGreater(label.to_dict(1280, 720)["area_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
