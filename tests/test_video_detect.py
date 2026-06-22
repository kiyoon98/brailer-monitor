"""Tests for video detection result serialization."""

from __future__ import annotations

import unittest

import numpy as np

from brailer_monitor.detector import Detection
from brailer_monitor.video_detect import _detection_to_dict


class VideoDetectTests(unittest.TestCase):
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
        for x, y in payload["polygon_xy"]:
            self.assertGreaterEqual(x, 7)
            self.assertLessEqual(x, 21)
            self.assertGreaterEqual(y, 5)
            self.assertLessEqual(y, 14)


if __name__ == "__main__":
    unittest.main()
