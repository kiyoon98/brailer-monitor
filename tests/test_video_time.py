"""Tests for video filename time parsing."""

from __future__ import annotations

import unittest
from datetime import datetime

from brailer_monitor.video_time import absolute_frame_time, parse_video_start_time


class VideoTimeTests(unittest.TestCase):
    def test_parse_standard_name(self) -> None:
        dt = parse_video_start_time("JJR-102283_stream04_260201_040016.mp4")
        self.assertEqual(dt, datetime(2026, 2, 1, 4, 0, 0))

    def test_absolute_frame_time(self) -> None:
        dt = absolute_frame_time("JJR-102283_stream04_260201_040016.mp4", 125.5)
        self.assertEqual(dt.replace(microsecond=0), datetime(2026, 2, 1, 4, 2, 5))

    def test_unknown_name(self) -> None:
        self.assertIsNone(parse_video_start_time("random_video.mp4"))


if __name__ == "__main__":
    unittest.main()
