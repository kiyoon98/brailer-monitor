"""Tests for video filename time parsing."""

from __future__ import annotations

import unittest
from datetime import datetime

from brailer_monitor.video_time import absolute_frame_time, parse_video_start_time


class VideoTimeTests(unittest.TestCase):
    def test_parse_jjr_name(self) -> None:
        dt = parse_video_start_time("JJR-102283_stream04_260201_040016.mp4")
        self.assertEqual(dt, datetime(2026, 2, 1, 4, 0, 0))

    def test_parse_lake_aurora_name(self) -> None:
        dt = parse_video_start_time("LAKE_AURORA_stream03_251017_004023.mp4")
        self.assertEqual(dt, datetime(2025, 10, 17, 0, 40, 0))

    def test_suffix_digits_do_not_affect_start_minute(self) -> None:
        jjr = parse_video_start_time("JJR-102283_stream04_260310_222016.mp4")
        lake = parse_video_start_time("LAKE_AURORA_stream03_251017_004023.mp4")
        self.assertEqual(jjr, datetime(2026, 3, 10, 22, 20, 0))
        self.assertEqual(lake, datetime(2025, 10, 17, 0, 40, 0))

    def test_absolute_frame_time(self) -> None:
        dt = absolute_frame_time("JJR-102283_stream04_260201_040016.mp4", 125.5)
        self.assertEqual(dt.replace(microsecond=0), datetime(2026, 2, 1, 4, 2, 5))

    def test_unknown_name(self) -> None:
        self.assertIsNone(parse_video_start_time("random_video.mp4"))


if __name__ == "__main__":
    unittest.main()
