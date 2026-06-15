"""Tests for Lake media server video discovery."""

from __future__ import annotations

import unittest

from brailer_monitor.lake_video_source import (
    LakeVideoConfig,
    build_filename,
    build_folder_path,
    iter_hours_in_range,
    list_candidate_videos,
)


class LakeVideoSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = LakeVideoConfig(
            base_url="http://10.2.10.158:8041/media/lake_win/2026_decrypted/",
            file_prefix="JJR-102283_stream04",
            year=2026,
            minute_slots=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55),
            second_suffix="16",
        )

    def test_build_folder_and_filename(self) -> None:
        hour = iter_hours_in_range(
            start_month=3,
            start_day=10,
            start_hour=22,
            end_month=3,
            end_day=10,
            end_hour=22,
            year=2026,
        )[0]
        self.assertEqual(build_folder_path(hour), "03/10/22/")
        self.assertEqual(build_filename(hour, 45, self.config), "JJR-102283_stream04_260310_224516.mp4")
        self.assertEqual(build_filename(hour, 0, self.config), "JJR-102283_stream04_260310_220016.mp4")

    def test_list_candidates_for_single_hour(self) -> None:
        videos = list_candidate_videos(
            start_month=3,
            start_day=10,
            start_hour=22,
            end_month=3,
            end_day=10,
            end_hour=22,
            config=self.config,
        )
        self.assertEqual(len(videos), 12)
        self.assertTrue(videos[0]["url"].endswith("/03/10/22/JJR-102283_stream04_260310_220016.mp4"))
        self.assertTrue(videos[-1]["url"].endswith("/03/10/22/JJR-102283_stream04_260310_225516.mp4"))

    def test_hour_range_crosses_midnight(self) -> None:
        hours = iter_hours_in_range(
            start_month=3,
            start_day=10,
            start_hour=22,
            end_month=3,
            end_day=11,
            end_hour=1,
            year=2026,
        )
        self.assertEqual(len(hours), 4)
        videos = list_candidate_videos(
            start_month=3,
            start_day=10,
            start_hour=22,
            end_month=3,
            end_day=11,
            end_hour=1,
            config=self.config,
        )
        self.assertEqual(len(videos), 48)


if __name__ == "__main__":
    unittest.main()
