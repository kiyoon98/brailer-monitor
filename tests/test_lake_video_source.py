"""Tests for Lake media server video discovery."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from brailer_monitor.lake_video_source import (
    LakeVideoConfig,
    build_filename,
    build_folder_path,
    discover_videos_in_range,
    iter_hours_in_range,
    list_candidate_videos,
    list_lake_profile_summaries,
    load_lake_video_config,
)


class LakeVideoSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.jjr = LakeVideoConfig(
            profile_id="jjr",
            label="JJR-102283 stream04",
            base_url="http://10.2.10.158:8041/media/lake_win/2026_decrypted/",
            file_prefix="JJR-102283_stream04",
            year=2026,
            minute_slots=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55),
            second_suffixes=("16",),
        )
        self.lake = LakeVideoConfig(
            profile_id="lake_aurora",
            label="LAKE AURORA stream03",
            base_url="http://10.2.10.158:8041/media/lake_win/2025_decrypted/",
            file_prefix="LAKE_AURORA_stream03",
            year=2025,
            minute_slots=(0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55),
            second_suffixes=("23",),
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
        self.assertEqual(build_filename(hour, 45, self.jjr), "JJR-102283_stream04_260310_224516.mp4")
        self.assertEqual(build_filename(hour, 0, self.jjr), "JJR-102283_stream04_260310_220016.mp4")

    def test_build_lake_aurora_filename(self) -> None:
        hour = iter_hours_in_range(
            start_month=10,
            start_day=17,
            start_hour=0,
            end_month=10,
            end_day=17,
            end_hour=0,
            year=2025,
        )[0]
        self.assertEqual(
            build_filename(hour, 40, self.lake),
            "LAKE_AURORA_stream03_251017_004023.mp4",
        )

    def test_list_candidates_for_single_hour(self) -> None:
        videos = list_candidate_videos(
            start_month=3,
            start_day=10,
            start_hour=22,
            end_month=3,
            end_day=10,
            end_hour=22,
            config=self.jjr,
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
            config=self.jjr,
        )
        self.assertEqual(len(videos), 48)

    @patch("brailer_monitor.lake_video_source.probe_video_exists")
    def test_discover_tries_multiple_suffixes(self, fake_probe) -> None:
        urls: list[str] = []

        def probe(url: str, *, timeout: float = 8.0) -> bool:
            urls.append(url)
            return url.endswith("004023.mp4")

        fake_probe.side_effect = probe
        config = LakeVideoConfig(
            profile_id="test",
            label="test",
            base_url="http://example.invalid/",
            file_prefix="LAKE_AURORA_stream03",
            year=2025,
            minute_slots=(40,),
            second_suffixes=("16", "23"),
        )
        videos = discover_videos_in_range(
            start_month=10,
            start_day=17,
            start_hour=0,
            end_month=10,
            end_day=17,
            end_hour=0,
            config=config,
            check_exists=True,
        )
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["filename"], "LAKE_AURORA_stream03_251017_004023.mp4")
        self.assertEqual(len(urls), 2)

    def test_load_profiles_from_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "lake_video.json"
            path.write_text(
                json.dumps(
                    {
                        "default_profile": "lake_aurora",
                        "profiles": {
                            "jjr": {
                                "label": "JJR",
                                "file_prefix": "JJR-102283_stream04",
                                "year": 2026,
                                "second_suffixes": ["16"],
                            },
                            "lake_aurora": {
                                "label": "Aurora",
                                "file_prefix": "LAKE_AURORA_stream03",
                                "year": 2025,
                                "second_suffixes": ["23"],
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            summaries = list_lake_profile_summaries(path)
            self.assertEqual(len(summaries), 2)
            config = load_lake_video_config(path)
            self.assertEqual(config.profile_id, "lake_aurora")
            self.assertEqual(config.file_prefix, "LAKE_AURORA_stream03")
            self.assertEqual(config.second_suffixes, ("23",))


if __name__ == "__main__":
    unittest.main()
