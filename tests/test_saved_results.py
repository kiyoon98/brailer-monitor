"""Tests for saving and restoring named detection result sets."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from brailer_monitor.web.detect_pipeline import DetectPipelineManager


class SavedResultsTests(unittest.TestCase):
    def test_save_and_load_detection_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pipeline_root = root / "pipeline"
            manager = DetectPipelineManager(
                root=pipeline_root,
                dataset_root=root / "dataset",
                config_dir=root / "config",
            )
            job_id = "job-a"
            job_dir = pipeline_root / "detect_jobs" / job_id
            (job_dir / "previews").mkdir(parents=True)
            (job_dir / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": job_id,
                        "video_name": "JJR-102283_stream04_260201_040016.mp4",
                        "created_at": "2026-02-01T00:00:00+00:00",
                        "status": "completed",
                    }
                ),
                encoding="utf-8",
            )
            (job_dir / "detections.json").write_text("{}", encoding="utf-8")
            (job_dir / "previews" / "frame_000001.jpg").write_bytes(b"fake")
            timeline = {
                "updated_at": "2026-02-01T00:00:00+00:00",
                "videos": [
                    {
                        "job_id": job_id,
                        "video_name": "JJR-102283_stream04_260201_040016.mp4",
                        "duration_sec": 10,
                        "frame_stride": 1,
                    }
                ],
                "events": [
                    {
                        "job_id": job_id,
                        "video_name": "JJR-102283_stream04_260201_040016.mp4",
                        "frame_index": 1,
                        "timestamp_sec": 1.0,
                        "absolute_time": "2026-02-01T04:00:01",
                        "absolute_time_label": "2026-02-01 04:00:01",
                        "preview_path": "frame_000001.jpg",
                        "detections": [{"class_name": "brailer", "confidence": 0.9}],
                    }
                ],
            }
            (pipeline_root / "detect_timeline.json").write_text(
                json.dumps(timeline),
                encoding="utf-8",
            )

            saved = manager.save_current_results("조업선-운반선 적재")
            self.assertEqual(saved["name"], "조업선-운반선 적재")
            self.assertEqual(saved["segment_count"], 1)
            self.assertEqual(saved["job_count"], 1)

            (pipeline_root / "detect_timeline.json").unlink()
            import shutil

            shutil.rmtree(pipeline_root / "detect_jobs")

            loaded = manager.load_saved_results(saved["id"])
            self.assertTrue(loaded["loaded"])
            self.assertTrue((pipeline_root / "detect_timeline.json").exists())
            self.assertTrue((pipeline_root / "detect_jobs" / job_id / "previews" / "frame_000001.jpg").exists())
            self.assertEqual(manager.get_state()["detect_timeline"]["segment_count"], 1)


if __name__ == "__main__":
    unittest.main()
