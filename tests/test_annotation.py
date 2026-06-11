"""Tests for manual annotation manager."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from brailer_monitor.web.annotation import AnnotationManager


class AnnotationManagerTests(unittest.TestCase):
    def _make_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
        for i in range(45):
            frame = np.full((480, 640, 3), 200, dtype=np.uint8)
            if 10 <= i <= 20:
                cv2.rectangle(frame, (280, 40), (380, 180), (25, 25, 25), -1)
            writer.write(frame)
        writer.release()

    def test_capture_and_save_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "test.mp4"
            self._make_video(video)
            manager = AnnotationManager(root=Path(tmp) / "jobs")
            job = manager.create_job(video, "test.mp4")

            record = manager.capture_frame(job.job_id, 1.0)
            self.assertTrue((manager.job_dir(job.job_id) / "frames" / record.image).exists())

            polygon = [[0.4, 0.1], [0.6, 0.1], [0.6, 0.4], [0.4, 0.4]]
            result = manager.save_label(job.job_id, record.frame_id, polygon)
            self.assertTrue(result["saved"])

            frames = manager.list_frames(job.job_id)
            self.assertEqual(len(frames), 1)
            self.assertTrue(frames[0]["has_label"])
            self.assertEqual(frames[0]["label"]["point_count"], 4)


if __name__ == "__main__":
    unittest.main()
