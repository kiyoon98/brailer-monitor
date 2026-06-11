"""Tests for brailer segment detection and frame extraction."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from brailer_monitor.frame_extractor import (
    BrailerSegment,
    ExtractOptions,
    extract_brailer_frames,
    find_brailer_bbox,
    save_segment_manifest,
    scan_brailer_segments,
)


class FindBrailerBboxTests(unittest.TestCase):
    def test_detects_dark_blob_in_upper_center(self) -> None:
        frame = np.full((720, 1280, 3), 200, dtype=np.uint8)
        cv2.rectangle(frame, (600, 80), (750, 280), (30, 30, 30), -1)
        bbox = find_brailer_bbox(frame)
        self.assertIsNotNone(bbox)
        assert bbox is not None
        self.assertGreater(bbox.score, 0)
        self.assertLess(bbox.y1, 360)

    def test_returns_none_for_empty_frame(self) -> None:
        frame = np.full((720, 1280, 3), 220, dtype=np.uint8)
        self.assertIsNone(find_brailer_bbox(frame))


class SegmentScanTests(unittest.TestCase):
    def _write_synthetic_video(self, path: Path, fps: float = 15.0) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (640, 480),
        )
        for i in range(90):
            frame = np.full((480, 640, 3), 200, dtype=np.uint8)
            if 20 <= i <= 50:
                cv2.rectangle(frame, (280, 40), (380, 180), (25, 25, 25), -1)
            writer.write(frame)
        writer.release()

    def test_scan_finds_one_segment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "test.mp4"
            self._write_synthetic_video(video)
            opts = ExtractOptions(scan_stride=5, gap_tolerance_sec=0.5, segment_padding_sec=0.2)
            segments, fps, total = scan_brailer_segments(video, opts)
            self.assertEqual(total, 90)
            self.assertGreaterEqual(len(segments), 1)
            self.assertGreater(segments[0].detection_count, 0)

    def test_extract_from_known_segments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "test.mp4"
            out = Path(tmp) / "frames"
            self._write_synthetic_video(video)
            segments = [
                BrailerSegment(
                    segment_id=0,
                    start_frame=20,
                    end_frame=50,
                    start_sec=20 / 15,
                    end_sec=50 / 15,
                    detection_count=5,
                )
            ]
            opts = ExtractOptions(extract_stride=10)
            extracted, _ = extract_brailer_frames(
                video, out, prefix="test", options=opts, segments=segments
            )
            self.assertGreater(len(extracted), 0)
            self.assertTrue(extracted[0].image_path.exists())

    def test_save_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "segments.json"
            segments = [
                BrailerSegment(0, 0, 30, 0.0, 2.0, 3),
            ]
            save_segment_manifest(segments, [], manifest, fps=15.0)
            payload = json.loads(manifest.read_text())
            self.assertEqual(payload["segment_count"], 1)


class RealVideoTests(unittest.TestCase):
    VIDEO = Path(__file__).resolve().parents[1] / "data" / "raw" / "JJR-102283_stream04_260310_202016.mp4"

    @unittest.skipUnless(VIDEO.exists(), "raw video not present")
    def test_scan_lake_win_video(self) -> None:
        opts = ExtractOptions(scan_stride=15, gap_tolerance_sec=3.0)
        segments, fps, total = scan_brailer_segments(self.VIDEO, opts)
        self.assertEqual(fps, 15.0)
        self.assertEqual(total, 4500)
        self.assertGreater(len(segments), 0)
        duration = total / fps
        for seg in segments:
            self.assertGreaterEqual(seg.start_sec, 0.0)
            self.assertLessEqual(seg.end_sec, duration)


if __name__ == "__main__":
    unittest.main()
