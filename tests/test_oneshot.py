"""Tests for one-shot brailer detection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from brailer_monitor.labeling import sam_polygon
from brailer_monitor.oneshot import (
    _interior_color_profile,
    _polygon_to_mask,
    _score_interior_color,
    _score_teardrop,
    _teardrop_profile,
    build_reference,
    detect_oneshot,
    propose_regions,
    score_region,
)
from brailer_monitor.web.annotation import AnnotationManager


def _make_brailer_frame(
    width: int = 640,
    height: int = 480,
    *,
    center: tuple[int, int] = (320, 120),
    size: tuple[int, int] = (100, 80),
    dark_bottom: bool = True,
) -> np.ndarray:
    """Teardrop brailer: narrow top, wide bottom; dark tuna fill at bottom."""
    frame = np.full((height, width, 3), 200, dtype=np.uint8)
    cx, cy = center
    bw, bh = size
    top_y = cy - bh // 2
    bot_y = cy + bh // 2

    pts = np.array(
        [
            [cx, top_y],
            [cx + bw // 3, top_y + bh // 4],
            [cx + bw // 2, bot_y],
            [cx - bw // 2, bot_y],
            [cx - bw // 3, top_y + bh // 4],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(frame, [pts], (140, 140, 140))

    if dark_bottom:
        overlay = frame.copy()
        dark_pts = np.array(
            [
                [cx - bw // 3, top_y + bh // 3],
                [cx + bw // 3, top_y + bh // 3],
                [cx + bw // 2, bot_y],
                [cx - bw // 2, bot_y],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [dark_pts], (25, 25, 25))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)
        frame = np.where(mask[..., None] == 255, overlay, frame)

    return frame


def _teardrop_polygon_norm(
    center: tuple[int, int],
    size: tuple[int, int],
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    cx, cy = center
    bw, bh = size
    top_y = cy - bh // 2
    bot_y = cy + bh // 2
    pts = [
        (cx, top_y),
        (cx + bw // 3, top_y + bh // 4),
        (cx + bw // 2, bot_y),
        (cx - bw // 2, bot_y),
        (cx - bw // 3, top_y + bh // 4),
    ]
    return [(x / width, y / height) for x, y in pts]


class _EmptyMasks:
    def __init__(self) -> None:
        self.data = []


class _SamResult:
    def __init__(self, masks: _EmptyMasks | None) -> None:
        self.masks = masks


class SamPolygonTests(unittest.TestCase):
    def test_sam_polygon_handles_empty_mask_tensor(self) -> None:
        frame = _make_brailer_frame()

        def mock_sam(_frame, bboxes=None, verbose=False):
            return [_SamResult(_EmptyMasks())]

        result = sam_polygon(frame, (280, 40, 380, 180), mock_sam)
        self.assertIsNone(result)

    def test_sam_polygon_handles_tiny_bbox(self) -> None:
        frame = _make_brailer_frame()
        calls = {"count": 0}

        def mock_sam(_frame, bboxes=None, verbose=False):
            calls["count"] += 1
            return [_SamResult(None)]

        result = sam_polygon(frame, (300, 100, 302, 102), mock_sam)
        self.assertIsNone(result)
        self.assertEqual(calls["count"], 0)


class OneshotTests(unittest.TestCase):
    def test_build_reference_and_detect_similar(self) -> None:
        w, h = 640, 480
        ref_frame = _make_brailer_frame(w, h)
        polygon = _teardrop_polygon_norm((320, 120), (100, 80), w, h)
        signature = build_reference([(ref_frame, polygon)])

        target = _make_brailer_frame(w, h, center=(310, 130), size=(95, 75))
        result = detect_oneshot(target, signature, sam_model=None, threshold=0.4)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertGreaterEqual(result.confidence, 0.4)
        self.assertGreaterEqual(len(result.polygon_norm), 4)

    def test_reject_dissimilar_object(self) -> None:
        w, h = 640, 480
        ref_frame = _make_brailer_frame(w, h)
        polygon = _teardrop_polygon_norm((320, 120), (100, 80), w, h)
        signature = build_reference([(ref_frame, polygon)])

        # Bright object in lower region — should not match brailer signature.
        target = np.full((h, w, 3), 200, dtype=np.uint8)
        cv2.rectangle(target, (100, 350), (200, 420), (240, 240, 240), -1)

        result = detect_oneshot(target, signature, sam_model=None, threshold=0.55)
        self.assertIsNone(result)

    def test_propose_regions_finds_candidate(self) -> None:
        w, h = 640, 480
        ref_frame = _make_brailer_frame(w, h)
        polygon = _teardrop_polygon_norm((320, 120), (100, 80), w, h)
        signature = build_reference([(ref_frame, polygon)])

        target = _make_brailer_frame(w, h, center=(300, 110), size=(90, 70))
        regions = propose_regions(target, signature)
        self.assertGreater(len(regions), 0)

    def test_reference_captures_dark_bottom_teardrop(self) -> None:
        w, h = 640, 480
        ref_frame = _make_brailer_frame(w, h)
        polygon = _teardrop_polygon_norm((320, 120), (100, 80), w, h)
        mask = _polygon_to_mask(polygon, w, h)

        color_profile = _interior_color_profile(ref_frame, mask)
        teardrop_profile = _teardrop_profile(mask)

        self.assertLess(color_profile["bottom_top_delta"], 0)
        self.assertGreater(teardrop_profile["width_ratio_bottom_top"], 1.0)
        self.assertGreater(teardrop_profile["widest_y_frac"], 0.4)

        signature = build_reference([(ref_frame, polygon)])
        self.assertLess(signature.bottom_top_v_delta, 0)
        self.assertGreater(signature.width_ratio_bottom_top, 1.0)

    def test_interior_and_teardrop_score_favor_dark_droplet(self) -> None:
        w, h = 640, 480
        ref_frame = _make_brailer_frame(w, h)
        polygon = _teardrop_polygon_norm((320, 120), (100, 80), w, h)
        signature = build_reference([(ref_frame, polygon)])

        good_mask = _polygon_to_mask(polygon, w, h)
        good_frame = _make_brailer_frame(w, h, center=(320, 120), size=(100, 80))

        bad_frame = np.full((h, w, 3), 200, dtype=np.uint8)
        bad_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(bad_mask, (280, 80), (360, 160), 255, -1)
        cv2.rectangle(bad_frame, (280, 80), (360, 160), (180, 180, 180), -1)

        good_interior = _score_interior_color(good_frame, good_mask, signature)
        bad_interior = _score_interior_color(bad_frame, bad_mask, signature)
        good_teardrop = _score_teardrop(good_mask, signature)
        bad_teardrop = _score_teardrop(bad_mask, signature)

        self.assertGreater(good_interior, bad_interior)
        self.assertGreater(good_teardrop, bad_teardrop)

    def test_score_region_higher_for_match(self) -> None:
        w, h = 640, 480
        ref_frame = _make_brailer_frame(w, h)
        polygon = _teardrop_polygon_norm((320, 120), (100, 80), w, h)
        signature = build_reference([(ref_frame, polygon)])

        match_frame = _make_brailer_frame(w, h)
        mismatch_frame = np.full((h, w, 3), 200, dtype=np.uint8)
        cv2.rectangle(mismatch_frame, (50, 400), (150, 460), (180, 180, 180), -1)

        mask_match = _polygon_to_mask(polygon, w, h)
        mask_mismatch = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask_mismatch, (50, 400), (150, 460), 255, -1)

        score_match = score_region(match_frame, mask_match, signature)
        score_mismatch = score_region(mismatch_frame, mask_mismatch, signature)
        self.assertGreater(score_match, score_mismatch)


class AnnotationAutoDetectTests(unittest.TestCase):
    def _make_video(self, path: Path) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
        for i in range(60):
            if 10 <= i <= 20:
                frame = _make_brailer_frame()
            else:
                frame = np.full((480, 640, 3), 200, dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_build_reference_from_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "test.mp4"
            self._make_video(video)
            manager = AnnotationManager(root=Path(tmp) / "jobs")
            job = manager.create_job(video, "test.mp4")

            record = manager.capture_frame(job.job_id, 1.0)
            polygon = [[0.35, 0.15], [0.65, 0.15], [0.65, 0.35], [0.35, 0.35]]
            manager.save_label(job.job_id, record.frame_id, polygon)

            signature = manager.build_reference_from_job(job.job_id)
            self.assertIsNotNone(signature.hist_hs)
            self.assertGreater(signature.area_ratio_max, 0)


if __name__ == "__main__":
    unittest.main()
