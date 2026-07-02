"""Tests for imported dataset frame preview."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from brailer_monitor.dataset_preview import (
    list_dataset_frames,
    render_dataset_preview,
    resolve_dataset_image,
)


class DatasetPreviewTests(unittest.TestCase):
    def _make_dataset(self, root: Path) -> None:
        (root / "import_meta.json").write_text(
            json.dumps({"class_names": ["brailer"], "task_type": "segment"}),
            encoding="utf-8",
        )
        img_dir = root / "images" / "train"
        lbl_dir = root / "labels" / "train"
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        img = np.full((120, 160, 3), 180, dtype=np.uint8)
        cv2.imwrite(str(img_dir / "frame_000010.jpg"), img)
        lbl_dir.joinpath("frame_000010.txt").write_text(
            "0 0.10 0.10 0.50 0.10 0.50 0.50 0.10 0.50\n",
            encoding="utf-8",
        )

    def test_list_dataset_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_dataset(root)
            result = list_dataset_frames(root)
            self.assertEqual(result["total"], 1)
            frame = result["frames"][0]
            self.assertEqual(frame["frame_index"], 10)
            self.assertEqual(frame["objects"][0]["class_name"], "brailer")
            self.assertEqual(frame["objects"][0]["shape"], "polygon")
            self.assertNotIn("polygon_norm", frame["objects"][0])
            self.assertNotIn("polygon_px", frame["objects"][0])

    def test_list_dataset_frames_can_include_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_dataset(root)
            result = list_dataset_frames(root, include_geometry=True)
            frame = result["frames"][0]
            obj = frame["objects"][0]

            self.assertEqual(frame["width"], 160)
            self.assertEqual(frame["height"], 120)
            self.assertEqual(obj["polygon_norm"][0], [0.10, 0.10])
            self.assertEqual(obj["polygon_px"][0], [16, 12])

    def test_render_dataset_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_dataset(root)
            vis = render_dataset_preview(root, "train", "frame_000010.jpg")
            self.assertEqual(vis.shape[:2], (120, 160))


if __name__ == "__main__":
    unittest.main()
