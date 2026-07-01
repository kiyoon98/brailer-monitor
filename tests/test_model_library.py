"""Tests for trained model library storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brailer_monitor.model_library import ModelLibrary


class ModelLibraryTests(unittest.TestCase):
    def test_register_and_get_model(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "best.pt"
            weights.write_bytes(b"fake weights")
            library = ModelLibrary(root / "library")

            record = library.register(
                weights,
                task_type="segment",
                epochs=12,
                class_names=["brailer"],
                train_images=10,
                val_images=2,
                dataset_frames=[
                    {
                        "split": "train",
                        "image_name": "frame_000001.jpg",
                        "frame_index": 1,
                        "preview_url": "/api/pipeline/dataset/preview/train/frame_000001.jpg",
                        "objects": [],
                    }
                ],
            )

            loaded = library.get(record.id)
            self.assertEqual(loaded.id, record.id)
            self.assertEqual(loaded.task_type, "segment")
            self.assertEqual(loaded.epochs, 12)
            self.assertEqual(loaded.dataset_frames[0]["image_name"], "frame_000001.jpg")
            self.assertTrue(Path(loaded.weights_path).exists())

    def test_invalid_model_id_cannot_escape_library_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = ModelLibrary(root / "library")
            outside = root / "outside"
            outside.mkdir()
            (outside / "sentinel.txt").write_text("keep", encoding="utf-8")

            with self.assertRaises(ValueError):
                library.delete("../outside")

            with self.assertRaises(ValueError):
                library.get("../outside")

            self.assertTrue(outside.exists())
            self.assertTrue((outside / "sentinel.txt").exists())
            self.assertFalse(library.exists("../outside"))


if __name__ == "__main__":
    unittest.main()
