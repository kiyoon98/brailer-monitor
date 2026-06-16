"""Tests for YOLO training utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brailer_monitor.train import reset_training_artifacts


class TrainResetTests(unittest.TestCase):
    def test_reset_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            weights = root / "models" / "brailer_seg.pt"
            weights.parent.mkdir(parents=True)
            weights.write_bytes(b"fake")
            run_dir = root / "runs" / "segment" / "brailer_seg-1" / "weights"
            run_dir.mkdir(parents=True)
            (run_dir / "best.pt").write_bytes(b"fake")

            deleted = reset_training_artifacts(project_root=root)

            self.assertFalse(weights.exists())
            self.assertFalse((root / "runs" / "segment" / "brailer_seg-1").exists())
            self.assertGreaterEqual(len(deleted), 2)


if __name__ == "__main__":
    unittest.main()
