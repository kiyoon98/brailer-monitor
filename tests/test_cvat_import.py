"""Tests for CVAT 1.1 import."""

from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import cv2
import numpy as np

from brailer_monitor.cvat_import import (
    _extract_frames_from_video,
    _parse_cvat_xml,
    import_cvat,
    import_cvat_zip,
)


def _write_test_image(path: Path, color: tuple[int, int, int] = (200, 200, 200)) -> None:
    img = np.full((480, 640, 3), color, dtype=np.uint8)
    cv2.imwrite(str(path), img)


class CvatImportTests(unittest.TestCase):
    def _make_track_zip(self, directory: Path) -> Path:
        images_dir = directory / "images"
        images_dir.mkdir(parents=True)
        for i in (0, 5, 10):
            _write_test_image(images_dir / f"frame_{i:06d}.jpg")

        xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta>
    <task>
      <labels>
        <label><name>brailer_loaded</name></label>
      </labels>
    </task>
  </meta>
  <track id="0" label="brailer_loaded" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="100" ytl="80" xbr="300" ybr="260"/>
    <box frame="5" outside="0" occluded="0" keyframe="1" xtl="110" ytl="85" xbr="310" ybr="265"/>
    <box frame="10" outside="1" occluded="0" keyframe="1" xtl="0" ytl="0" xbr="1" ybr="1"/>
  </track>
</annotations>
"""
        (directory / "annotations.xml").write_text(xml, encoding="utf-8")
        zip_path = directory / "cvat.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(directory / "annotations.xml", "annotations.xml")
            for img in images_dir.glob("*.jpg"):
                zf.write(img, f"images/{img.name}")
        return zip_path

    def test_parse_track_xml(self) -> None:
        xml = """<annotations>
          <track id="0" label="fish">
            <polygon frame="3" outside="0" points="10,10;50,10;50,50" keyframe="1"/>
          </track>
        </annotations>"""
        path = Path(tempfile.mkdtemp()) / "a.xml"
        path.write_text(xml, encoding="utf-8")
        anns, _, labels = _parse_cvat_xml(path)
        self.assertEqual(labels, ["fish"])
        self.assertEqual(len(anns), 1)
        self.assertEqual(anns[0].shape, "polygon")
        self.assertEqual(anns[0].frame_id, 3)

    def _make_xml_only_zip(self, directory: Path) -> Path:
        xml = """<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <meta><task><labels><label><name>brailer_loaded</name></label></labels></task></meta>
  <track id="0" label="brailer_loaded" source="manual">
    <box frame="0" outside="0" occluded="0" keyframe="1" xtl="100" ytl="80" xbr="300" ybr="260"/>
    <box frame="5" outside="0" occluded="0" keyframe="1" xtl="110" ytl="85" xbr="310" ybr="265"/>
  </track>
</annotations>
"""
        xml_path = directory / "annotations.xml"
        xml_path.write_text(xml, encoding="utf-8")
        zip_path = directory / "cvat_no_images.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.write(xml_path, "annotations.xml")
        return zip_path

    def _make_test_video(self, path: Path, frames: int = 15) -> None:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (640, 480))
        for i in range(frames):
            frame = np.full((480, 640, 3), 180, dtype=np.uint8)
            if i in (0, 5):
                cv2.rectangle(frame, (100, 80), (300, 260), (30, 30, 30), -1)
            writer.write(frame)
        writer.release()

    def test_import_with_video_when_no_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = self._make_xml_only_zip(tmp_path)
            video_path = tmp_path / "source.mp4"
            self._make_test_video(video_path)

            dataset_root = tmp_path / "dataset"
            result = import_cvat(
                zip_path,
                dataset_root,
                video_path=video_path,
                config_dir=tmp_path / "config",
            )
            self.assertEqual(result.train_images + result.val_images, 2)
            self.assertEqual(
                json.loads((dataset_root / "import_meta.json").read_text())["frame_source"],
                "video",
            )

    def test_import_requires_video_without_images(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = self._make_xml_only_zip(tmp_path)
            with self.assertRaises(ValueError):
                import_cvat(zip_path, tmp_path / "dataset", config_dir=tmp_path / "config")

    def test_extract_frames_from_video(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "source.mp4"
            self._make_test_video(video_path)
            out_dir = tmp_path / "frames"
            mapping = _extract_frames_from_video(video_path, [0, 5], out_dir)
            self.assertEqual(len(mapping), 2)
            self.assertTrue(mapping[0].exists())

    def test_import_cvat_zip_track_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            zip_path = self._make_track_zip(tmp_path)
            dataset_root = tmp_path / "dataset"
            result = import_cvat_zip(
                zip_path,
                dataset_root,
                config_dir=tmp_path / "config",
            )
            self.assertEqual(result.train_images + result.val_images, 2)
            self.assertEqual(result.class_names, ["brailer_loaded"])
            self.assertTrue((dataset_root / "images" / "train").exists())
            train_labels = list((dataset_root / "labels" / "train").glob("*.txt"))
            self.assertGreaterEqual(len(train_labels), 1)
            label_text = train_labels[0].read_text(encoding="utf-8").strip()
            self.assertTrue(label_text.startswith("0 "))


if __name__ == "__main__":
    unittest.main()
