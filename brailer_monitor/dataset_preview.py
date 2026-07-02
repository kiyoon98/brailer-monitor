"""List imported YOLO dataset frames and render label overlays."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

_FRAME_STEM_RE = re.compile(r"frame_(\d+)")


@dataclass(frozen=True)
class DatasetObject:
    class_id: int
    class_name: str
    shape: str  # box | polygon
    polygon_norm: list[tuple[float, float]] | None = None
    bbox_norm: tuple[float, float, float, float] | None = None

    def to_dict(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        include_geometry: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "shape": self.shape,
        }
        if include_geometry:
            if self.polygon_norm:
                payload["polygon_norm"] = [[x, y] for x, y in self.polygon_norm]
            if self.bbox_norm:
                payload["bbox_norm"] = list(self.bbox_norm)
        if width and height:
            if self.polygon_norm:
                payload["polygon_px"] = [
                    [int(x * width), int(y * height)] for x, y in self.polygon_norm
                ]
            if self.bbox_norm:
                cx, cy, bw, bh = self.bbox_norm
                x1 = int((cx - bw / 2) * width)
                y1 = int((cy - bh / 2) * height)
                x2 = int((cx + bw / 2) * width)
                y2 = int((cy + bh / 2) * height)
                payload["bbox_px"] = [x1, y1, x2, y2]
        return payload


@dataclass(frozen=True)
class DatasetFrame:
    split: str
    stem: str
    frame_index: int
    image_name: str
    objects: tuple[DatasetObject, ...]
    width: int | None = None
    height: int | None = None

    def to_dict(self, *, include_geometry: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "split": self.split,
            "stem": self.stem,
            "frame_index": self.frame_index,
            "image_name": self.image_name,
            "preview_url": f"/api/pipeline/dataset/preview/{self.split}/{self.image_name}",
            "objects": [
                obj.to_dict(self.width, self.height, include_geometry=include_geometry)
                for obj in self.objects
            ],
        }
        if include_geometry and self.width and self.height:
            payload["width"] = self.width
            payload["height"] = self.height
        return payload


def load_class_names(dataset_root: Path) -> list[str]:
    meta_path = dataset_root / "import_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        names = meta.get("class_names")
        if names:
            return list(names)
    return ["object"]


def _frame_index_from_stem(stem: str) -> int:
    match = _FRAME_STEM_RE.search(stem)
    return int(match.group(1)) if match else 0


def _parse_label_line(line: str, class_names: list[str]) -> DatasetObject | None:
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(float(parts[0]))
    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
    coords = [float(v) for v in parts[1:]]

    if len(parts) == 5:
        cx, cy, bw, bh = coords
        return DatasetObject(
            class_id=class_id,
            class_name=class_name,
            shape="box",
            bbox_norm=(cx, cy, bw, bh),
        )

    if len(coords) % 2 != 0:
        return None
    polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return DatasetObject(
        class_id=class_id,
        class_name=class_name,
        shape="polygon",
        polygon_norm=polygon,
    )


def _load_label_objects(label_path: Path, class_names: list[str]) -> list[DatasetObject]:
    if not label_path.exists():
        return []
    objects: list[DatasetObject] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = _parse_label_line(line, class_names)
        if obj is not None:
            objects.append(obj)
    return objects


def _read_image_size(image_path: Path) -> tuple[int | None, int | None]:
    image = cv2.imread(str(image_path))
    if image is None:
        return None, None
    height, width = image.shape[:2]
    return width, height


def _iter_split_frames(
    dataset_root: Path,
    split: str,
    class_names: list[str],
    *,
    include_geometry: bool = False,
) -> list[DatasetFrame]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    if not image_dir.exists():
        return []

    frames: list[DatasetFrame] = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        stem = image_path.stem
        objects = _load_label_objects(label_dir / f"{stem}.txt", class_names)
        width, height = _read_image_size(image_path) if include_geometry else (None, None)
        frames.append(
            DatasetFrame(
                split=split,
                stem=stem,
                frame_index=_frame_index_from_stem(stem),
                image_name=image_path.name,
                objects=tuple(objects),
                width=width,
                height=height,
            )
        )
    return frames


def list_dataset_frames(
    dataset_root: Path,
    *,
    split: str = "all",
    offset: int = 0,
    limit: int = 60,
    include_geometry: bool = False,
) -> dict[str, Any]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_root}")

    class_names = load_class_names(dataset_root)
    splits = ["train", "val"] if split == "all" else [split]
    frames: list[DatasetFrame] = []
    for item in splits:
        frames.extend(
            _iter_split_frames(
                dataset_root,
                item,
                class_names,
                include_geometry=include_geometry,
            )
        )

    frames.sort(key=lambda frame: frame.frame_index)
    total = len(frames)
    page = frames[offset : offset + limit]
    return {
        "class_names": class_names,
        "total": total,
        "offset": offset,
        "limit": limit,
        "frames": [frame.to_dict(include_geometry=include_geometry) for frame in page],
    }


def resolve_dataset_image(dataset_root: Path, split: str, filename: str) -> Path:
    if split not in {"train", "val"}:
        raise ValueError(f"Invalid split: {split}")
    if "/" in filename or "\\" in filename or filename.startswith("."):
        raise ValueError("Invalid filename")
    path = (dataset_root / "images" / split / filename).resolve()
    root = (dataset_root / "images" / split).resolve()
    if not str(path).startswith(str(root)):
        raise ValueError("Invalid image path")
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {filename}")
    return path


def _color_for_class(class_id: int) -> tuple[int, int, int]:
    palette = [
        (59, 130, 246),
        (34, 197, 94),
        (234, 179, 8),
        (168, 85, 247),
        (239, 68, 68),
    ]
    return palette[class_id % len(palette)]


def render_dataset_preview(dataset_root: Path, split: str, filename: str) -> np.ndarray:
    image_path = resolve_dataset_image(dataset_root, split, filename)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    class_names = load_class_names(dataset_root)
    label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
    objects = _load_label_objects(label_path, class_names)
    height, width = image.shape[:2]
    vis = image.copy()

    for obj in objects:
        color = _color_for_class(obj.class_id)
        if obj.shape == "polygon" and obj.polygon_norm:
            pts = np.array(
                [[int(x * width), int(y * height)] for x, y in obj.polygon_norm],
                dtype=np.int32,
            )
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            vis = cv2.addWeighted(overlay, 0.35, vis, 0.65, 0)
            cv2.polylines(vis, [pts], True, color, 2)
            x1, y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
        elif obj.bbox_norm is not None:
            cx, cy, bw, bh = obj.bbox_norm
            x1 = int((cx - bw / 2) * width)
            y1 = int((cy - bh / 2) * height)
            x2 = int((cx + bw / 2) * width)
            y2 = int((cy + bh / 2) * height)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        else:
            continue
        cv2.putText(
            vis,
            obj.class_name,
            (x1, max(y1 - 6, 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return vis
