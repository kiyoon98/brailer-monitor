"""Parse and render YOLO segmentation labels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


CLASS_NAMES = {0: "brailer_loaded", 1: "brailer_empty", 2: "reference"}


@dataclass(frozen=True)
class SegLabel:
    class_id: int
    class_name: str
    polygon_norm: list[tuple[float, float]]

    def polygon_px(self, width: int, height: int) -> list[tuple[int, int]]:
        return [(int(x * width), int(y * height)) for x, y in self.polygon_norm]

    def bbox_px(self, width: int, height: int) -> tuple[int, int, int, int]:
        pts = self.polygon_px(width, height)
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        return min(xs), min(ys), max(xs), max(ys)

    def to_dict(self, width: int | None = None, height: int | None = None) -> dict:
        payload = {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "polygon_norm": [[x, y] for x, y in self.polygon_norm],
            "point_count": len(self.polygon_norm),
        }
        if width and height:
            payload["polygon_px"] = [[x, y] for x, y in self.polygon_px(width, height)]
            x1, y1, x2, y2 = self.bbox_px(width, height)
            payload["bbox_px"] = [x1, y1, x2, y2]
            area_px = _shoelace(self.polygon_px(width, height))
            payload["area_px"] = round(area_px, 1)
            payload["area_ratio"] = round(area_px / (width * height), 4)
        return payload


def _shoelace(points: list[tuple[int, int]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def parse_yolo_seg_line(line: str) -> SegLabel | None:
    parts = line.strip().split()
    if len(parts) < 7 or len(parts) % 2 != 1:
        return None
    class_id = int(float(parts[0]))
    coords = [float(v) for v in parts[1:]]
    polygon = [(coords[i], coords[i + 1]) for i in range(0, len(coords), 2)]
    return SegLabel(
        class_id=class_id,
        class_name=CLASS_NAMES.get(class_id, f"class_{class_id}"),
        polygon_norm=polygon,
    )


def load_frame_label(label_path: Path) -> SegLabel | None:
    if not label_path.exists():
        return None
    line = label_path.read_text(encoding="utf-8").strip().splitlines()
    if not line:
        return None
    return parse_yolo_seg_line(line[0])
