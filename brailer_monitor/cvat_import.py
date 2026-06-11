"""Import CVAT for video/images 1.1 ZIP exports into YOLO dataset layout."""

from __future__ import annotations

import json
import logging
import random
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import cv2

logger = logging.getLogger(__name__)

_FRAME_NUM_RE = re.compile(r"(\d+)")


@dataclass
class CvatAnnotation:
    label: str
    shape: str  # box | polygon
    frame_id: int
    box_xyxy: tuple[float, float, float, float] | None = None
    polygon_xy: list[tuple[float, float]] | None = None


@dataclass
class ImportResult:
    dataset_root: Path
    task_type: str  # detect | segment
    class_names: list[str]
    train_images: int
    val_images: int
    total_annotations: int
    yaml_path: Path
    meta_path: Path

    label_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "dataset_root": str(self.dataset_root),
            "task_type": self.task_type,
            "class_names": self.class_names,
            "train_images": self.train_images,
            "val_images": self.val_images,
            "total_annotations": self.total_annotations,
            "yaml_path": str(self.yaml_path),
            "meta_path": str(self.meta_path),
        }
        if self.label_summary is not None:
            payload["label_summary"] = self.label_summary
        return payload


def _parse_points(points: str) -> list[tuple[float, float]]:
    coords: list[tuple[float, float]] = []
    for pair in points.strip().split(";"):
        if not pair.strip():
            continue
        x_str, y_str = pair.split(",")
        coords.append((float(x_str), float(y_str)))
    return coords


def _box_from_points(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def _parse_box_element(elem: ET.Element) -> tuple[float, float, float, float]:
    return (
        float(elem.attrib["xtl"]),
        float(elem.attrib["ytl"]),
        float(elem.attrib["xbr"]),
        float(elem.attrib["ybr"]),
    )


def _is_visible(elem: ET.Element) -> bool:
    return elem.attrib.get("outside", "0") != "1"


def _parse_cvat_xml(xml_path: Path) -> tuple[list[CvatAnnotation], dict[int, tuple[int, int]], list[str]]:
    root = ET.parse(xml_path).getroot()
    labels: list[str] = []
    for label_elem in root.findall(".//labels/label"):
        name = label_elem.findtext("name")
        if name:
            labels.append(name.strip())

    frame_sizes: dict[int, tuple[int, int]] = {}
    for image_elem in root.findall("image"):
        frame_id = int(image_elem.attrib["id"])
        width = int(float(image_elem.attrib["width"]))
        height = int(float(image_elem.attrib["height"]))
        frame_sizes[frame_id] = (width, height)

    annotations: list[CvatAnnotation] = []

    for image_elem in root.findall("image"):
        frame_id = int(image_elem.attrib["id"])
        label_name = image_elem.attrib.get("label", "")
        for box in image_elem.findall("box"):
            if not _is_visible(box):
                continue
            annotations.append(
                CvatAnnotation(
                    label=box.attrib["label"],
                    shape="box",
                    frame_id=frame_id,
                    box_xyxy=_parse_box_element(box),
                )
            )
        for polygon in image_elem.findall("polygon"):
            if not _is_visible(polygon):
                continue
            pts = _parse_points(polygon.attrib["points"])
            annotations.append(
                CvatAnnotation(
                    label=polygon.attrib["label"],
                    shape="polygon",
                    frame_id=frame_id,
                    polygon_xy=pts,
                    box_xyxy=_box_from_points(pts),
                )
            )

    for track in root.findall("track"):
        track_label = track.attrib.get("label", "object")
        for box in track.findall("box"):
            if not _is_visible(box):
                continue
            frame_id = int(box.attrib["frame"])
            annotations.append(
                CvatAnnotation(
                    label=track_label,
                    shape="box",
                    frame_id=frame_id,
                    box_xyxy=_parse_box_element(box),
                )
            )
        for polygon in track.findall("polygon"):
            if not _is_visible(polygon):
                continue
            frame_id = int(polygon.attrib["frame"])
            pts = _parse_points(polygon.attrib["points"])
            annotations.append(
                CvatAnnotation(
                    label=track_label,
                    shape="polygon",
                    frame_id=frame_id,
                    polygon_xy=pts,
                    box_xyxy=_box_from_points(pts),
                )
            )

    if not labels:
        labels = sorted({ann.label for ann in annotations})

    return annotations, frame_sizes, labels


def _parse_cvat_label_meta(xml_path: Path) -> dict[str, dict[str, str | None]]:
    root = ET.parse(xml_path).getroot()
    meta: dict[str, dict[str, str | None]] = {}
    for label_elem in root.findall(".//labels/label"):
        name = label_elem.findtext("name")
        if not name:
            continue
        name = name.strip()
        meta[name] = {
            "color": (label_elem.findtext("color") or "").strip() or None,
            "type": (label_elem.findtext("type") or "").strip() or None,
        }
    return meta


def summarize_cvat_annotations(
    annotations: list[CvatAnnotation],
    class_names: list[str],
    label_meta: dict[str, dict[str, str | None]] | None = None,
) -> dict[str, Any]:
    """Build a per-label summary from parsed CVAT annotations."""
    label_meta = label_meta or {}
    names = list(class_names)
    for ann in annotations:
        if ann.label not in names:
            names.append(ann.label)

    per_label: dict[str, dict[str, Any]] = {}
    for name in names:
        meta = label_meta.get(name, {})
        per_label[name] = {
            "name": name,
            "color": meta.get("color"),
            "cvat_type": meta.get("type"),
            "annotation_count": 0,
            "box_count": 0,
            "polygon_count": 0,
            "frame_ids": set(),
        }

    for ann in annotations:
        entry = per_label.setdefault(
            ann.label,
            {
                "name": ann.label,
                "color": label_meta.get(ann.label, {}).get("color"),
                "cvat_type": label_meta.get(ann.label, {}).get("type"),
                "annotation_count": 0,
                "box_count": 0,
                "polygon_count": 0,
                "frame_ids": set(),
            },
        )
        entry["annotation_count"] += 1
        entry["frame_ids"].add(ann.frame_id)
        if ann.shape == "polygon":
            entry["polygon_count"] += 1
        else:
            entry["box_count"] += 1

    objects = []
    for name in names:
        entry = per_label[name]
        objects.append(
            {
                "name": entry["name"],
                "color": entry["color"],
                "cvat_type": entry["cvat_type"],
                "annotation_count": entry["annotation_count"],
                "box_count": entry["box_count"],
                "polygon_count": entry["polygon_count"],
                "frame_count": len(entry["frame_ids"]),
            }
        )

    has_polygon = any(ann.shape == "polygon" for ann in annotations)
    return {
        "class_names": names,
        "objects": objects,
        "total_annotations": len(annotations),
        "annotated_frames": len({ann.frame_id for ann in annotations}),
        "task_type": "segment" if has_polygon else "detect",
    }


def inspect_cvat(annotations_path: Path) -> dict[str, Any]:
    """Read CVAT 1.1 export and return defined labels with annotation counts."""
    if not annotations_path.exists():
        raise FileNotFoundError(f"CVAT annotations not found: {annotations_path}")

    with tempfile.TemporaryDirectory(prefix="cvat_inspect_") as tmp:
        work_dir = Path(tmp)
        xml_path, _ = _prepare_annotations_source(annotations_path, work_dir)
        annotations, _, class_names = _parse_cvat_xml(xml_path)
        label_meta = _parse_cvat_label_meta(xml_path)
        if not class_names and not label_meta:
            raise ValueError("No labels found in CVAT XML")
        if not class_names:
            class_names = sorted(label_meta.keys())
        return summarize_cvat_annotations(annotations, class_names, label_meta)


def _frame_number_from_name(name: str) -> int | None:
    stem = Path(name).stem
    match = _FRAME_NUM_RE.search(stem)
    if not match:
        return None
    return int(match.group(1))


def _find_xml_file(path: Path, extract_dir: Path | None = None) -> Path:
    if path.suffix.lower() == ".xml":
        return path
    search_root = extract_dir or path.parent
    xml_files = sorted(search_root.rglob("annotations.xml"))
    if not xml_files:
        xml_files = sorted(search_root.rglob("*.xml"))
    if not xml_files:
        raise ValueError(f"No annotations.xml found in {path}")
    return xml_files[0]


def _prepare_annotations_source(
    annotations_path: Path,
    work_dir: Path,
) -> tuple[Path, Path | None]:
    """Return (xml_path, extract_dir or None). Extracts zip into work_dir when needed."""
    suffix = annotations_path.suffix.lower()
    if suffix == ".xml":
        return annotations_path, None
    if suffix == ".zip" or zipfile.is_zipfile(annotations_path):
        extract_dir = work_dir / "_cvat_extract"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(annotations_path, "r") as archive:
            archive.extractall(extract_dir)
        return _find_xml_file(annotations_path, extract_dir), extract_dir
    raise ValueError(f"Unsupported annotations file: {annotations_path}")


def _extract_frames_from_video(
    video_path: Path,
    frame_ids: list[int],
    output_dir: Path,
) -> dict[int, Path]:
    """Extract annotated frames from the source video by CVAT frame index."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    mapping: dict[int, Path] = {}

    for frame_id in sorted(set(frame_ids)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            logger.warning("Failed to read frame %d from %s", frame_id, video_path)
            continue
        out_path = output_dir / f"frame_{frame_id:06d}.jpg"
        cv2.imwrite(str(out_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        mapping[frame_id] = out_path

    cap.release()
    if not mapping:
        raise ValueError(f"No frames could be extracted from video: {video_path}")
    return mapping


def _index_images(extract_dir: Path) -> dict[int, Path]:
    mapping: dict[int, Path] = {}
    image_dirs = [extract_dir / "images", extract_dir]
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for base in image_dirs:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.suffix.lower() not in extensions:
                continue
            frame_id = _frame_number_from_name(path.name)
            if frame_id is None:
                continue
            mapping.setdefault(frame_id, path)
    return mapping


def _yolo_detect_line(class_id: int, box: tuple[float, float, float, float], w: int, h: int) -> str:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _yolo_seg_line(class_id: int, points: list[tuple[float, float]], w: int, h: int) -> str:
    parts = [str(class_id)]
    for x, y in points:
        parts.append(f"{x / w:.6f}")
        parts.append(f"{y / h:.6f}")
    return " ".join(parts)


def _write_dataset_yaml(
    yaml_path: Path,
    dataset_root: Path,
    class_names: list[str],
    train_n: int,
    val_n: int,
    task_type: str,
) -> None:
    names_block = "\n".join(f"  {i}: {name}" for i, name in enumerate(class_names))
    content = f"""# Generated from CVAT 1.1 import
path: {dataset_root.resolve()}
train: images/train
val: images/val

names:
{names_block}

# task: {task_type}, train: {train_n}, val: {val_n}
"""
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(content, encoding="utf-8")


def import_cvat(
    annotations_path: Path,
    dataset_root: Path,
    *,
    video_path: Path | None = None,
    config_dir: Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = True,
) -> ImportResult:
    """Convert CVAT 1.1 export (.zip or .xml) to Ultralytics YOLO dataset."""
    if not annotations_path.exists():
        raise FileNotFoundError(f"CVAT annotations not found: {annotations_path}")

    work_dir = dataset_root / "_cvat_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    xml_path, extract_dir = _prepare_annotations_source(annotations_path, work_dir)
    annotations, frame_sizes, class_names = _parse_cvat_xml(xml_path)
    label_meta = _parse_cvat_label_meta(xml_path)
    label_summary = summarize_cvat_annotations(annotations, class_names, label_meta)
    if not annotations:
        raise ValueError("No annotations found in CVAT XML")

    image_map: dict[int, Path] = {}
    frame_source = "zip_images"
    if extract_dir is not None:
        image_map = _index_images(extract_dir)

    by_frame: dict[int, list[CvatAnnotation]] = {}
    for ann in annotations:
        by_frame.setdefault(ann.frame_id, []).append(ann)

    needed_frames = sorted(by_frame.keys())

    if not image_map:
        if video_path is None:
            raise ValueError(
                "CVAT export has no images. Upload the source video used for annotation "
                "(via --video CLI flag or the video field in the web UI)."
            )
        frames_dir = work_dir / "video_frames"
        image_map = _extract_frames_from_video(video_path, needed_frames, frames_dir)
        frame_source = "video"

    has_polygon = any(ann.shape == "polygon" for ann in annotations)
    task_type = "segment" if has_polygon else "detect"
    class_to_id = {name: idx for idx, name in enumerate(class_names)}

    frame_ids = sorted(fid for fid in needed_frames if fid in image_map)
    if not frame_ids:
        raise ValueError("Could not match CVAT frame ids to exported images")

    if clean:
        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            target = dataset_root / sub
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    shuffled = frame_ids.copy()
    random.shuffle(shuffled)
    val_count = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:val_count])

    train_n = val_n = 0
    total_ann = 0

    for frame_id in frame_ids:
        src_image = image_map[frame_id]
        split = "val" if frame_id in val_set else "train"
        stem = f"frame_{frame_id:06d}"
        dst_image = dataset_root / "images" / split / f"{stem}{src_image.suffix.lower()}"
        dst_label = dataset_root / "labels" / split / f"{stem}.txt"
        shutil.copy2(src_image, dst_image)

        import cv2

        img = cv2.imread(str(dst_image))
        if img is None:
            raise ValueError(f"Failed to read image: {dst_image}")
        height, width = img.shape[:2]

        lines: list[str] = []
        for ann in by_frame[frame_id]:
            class_id = class_to_id[ann.label]
            if task_type == "segment" and ann.polygon_xy:
                lines.append(_yolo_seg_line(class_id, ann.polygon_xy, width, height))
            elif ann.box_xyxy:
                lines.append(_yolo_detect_line(class_id, ann.box_xyxy, width, height))
        dst_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        total_ann += len(lines)

        if split == "val":
            val_n += 1
        else:
            train_n += 1

    config_dir = config_dir or dataset_root.parent.parent / "config"
    yaml_path = config_dir / "dataset.yaml"
    _write_dataset_yaml(yaml_path, dataset_root, class_names, train_n, val_n, task_type)

    meta_path = dataset_root / "import_meta.json"
    meta = {
        "source_annotations": str(annotations_path.resolve()),
        "source_video": str(video_path.resolve()) if video_path else None,
        "frame_source": frame_source,
        "task_type": task_type,
        "class_names": class_names,
        "label_summary": label_summary,
        "train_images": train_n,
        "val_images": val_n,
        "total_annotations": total_ann,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    shutil.rmtree(work_dir, ignore_errors=True)

    logger.info(
        "Imported CVAT dataset: %s (%s), train=%d val=%d annotations=%d",
        task_type,
        ", ".join(class_names),
        train_n,
        val_n,
        total_ann,
    )

    return ImportResult(
        dataset_root=dataset_root,
        task_type=task_type,
        class_names=class_names,
        train_images=train_n,
        val_images=val_n,
        total_annotations=total_ann,
        yaml_path=yaml_path,
        meta_path=meta_path,
        label_summary=label_summary,
    )


def import_cvat_zip(
    zip_path: Path,
    dataset_root: Path,
    *,
    video_path: Path | None = None,
    config_dir: Path | None = None,
    val_ratio: float = 0.2,
    seed: int = 42,
    clean: bool = True,
) -> ImportResult:
    """Backward-compatible alias for import_cvat."""
    return import_cvat(
        zip_path,
        dataset_root,
        video_path=video_path,
        config_dir=config_dir,
        val_ratio=val_ratio,
        seed=seed,
        clean=clean,
    )
