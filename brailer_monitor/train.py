"""YOLO11-seg training utilities for custom brailer models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


def _default_base_model(task_type: str) -> str:
    return "yolo11n-seg.pt" if task_type == "segment" else "yolo11n.pt"


def _default_output_name(task_type: str) -> str:
    return "brailer_seg" if task_type == "segment" else "brailer_detect"


def _default_weights_name(task_type: str) -> str:
    return "brailer_seg.pt" if task_type == "segment" else "brailer_detect.pt"


def load_task_type(dataset_root: Path | None = None) -> str:
    import json

    meta_path = (dataset_root or Path("data/dataset")) / "import_meta.json"
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8")).get("task_type", "segment")
    return "segment"


def run_training(
    dataset_yaml: Path,
    base_model: str | None = None,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str | int = 0,
    project: str | None = None,
    name: str | None = None,
    task_type: str | None = None,
    on_epoch_end: Callable[[int, int], None] | None = None,
) -> Path:
    """
    Train a segmentation model on labeled brailer images.

    Prepare data under data/dataset/:
      images/train, images/val
      labels/train, labels/val  (YOLO polygon format)
    Update config/dataset.yaml paths accordingly.
    """
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"Dataset config not found: {dataset_yaml}. "
            "Import CVAT zip or update config/dataset.yaml."
        )

    if task_type is None:
        task_type = load_task_type()
    if base_model is None:
        base_model = _default_base_model(task_type)
    if project is None:
        project = "runs/segment" if task_type == "segment" else "runs/detect"
    if name is None:
        name = _default_output_name(task_type)

    from ultralytics import YOLO

    model = YOLO(base_model)
    if on_epoch_end is not None:

        def _epoch_callback(trainer: object) -> None:
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", epochs))
            on_epoch_end(epoch, total)

        model.add_callback("on_train_epoch_end", _epoch_callback)

    logger.info(
        "Starting training: dataset=%s base=%s epochs=%d",
        dataset_yaml,
        base_model,
        epochs,
    )
    results = model.train(
        data=str(dataset_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=20,
        save=True,
        plots=True,
    )
    best = Path(project) / name / "weights" / "best.pt"
    if best.exists():
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        target = models_dir / _default_weights_name(task_type)
        target.write_bytes(best.read_bytes())
        logger.info("Copied best weights to %s", target)
        return target

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir) / "weights" / "best.pt"
    raise RuntimeError("Training finished but best.pt was not found")
