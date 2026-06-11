"""YOLO11-seg training utilities for custom brailer models."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_training(
    dataset_yaml: Path,
    base_model: str = "yolo11n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 8,
    device: str | int = 0,
    project: str = "runs/segment",
    name: str = "brailer_seg",
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
            "Label images with CVAT/Roboflow and update config/dataset.yaml."
        )

    from ultralytics import YOLO

    model = YOLO(base_model)
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
        target = models_dir / "brailer_seg.pt"
        target.write_bytes(best.read_bytes())
        logger.info("Copied best weights to %s", target)
        return target

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        return Path(save_dir) / "weights" / "best.pt"
    raise RuntimeError("Training finished but best.pt was not found")
