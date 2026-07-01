"""YOLO11-seg training utilities for custom brailer models."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class TrainingCancelled(Exception):
    """Raised when the user stops an in-progress training job."""


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


def deployment_weights_path(task_type: str, *, project_root: Path | None = None) -> Path:
    root = project_root or Path.cwd()
    name = _default_weights_name(task_type)
    return root / "models" / name


def reset_training_artifacts(*, project_root: Path | None = None) -> list[str]:
    """Remove trained weights and YOLO run folders so the next train starts fresh."""
    import shutil

    root = project_root or Path.cwd()
    deleted: list[str] = []

    models_dir = root / "models"
    for name in ("brailer_seg.pt", "brailer_detect.pt"):
        path = models_dir / name
        if path.exists():
            path.unlink()
            deleted.append(str(path.resolve()))

    for runs_root in (root / "runs" / "segment", root / "runs" / "detect"):
        if not runs_root.exists():
            continue
        run_dirs = sorted(
            (
                path
                for path in runs_root.rglob("*")
                if path.is_dir()
                and (path.name.startswith("brailer_seg") or path.name.startswith("brailer_detect"))
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for path in run_dirs:
            if not path.exists():
                continue
            shutil.rmtree(path)
            deleted.append(str(path.resolve()))

    return deleted


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
    on_batch_end: Callable[[int, int, int, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
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
    if on_epoch_end is not None or on_batch_end is not None or should_cancel is not None:

        batch_epoch = 0
        batch_index = 0

        def _report_epoch_progress(trainer: object) -> None:
            if on_epoch_end is None:
                return
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total = int(getattr(trainer, "epochs", epochs))
            on_epoch_end(epoch, total)

        def _report_batch_progress(trainer: object, index: int) -> None:
            if on_batch_end is None:
                return
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            total_epochs = int(getattr(trainer, "epochs", epochs))
            loader = getattr(trainer, "train_loader", None)
            try:
                total_batches = len(loader) if loader is not None else 0
            except TypeError:
                total_batches = 0
            on_batch_end(epoch, total_epochs, index, total_batches)

        def _epoch_start_callback(trainer: object) -> None:
            if should_cancel and should_cancel():
                setattr(trainer, "stop", True)
                return
            _report_epoch_progress(trainer)

        def _epoch_end_callback(trainer: object) -> None:
            if should_cancel and should_cancel():
                setattr(trainer, "stop", True)
                return
            _report_epoch_progress(trainer)

        def _batch_callback(trainer: object) -> None:
            nonlocal batch_epoch, batch_index
            if should_cancel and should_cancel():
                setattr(trainer, "stop", True)
                return
            epoch = int(getattr(trainer, "epoch", 0)) + 1
            if epoch != batch_epoch:
                batch_epoch = epoch
                batch_index = 0
            batch_index += 1
            _report_batch_progress(trainer, batch_index)

        model.add_callback("on_train_epoch_start", _epoch_start_callback)
        model.add_callback("on_train_epoch_end", _epoch_end_callback)
        model.add_callback("on_train_batch_end", _batch_callback)

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
        workers=2,
        cache=False,
    )
    if should_cancel and should_cancel():
        raise TrainingCancelled("Training cancelled by user")
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
