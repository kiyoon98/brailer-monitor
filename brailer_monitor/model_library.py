"""Library of trained YOLO models so users can keep and reload multiple models."""

from __future__ import annotations

import json
import logging
import re
import shutil
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_ID_RE = re.compile(r"^\d{6}-\d{6}-[0-9a-f]{4}$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelRecord:
    id: str
    name: str
    task_type: str
    created_at: str
    weights_path: str
    epochs: int = 0
    class_names: list[str] = field(default_factory=list)
    train_images: int = 0
    val_images: int = 0
    metrics: dict[str, float] | None = None
    size_bytes: int = 0
    source: str = "train"
    dataset_frames: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ModelLibrary:
    """Stores trained model weights plus metadata under a library directory."""

    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _validate_model_id(model_id: str) -> str:
        if not MODEL_ID_RE.fullmatch(model_id):
            raise ValueError(f"Invalid model id: {model_id}")
        return model_id

    def _model_dir(self, model_id: str) -> Path:
        model_id = self._validate_model_id(model_id)
        return self.root / model_id

    def model_dir(self, model_id: str) -> Path:
        return self._model_dir(model_id)

    def _meta_path(self, model_id: str) -> Path:
        return self._model_dir(model_id) / "meta.json"

    @staticmethod
    def _new_id() -> str:
        stamp = datetime.now().strftime("%y%m%d-%H%M%S")
        return f"{stamp}-{uuid.uuid4().hex[:4]}"

    def register(
        self,
        weights_src: Path,
        *,
        task_type: str,
        name: str | None = None,
        epochs: int = 0,
        class_names: list[str] | None = None,
        train_images: int = 0,
        val_images: int = 0,
        metrics: dict[str, float] | None = None,
        dataset_frames: list[dict[str, Any]] | None = None,
        source: str = "train",
    ) -> ModelRecord:
        weights_src = Path(weights_src)
        if not weights_src.exists():
            raise FileNotFoundError(f"Weights not found: {weights_src}")

        model_id = self._new_id()
        directory = self._model_dir(model_id)
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / "weights.pt"
        target.write_bytes(weights_src.read_bytes())

        if not name:
            name = f"{task_type}-{model_id}"

        record = ModelRecord(
            id=model_id,
            name=name,
            task_type=task_type,
            created_at=_now_iso(),
            weights_path=str(target.resolve()),
            epochs=epochs,
            class_names=list(class_names or []),
            train_images=train_images,
            val_images=val_images,
            metrics=metrics,
            size_bytes=target.stat().st_size,
            source=source,
            dataset_frames=list(dataset_frames or []),
        )
        self._write_meta(record)
        logger.info("Registered model %s (%s) -> %s", model_id, name, target)
        return record

    def _write_meta(self, record: ModelRecord) -> None:
        self._meta_path(record.id).write_text(
            json.dumps(record.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _read_meta(self, model_id: str) -> ModelRecord:
        meta_path = self._meta_path(model_id)
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(ModelRecord)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return ModelRecord(**filtered)

    def get(self, model_id: str) -> ModelRecord:
        if not self._meta_path(model_id).exists():
            raise FileNotFoundError(f"Model not found: {model_id}")
        return self._read_meta(model_id)

    def exists(self, model_id: str | None) -> bool:
        if not model_id:
            return False
        try:
            self._validate_model_id(model_id)
        except ValueError:
            return False
        return self._meta_path(model_id).exists()

    def weights_path(self, model_id: str) -> Path:
        return Path(self.get(model_id).weights_path)

    def list_models(self) -> list[ModelRecord]:
        records: list[ModelRecord] = []
        if not self.root.exists():
            return records
        for directory in self.root.iterdir():
            if not (directory / "meta.json").exists():
                continue
            try:
                records.append(self._read_meta(directory.name))
            except Exception:
                logger.warning("Skipping unreadable model meta in %s", directory)
                continue
        records.sort(key=lambda record: record.created_at, reverse=True)
        return records

    def delete(self, model_id: str) -> bool:
        directory = self._model_dir(model_id)
        if not directory.exists():
            return False
        shutil.rmtree(directory)
        logger.info("Deleted model %s", model_id)
        return True

    def update_dataset_frames(
        self,
        model_id: str,
        dataset_frames: list[dict[str, Any]],
    ) -> ModelRecord:
        record = self.get(model_id)
        record.dataset_frames = list(dataset_frames)
        self._write_meta(record)
        return record

    def rename(self, model_id: str, name: str) -> ModelRecord:
        record = self.get(model_id)
        record.name = name.strip() or record.name
        self._write_meta(record)
        return record
