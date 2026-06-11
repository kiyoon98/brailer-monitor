"""YOLO-based brailer detector with TensorRT / PyTorch fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str
    mask: np.ndarray | None = None
    track_id: int | None = None


class BrailerDetector:
    """Wrap Ultralytics YOLO with automatic backend selection."""

    def __init__(
        self,
        model_path: str | Path,
        confidence_threshold: float = 0.35,
        device: str | int = 0,
        use_segmentation: bool = True,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_segmentation = use_segmentation
        self._model = None
        self._resolved_path: Path | None = None

    def _resolve_model_path(self) -> Path:
        if self._resolved_path is not None:
            return self._resolved_path

        path = self.model_path
        if path.exists():
            self._resolved_path = path
            return path

        candidates = [
            path,
            Path("models/brailer_seg.engine"),
            Path("models/brailer_seg.pt"),
            Path("models/brailer_detect.engine"),
            Path("models/brailer_detect.pt"),
        ]
        for candidate in candidates:
            if candidate.exists():
                logger.info("Using model: %s", candidate)
                self._resolved_path = candidate
                return candidate

        # Fall back to a pretrained nano seg model for development without custom weights.
        logger.warning(
            "Custom model not found at %s; falling back to yolo11n-seg.pt for development.",
            path,
        )
        self._resolved_path = Path("yolo11n-seg.pt")
        return self._resolved_path

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        from ultralytics import YOLO

        resolved = self._resolve_model_path()
        self._model = YOLO(str(resolved))
        logger.info("Loaded YOLO model from %s", resolved)
        return self._model

    def predict(self, frame: np.ndarray) -> list[Detection]:
        model = self._load_model()
        task = "segment" if self.use_segmentation else "detect"
        results = model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
            task=task,
        )
        return self._parse_results(results[0], frame.shape[:2])

    def track(self, frame: np.ndarray, tracker: str = "bytetrack.yaml") -> list[Detection]:
        model = self._load_model()
        results = model.track(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            persist=True,
            tracker=tracker,
            verbose=False,
        )
        return self._parse_results(results[0], frame.shape[:2])

    def _parse_results(self, result: Any, frame_shape: tuple[int, int]) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        names = result.names or {}
        masks = None
        if hasattr(result, "masks") and result.masks is not None:
            masks = result.masks.data.cpu().numpy()

        for index in range(len(boxes)):
            xyxy = boxes.xyxy[index].cpu().numpy().tolist()
            conf = float(boxes.conf[index].cpu().numpy())
            class_id = int(boxes.cls[index].cpu().numpy())
            class_name = str(names.get(class_id, f"class_{class_id}"))
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[index].cpu().numpy())
            mask = None
            if masks is not None and index < len(masks):
                mask = masks[index]
            detections.append(
                Detection(
                    bbox_xyxy=(xyxy[0], xyxy[1], xyxy[2], xyxy[3]),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                    mask=mask,
                    track_id=track_id,
                )
            )
        return detections

    @staticmethod
    def export_tensorrt(
        weights_path: str | Path,
        output_dir: str | Path = "models",
        imgsz: int = 640,
        device: int = 0,
    ) -> Path:
        """Export a .pt checkpoint to TensorRT .engine."""
        from ultralytics import YOLO

        weights = Path(weights_path)
        if not weights.exists():
            raise FileNotFoundError(f"Weights not found: {weights}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model = YOLO(str(weights))
        exported = model.export(format="engine", imgsz=imgsz, device=device)
        exported_path = Path(exported)
        target = output_dir / exported_path.name
        if exported_path.resolve() != target.resolve():
            target.write_bytes(exported_path.read_bytes())
        logger.info("Exported TensorRT engine to %s", target)
        return target
