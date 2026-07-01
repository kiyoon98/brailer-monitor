"""YOLO-based brailer detector with TensorRT / PyTorch fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _detect_predictor_class() -> type:
    from ultralytics.models.yolo.detect import DetectionPredictor
    from ultralytics.utils import nms, ops

    class TupleUnwrappingDetectionPredictor(DetectionPredictor):
        def postprocess(self, preds: Any, img: Any, orig_imgs: Any, **kwargs: Any) -> Any:
            if isinstance(preds, (tuple, list)):
                preds = preds[0]
            preds = nms.non_max_suppression(
                preds,
                self.args.conf,
                kwargs.pop("iou", self.args.iou),
                self.args.classes,
                self.args.agnostic_nms,
                self.args.max_det,
                nc=len(self.model.names),
                end2end=getattr(self.model, "end2end", False),
                rotated=False,
            )
            if not isinstance(orig_imgs, list):
                orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
            return self.construct_results(preds, img, orig_imgs, **kwargs)

    return TupleUnwrappingDetectionPredictor


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
        imgsz: int = 416,
    ):
        self.model_path = Path(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.use_segmentation = use_segmentation
        self.imgsz = imgsz
        self._model = None
        self._detect_model = None
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

    def _load_detect_model(self) -> Any:
        model = self._load_model()
        if str(getattr(model, "task", "") or "").lower() == "detect":
            return model
        if self._detect_model is not None:
            return self._detect_model
        from ultralytics import YOLO

        resolved = self._resolve_model_path()
        self._detect_model = YOLO(str(resolved), task="detect")
        logger.info("Loaded YOLO detect fallback model from %s", resolved)
        return self._detect_model

    @staticmethod
    def _is_empty_mask_coeff_error(exc: RuntimeError) -> bool:
        message = str(exc)
        return (
            "mat1 and mat2 shapes cannot be multiplied" in message
            and "x0" in message
        )

    def _predict_task(self, model: Any) -> str:
        if not self.use_segmentation:
            return "detect"
        model_task = str(getattr(model, "task", "") or "").lower()
        if model_task == "detect":
            logger.warning(
                "Model %s is a detect model; using detect mode instead of segmentation.",
                self._resolved_path or self.model_path,
            )
            self.use_segmentation = False
            return "detect"
        return "segment"

    def _predict_with_task(self, model: Any, frame: np.ndarray, task: str) -> Any:
        predictor = None
        if task == "detect":
            model = self._load_detect_model()
            predictor = _detect_predictor_class()
            if getattr(model, "predictor", None) is not None and model.predictor.__class__ is not predictor:
                model.predictor = None
        return model.predict(
            source=frame,
            conf=self.confidence_threshold,
            device=self.device,
            imgsz=self.imgsz,
            half=not (isinstance(self.device, str) and self.device.lower() == "cpu"),
            verbose=False,
            task=task,
            predictor=predictor,
        )

    def predict(self, frame: np.ndarray) -> list[Detection]:
        model = self._load_model()
        task = self._predict_task(model)
        try:
            results = self._predict_with_task(model, frame, task)
        except RuntimeError as exc:
            if not self._is_empty_mask_coeff_error(exc):
                raise
            logger.warning(
                "Segmentation postprocess failed for %s; retrying in detect mode.",
                self._resolved_path or self.model_path,
            )
            self.use_segmentation = False
            results = self._predict_with_task(model, frame, "detect")
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
