"""Hybrid semantic/classical sea-area analysis with temporal encounter state."""

from __future__ import annotations

import json
import logging
import os
from collections import deque
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "sea_area.json"
DEFAULT_MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"
DEFAULT_MODEL_REVISION = "c30f696ac62fc44d5c37785c8d8b8f9c3ef09c16"
HYBRID_METHOD = "segformer_ade20k_hybrid_v1"
LEGACY_METHOD = "hsv_lab_grabcut_v3"

DEFAULT_CONFIG: dict[str, Any] = {
    "engine": "hybrid",
    "model_id": DEFAULT_MODEL_ID,
    "model_revision": DEFAULT_MODEL_REVISION,
    "semantic_half_precision": False,
    "semantic_input_size": 384,
    "semantic_sea_labels": ["water", "sea", "river", "lake"],
    "vessel_labels": ["boat", "ship"],
    "sky_labels": ["sky"],
    "semantic_mask_threshold": 0.35,
    "vessel_mask_threshold": 0.35,
    "semantic_weight": 0.75,
    "legacy_weight": 0.25,
    "hybrid_mask_threshold": 0.4,
    "quality_threshold": 0.45,
    "horizon_score_threshold": 0.35,
    "horizon_history": 5,
    "baseline_window_sec": 600.0,
    "baseline_min_duration_sec": 60.0,
    "baseline_min_samples": 15,
    "encounter_vessel_ratio": 0.005,
    "encounter_sea_drop_ratio": 0.20,
    "encounter_duration_sec": 10.0,
    "departure_vessel_ratio": 0.001,
    "departure_sea_drop_ratio": 0.10,
    "departure_duration_sec": 30.0,
}


def load_sea_area_config(path: Path | None = None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Invalid sea-area config: {config_path}")
        config.update(payload)
    return config


def _normalize_device(device: str | int) -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if isinstance(device, int):
        return f"cuda:{device}" if device >= 0 and torch.cuda.is_available() else "cpu"
    text = str(device).strip().lower()
    if text in {"cpu", "mps"}:
        return text
    if text.startswith("cuda"):
        return text if torch.cuda.is_available() else "cpu"
    if text.lstrip("-").isdigit():
        index = int(text)
        return f"cuda:{index}" if index >= 0 and torch.cuda.is_available() else "cpu"
    return text


class SemanticSeaSegmenter:
    """Lazy SegFormer inference wrapper returning semantic probabilities."""

    def __init__(self, config: dict[str, Any], device: str | int = "cpu") -> None:
        self.config = config
        self.device = _normalize_device(device)
        self.processor: Any | None = None
        self.model: Any | None = None
        self.torch: Any | None = None
        self.label_ids: dict[str, int] = {}

    def _load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

        model_id = str(self.config["model_id"])
        revision = str(self.config.get("model_revision") or "main")
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=revision, use_fast=True)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id, revision=revision)
        self.model.eval().to(self.device)
        if self.device.startswith("cuda") and bool(self.config.get("semantic_half_precision")):
            self.model.half()
        self.torch = torch
        labels = {str(label).strip().lower(): int(index) for index, label in self.model.config.id2label.items()}
        configured = set(
            self.config["semantic_sea_labels"] + self.config["vessel_labels"] + self.config["sky_labels"]
        )
        missing = sorted(label for label in configured if label not in labels)
        if missing:
            logger.warning("Sea model does not provide configured labels: %s", ", ".join(missing))
        if not any(label in labels for label in self.config["semantic_sea_labels"]):
            raise RuntimeError("Sea model does not provide any configured sea label")
        self.label_ids = labels

    def predict(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        self._load()
        assert self.processor is not None and self.model is not None and self.torch is not None
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_size = max(128, int(self.config.get("semantic_input_size") or 384))
        inputs = self.processor(
            images=rgb,
            return_tensors="pt",
            size={"height": input_size, "width": input_size},
        )
        pixel_values = inputs["pixel_values"].to(self.device)
        if self.device.startswith("cuda") and bool(self.config.get("semantic_half_precision")):
            pixel_values = pixel_values.half()
        with self.torch.inference_mode():
            logits = self.model(pixel_values=pixel_values).logits
            probabilities = logits.float().softmax(dim=1)[0]
            max_probability = probabilities.max(dim=0).values

        def combined(names: list[str]) -> Any:
            ids = [self.label_ids[name] for name in names if name in self.label_ids]
            if not ids:
                return self.torch.zeros_like(max_probability)
            return probabilities[ids].sum(dim=0)

        selected = self.torch.stack(
            (
                combined(list(self.config["semantic_sea_labels"])),
                combined(list(self.config["vessel_labels"])),
                combined(list(self.config["sky_labels"])),
                max_probability,
            )
        ).unsqueeze(0)
        selected = self.torch.nn.functional.interpolate(
            selected,
            size=frame.shape[:2],
            mode="bilinear",
            align_corners=False,
        )[0].detach().cpu().numpy()

        return {
            "sea": selected[0],
            "vessel": selected[1],
            "sky": selected[2],
            "max_probability": selected[3],
        }


class SeaEncounterTracker:
    """Turn noisy per-frame sea observations into stable encounter states."""

    def __init__(self, config: dict[str, Any], state_path: Path | None = None) -> None:
        self.config = config
        self.state_path = state_path
        self.state = "calibrating"
        self.baseline_samples: deque[tuple[float, float, float]] = deque()
        self.encounter_candidate_at: float | None = None
        self.departure_candidate_at: float | None = None
        self.last_timestamp: float | None = None
        if state_path is not None and state_path.exists():
            self._load_state()

    def _load_state(self) -> None:
        assert self.state_path is not None
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self.state = str(payload.get("state") or "calibrating")
            self.baseline_samples = deque(
                (float(item[0]), float(item[1]), float(item[2]) if len(item) >= 3 else 0.0)
                for item in payload.get("baseline_samples", [])
                if isinstance(item, list) and len(item) >= 2
            )
            self.encounter_candidate_at = payload.get("encounter_candidate_at")
            self.departure_candidate_at = payload.get("departure_candidate_at")
            self.last_timestamp = payload.get("last_timestamp")
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            logger.warning("Ignoring invalid sea state file: %s", self.state_path)

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        payload = {
            "state": self.state,
            "baseline_samples": [list(item) for item in self.baseline_samples],
            "encounter_candidate_at": self.encounter_candidate_at,
            "departure_candidate_at": self.departure_candidate_at,
            "last_timestamp": self.last_timestamp,
        }
        temp.write_text(json.dumps(payload), encoding="utf-8")
        os.replace(temp, self.state_path)

    def _baselines(self, timestamp: float) -> tuple[float | None, float | None]:
        window = float(self.config["baseline_window_sec"])
        while self.baseline_samples and self.baseline_samples[0][0] < timestamp - window:
            self.baseline_samples.popleft()
        if len(self.baseline_samples) < int(self.config["baseline_min_samples"]):
            return None, None
        if self.baseline_samples[-1][0] - self.baseline_samples[0][0] < float(
            self.config["baseline_min_duration_sec"]
        ):
            return None, None
        sea_values = np.asarray([ratio for _time, ratio, _vessel in self.baseline_samples], dtype=np.float32)
        vessel_values = np.asarray(
            [vessel for _time, _ratio, vessel in self.baseline_samples], dtype=np.float32
        )
        return float(np.percentile(sea_values, 95)), float(np.percentile(vessel_values, 95))

    def update(self, observation: dict[str, Any], timestamp_sec: float) -> dict[str, Any]:
        timestamp = float(timestamp_sec)
        if self.last_timestamp is not None and timestamp <= self.last_timestamp:
            timestamp = self.last_timestamp + 0.001
        self.last_timestamp = timestamp

        quality = str(observation.get("sea_quality") or "unknown")
        ratio = observation.get("sea_ratio")
        vessel_ratio = float(observation.get("vessel_ratio") or 0.0)
        baseline, vessel_baseline = self._baselines(timestamp)
        drop_ratio = None
        vessel_increase = None
        if baseline is not None and baseline > 0 and ratio is not None:
            drop_ratio = max(0.0, (baseline - float(ratio)) / baseline)
        if vessel_baseline is not None:
            vessel_increase = max(0.0, vessel_ratio - vessel_baseline)

        event: str | None = None
        if quality == "unknown" or ratio is None:
            self.encounter_candidate_at = None
            self.departure_candidate_at = None
            output_state = "unknown"
        else:
            encounter_evidence = baseline is not None and (
                (vessel_increase is not None and vessel_increase >= float(self.config["encounter_vessel_ratio"]))
                or (drop_ratio is not None and drop_ratio >= float(self.config["encounter_sea_drop_ratio"]))
            )
            if self.state != "encounter":
                if encounter_evidence:
                    if self.encounter_candidate_at is None:
                        self.encounter_candidate_at = timestamp
                    if timestamp - self.encounter_candidate_at >= float(self.config["encounter_duration_sec"]):
                        self.state = "encounter"
                        event = "encounter_start"
                        self.departure_candidate_at = None
                else:
                    self.encounter_candidate_at = None
                    if baseline is not None:
                        self.state = "open_sea"
            else:
                sea_recovered = drop_ratio is not None and drop_ratio <= float(
                    self.config["departure_sea_drop_ratio"]
                )
                if baseline is None:
                    sea_recovered = float(ratio) >= 0.2
                departure_evidence = (
                    vessel_increase is not None
                    and vessel_increase < float(self.config["departure_vessel_ratio"])
                    and sea_recovered
                )
                if departure_evidence:
                    if self.departure_candidate_at is None:
                        self.departure_candidate_at = timestamp
                    if timestamp - self.departure_candidate_at >= float(self.config["departure_duration_sec"]):
                        self.state = "open_sea"
                        event = "departure"
                        self.encounter_candidate_at = None
                else:
                    self.departure_candidate_at = None

            if self.state != "encounter" and not encounter_evidence:
                self.baseline_samples.append((timestamp, float(ratio), vessel_ratio))
                baseline, vessel_baseline = self._baselines(timestamp)
                if baseline is not None and baseline > 0:
                    drop_ratio = max(0.0, (baseline - float(ratio)) / baseline)
                if vessel_baseline is not None:
                    vessel_increase = max(0.0, vessel_ratio - vessel_baseline)
            output_state = self.state

        self._save_state()
        return {
            "sea_state": output_state,
            "sea_event": event,
            "sea_baseline_ratio": round(baseline, 4) if baseline is not None else None,
            "sea_drop_ratio": round(drop_ratio, 4) if drop_ratio is not None else None,
            "vessel_baseline_ratio": round(vessel_baseline, 4) if vessel_baseline is not None else None,
            "vessel_increase_ratio": round(vessel_increase, 4) if vessel_increase is not None else None,
        }


class SeaAreaAnalyzer:
    """Hybrid sea segmentation and temporal encounter analysis."""

    def __init__(
        self,
        *,
        device: str | int = "cpu",
        engine: str | None = None,
        config_path: Path | None = None,
        state_path: Path | None = None,
    ) -> None:
        self.config = load_sea_area_config(config_path)
        self.engine = str(engine or self.config.get("engine") or "hybrid")
        if self.engine not in {"hybrid", "legacy"}:
            raise ValueError(f"Unsupported sea engine: {self.engine}")
        self.segmenter = SemanticSeaSegmenter(self.config, device=device) if self.engine == "hybrid" else None
        self.tracker = SeaEncounterTracker(self.config, state_path=state_path)
        self.horizon_history: deque[int] = deque(maxlen=max(1, int(self.config["horizon_history"])))
        self.semantic_error: str | None = None

    @staticmethod
    def _legacy(frame: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        from .video_detect import sea_mask_for_frame

        return sea_mask_for_frame(frame)

    def _semantic_horizon(
        self,
        sea_probability: np.ndarray,
        sky_probability: np.ndarray,
    ) -> tuple[int | None, float]:
        height, width = sea_probability.shape[:2]
        left = int(width * 0.25)
        right = max(left + 1, int(width * 0.75))
        top = max(1, int(height * 0.05))
        bottom = max(top + 1, int(height * 0.70))
        sky_rows = sky_probability[:, left:right].mean(axis=1, dtype=np.float64)
        sea_rows = sea_probability[:, left:right].mean(axis=1, dtype=np.float64)
        sky_prefix = np.cumsum(sky_rows)
        sea_suffix = np.cumsum(sea_rows[::-1])[::-1]
        candidates = np.arange(top, bottom)
        sky_scores = sky_prefix[candidates - 1] / candidates
        sea_scores = sea_suffix[candidates] / np.maximum(height - candidates, 1)
        scores = (sky_scores + sea_scores) / 2.0
        best_index = int(np.argmax(scores))
        best_y = int(candidates[best_index])
        best_score = float(scores[best_index])
        if best_score < float(self.config["horizon_score_threshold"]):
            return None, best_score
        return best_y, best_score

    @staticmethod
    def _dark_frame(frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean = float(gray.mean())
        std = float(gray.std())
        p90 = float(np.percentile(gray, 90))
        return (mean < 35.0 and p90 < 70.0) or (mean < 45.0 and std < 18.0)

    def analyze(self, frame: np.ndarray, *, timestamp_sec: float = 0.0) -> dict[str, Any]:
        legacy_mask, legacy_stats = self._legacy(frame)
        frame_h, frame_w = frame.shape[:2]
        legacy_horizon = int(legacy_stats.get("sea_horizon_y") or 0)
        method = LEGACY_METHOD

        if self.segmenter is None or self.semantic_error is not None:
            stats = dict(legacy_stats)
            quality = "legacy" if not self._dark_frame(frame) else "unknown"
            if quality == "unknown":
                stats["sea_ratio"] = None
                stats["sea_percent"] = None
                stats["sea_area_px"] = 0
            stats.update(
                {
                    "sea_method": method,
                    "semantic_sea_ratio": None,
                    "legacy_sea_ratio": stats.get("sea_ratio"),
                    "vessel_ratio": 0.0,
                    "sea_confidence": None,
                    "sea_quality": quality,
                    "sea_fallback_reason": self.semantic_error,
                }
            )
            stats.update(self.tracker.update(stats, timestamp_sec))
            return stats

        try:
            semantic = self.segmenter.predict(frame)
        except Exception as exc:
            self.semantic_error = f"{type(exc).__name__}: {exc}"
            logger.warning("Semantic sea model unavailable; using legacy analysis: %s", self.semantic_error)
            failed_segmenter = self.segmenter
            self.segmenter = None
            if failed_segmenter is not None:
                failed_segmenter.model = None
                failed_segmenter.processor = None
            return self.analyze(frame, timestamp_sec=timestamp_sec)

        sea_probability = semantic["sea"]
        vessel_probability = semantic["vessel"]
        sky_probability = semantic["sky"]
        max_probability = semantic["max_probability"]
        semantic_horizon, horizon_score = self._semantic_horizon(sea_probability, sky_probability)
        horizon = semantic_horizon if semantic_horizon is not None else legacy_horizon
        self.horizon_history.append(max(0, min(frame_h - 1, int(horizon))))
        horizon = int(median(self.horizon_history))

        valid_roi = np.zeros((frame_h, frame_w), dtype=bool)
        valid_roi[horizon:, :] = True
        valid_count = max(1, int(np.count_nonzero(valid_roi)))
        semantic_mask = (sea_probability >= float(self.config["semantic_mask_threshold"])) & valid_roi
        legacy_roi_mask = legacy_mask.astype(bool) & valid_roi
        fused_score = (
            float(self.config["semantic_weight"]) * sea_probability
            + float(self.config["legacy_weight"]) * legacy_roi_mask.astype(np.float32)
        )
        sea_mask = (fused_score >= float(self.config["hybrid_mask_threshold"])) & valid_roi
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        sea_mask = cv2.morphologyEx(sea_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel).astype(bool)

        semantic_ratio = float(np.count_nonzero(semantic_mask)) / valid_count
        legacy_ratio = float(np.count_nonzero(legacy_roi_mask)) / valid_count
        sea_ratio = float(np.count_nonzero(sea_mask)) / valid_count
        vessel_mask = (vessel_probability >= float(self.config["vessel_mask_threshold"])) & valid_roi
        vessel_ratio = float(np.count_nonzero(vessel_mask)) / valid_count
        intersection = int(np.count_nonzero(semantic_mask & legacy_roi_mask))
        union = int(np.count_nonzero(semantic_mask | legacy_roi_mask))
        agreement = intersection / union if union else 1.0
        semantic_confidence = float(max_probability[valid_roi].mean()) if np.any(valid_roi) else 0.0
        confidence = 0.7 * semantic_confidence + 0.3 * agreement
        quality = "unknown" if self._dark_frame(frame) or confidence < float(self.config["quality_threshold"]) else "good"

        stats = {
            "sea_ratio": round(sea_ratio, 4) if quality != "unknown" else None,
            "sea_percent": round(sea_ratio * 100.0, 2) if quality != "unknown" else None,
            "sea_area_px": int(np.count_nonzero(sea_mask)) if quality != "unknown" else 0,
            "sea_method": HYBRID_METHOD,
            "sea_horizon_y": horizon,
            "sea_roi_xyxy": [0, horizon, frame_w, frame_h],
            "sea_candidate_area_px": int(np.count_nonzero(semantic_mask)),
            "semantic_sea_ratio": round(semantic_ratio, 4),
            "legacy_sea_ratio": round(legacy_ratio, 4),
            "vessel_ratio": round(vessel_ratio, 4),
            "sea_confidence": round(confidence, 4),
            "sea_quality": quality,
            "sea_fallback_reason": None,
            "sea_horizon_score": round(horizon_score, 4),
        }
        stats.update(self.tracker.update(stats, timestamp_sec))
        return stats
