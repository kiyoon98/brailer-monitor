"""CVAT import, YOLO training, and video detection job management."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cvat_import import import_cvat
from ..dataset_preview import list_dataset_frames, render_dataset_preview
from ..detect_timeline import (
    compact_timeline_segments,
    get_segment_frames,
    list_timeline,
    merge_frame_detection,
    merge_job_manifest,
    reset_timeline,
    timeline_summary,
)
from ..lake_video_source import download_video
from ..model_library import ModelLibrary
from ..train import TrainingCancelled, load_task_type, reset_training_artifacts, run_training
from ..video_detect import DetectionCancelled
from ..video_time import parse_video_start_time

logger = logging.getLogger(__name__)
SAVE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_cuda_oom_error(message: str) -> bool:
    lowered = message.lower()
    return "cuda" in lowered and (
        "out of memory" in lowered
        or "memoryallocation" in lowered
        or "cudaerrormemoryallocation" in lowered
    )


def _is_cpu_device(device: str | int) -> bool:
    return isinstance(device, str) and device.lower() == "cpu"


def _safe_saved_result_id(name: str) -> str:
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip().lower()).strip(".-")
    if not base:
        base = "result"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{base[:40]}"


def _copy_or_link(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


@dataclass
class PipelineState:
    import_status: str = "idle"
    import_result: dict[str, Any] | None = None
    import_error: str | None = None
    train_status: str = "idle"
    train_progress: str | None = None
    train_epoch: int = 0
    train_epochs: int = 0
    train_batch: int = 0
    train_batches: int = 0
    train_progress_pct: float = 0.0
    train_weights: str | None = None
    train_error: str | None = None
    active_model_id: str | None = None
    active_model_name: str | None = None
    detect_status: str = "idle"
    detect_job_id: str | None = None
    detect_video_name: str | None = None
    detect_progress_pct: float = 0.0
    detect_processed_frames: int = 0
    detect_total_frames: int = 0
    detect_frames_with_objects: int = 0
    detect_queue_pending: int = 0
    detect_batch_total: int = 0
    detect_batch_done: int = 0
    detect_batch_index: int = 0
    detect_timeline_events: int = 0
    detect_overlay_job_id: str | None = None
    detect_overlay_preview_path: str | None = None
    detect_overlay_frame_index: int | None = None
    detect_overlay_timestamp_sec: float | None = None
    detect_overlay_width: int = 0
    detect_overlay_height: int = 0
    detect_overlay_detections: list[dict[str, Any]] | None = None
    detect_overlay_updated_at: str | None = None
    detect_error: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_is_active(state: PipelineState, *, threads: dict[str, threading.Thread]) -> bool:
    return (
        state.detect_status in {"running", "cancelling"}
        or (state.detect_queue_pending or 0) > 0
        or any(thread.is_alive() for thread in threads.values())
    )

@dataclass
class DetectJob:
    job_id: str
    video_name: str
    created_at: str
    status: str = "pending"
    progress: float = 0.0
    processed_frames: int = 0
    total_frames: int = 0
    frames_with_detections: int = 0
    manifest_path: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class DetectPipelineManager:
    def __init__(self, root: Path, dataset_root: Path, config_dir: Path):
        self.root = root
        self.dataset_root = dataset_root
        self.config_dir = config_dir
        self.root.mkdir(parents=True, exist_ok=True)
        self.model_library = ModelLibrary(self.config_dir.parent / "models" / "library")
        self._state_path = self.root / "pipeline_state.json"
        self._lock = threading.Lock()
        self._train_thread: threading.Thread | None = None
        self._detect_threads: dict[str, threading.Thread] = {}
        self._detect_procs: dict[str, subprocess.Popen] = {}
        self._timeline_path = self.root / "detect_timeline.json"
        self._detect_queue: list[dict[str, Any]] = []
        self._batch_total = 0
        self._batch_done = 0
        self._batch_failed = 0
        self._detect_cancel = threading.Event()
        self._train_cancel = threading.Event()
        self._saved_results_dir = self.root / "saved_results"
        self._restore_backups_dir = self.root / "restore_backups"

    def _detection_cancelled(self) -> bool:
        return self._detect_cancel.is_set()

    def _clear_detection_cancel(self) -> None:
        self._detect_cancel.clear()

    def _clear_train_cancel(self) -> None:
        self._train_cancel.clear()

    def _request_detect_worker_stop(self) -> None:
        state = self._load_state()
        job_id = state.detect_job_id
        if not job_id:
            return
        try:
            (self._job_dir(job_id) / "stop.txt").write_text("stop", encoding="utf-8")
        except OSError:
            pass

    def cancel_detection(self) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if state.detect_status == "cancelled":
                return {"cancelled": False, "reason": "already_cancelled"}
            if not _detect_is_active(state, threads=self._detect_threads):
                return {"cancelled": False, "reason": "not_running"}

            self._detect_cancel.set()
            self._request_detect_worker_stop()
            self._detect_queue.clear()
            state.detect_status = "cancelling"
            state.detect_queue_pending = 0
            state.detect_error = "중지 요청됨"
            self._save_state(state)
        self._recover_stale_cancel()
        return {"cancelled": True}

    def cancel_training(self) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if state.train_status not in ("running", "cancelling"):
                return {"cancelled": False, "reason": "not_running"}
            if not self._train_thread_alive():
                self._finish_stale_train()
                return {"cancelled": True, "reason": "stale_recovered"}

            self._train_cancel.set()
            state.train_progress = "cancelling"
            self._save_state(state)
            return {"cancelled": True}

    def _finish_stale_train(self) -> None:
        state = self._load_state()
        state.train_status = "cancelled"
        state.train_progress = "cancelled"
        state.train_error = "학습 세션이 끊겨 중지됨 (서버 재시작 등)"
        self._save_state(state)

    def _recover_stale_train(self) -> None:
        state = self._load_state()
        if state.train_status not in ("running", "cancelling"):
            return
        if self._train_thread_alive():
            return
        with self._lock:
            state = self._load_state()
            if state.train_status in ("running", "cancelling") and not self._train_thread_alive():
                logger.warning("Recovering stale train_status=%s (thread not alive)", state.train_status)
                self._finish_stale_train()

    def reset_training(self) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if state.train_status == "running" and self._train_thread_alive():
                raise RuntimeError("학습이 진행 중입니다. 먼저 중지하세요.")
            if _detect_is_active(state, threads=self._detect_threads):
                raise RuntimeError("탐지가 진행 중입니다. 먼저 중지하세요.")

            project_root = self.config_dir.parent
            deleted = reset_training_artifacts(project_root=project_root)

            state.train_status = "idle"
            state.train_progress = None
            state.train_epoch = 0
            state.train_epochs = 0
            state.train_progress_pct = 0.0
            state.train_weights = None
            state.train_error = None
            state.active_model_id = None
            state.active_model_name = None
            self._save_state(state)
            return {"reset": True, "deleted": deleted}

    def reset_and_start_training(
        self,
        *,
        epochs: int = 50,
        batch: int = 8,
        imgsz: int = 640,
        device: str | int = 0,
    ) -> dict[str, Any]:
        reset_info = self.reset_training()
        started = self.start_training(epochs=epochs, batch=batch, imgsz=imgsz, device=device)
        return {**reset_info, **started}

    def _dataset_summary(self) -> dict[str, Any]:
        meta_path = self.dataset_root / "import_meta.json"
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _register_trained_model(self, weights: Path, epochs: int):
        try:
            meta = self._dataset_summary()
            task_type = meta.get("task_type") or load_task_type(self.dataset_root)
            class_names = (
                meta.get("class_names")
                or meta.get("label_summary", {}).get("class_names")
                or []
            )
            record = self.model_library.register(
                weights,
                task_type=task_type,
                epochs=epochs,
                class_names=class_names,
                train_images=int(meta.get("train_images", 0) or 0),
                val_images=int(meta.get("val_images", 0) or 0),
                dataset_frames=[],
                source="train",
            )
            dataset_frames = self._snapshot_dataset_frame_samples(record.id, limit=48)
            if dataset_frames:
                record = self.model_library.update_dataset_frames(record.id, dataset_frames)
            return record
        except Exception:
            logger.exception("Failed to register trained model into library")
            return None

    def _dataset_frame_samples(self, *, limit: int = 48) -> list[dict[str, Any]]:
        try:
            result = list_dataset_frames(self.dataset_root, split="all", offset=0, limit=limit)
        except Exception:
            logger.exception("Failed to collect dataset frame samples")
            return []
        return list(result.get("frames") or [])

    def _snapshot_dataset_frame_samples(self, model_id: str, *, limit: int = 48) -> list[dict[str, Any]]:
        try:
            result = list_dataset_frames(
                self.dataset_root,
                split="all",
                offset=0,
                limit=limit,
                include_geometry=True,
            )
        except Exception:
            logger.exception("Failed to collect dataset frame samples")
            return []

        preview_root = self.model_library.model_dir(model_id) / "previews"
        frames = list(result.get("frames") or [])
        for frame in frames:
            split = str(frame.get("split") or "")
            image_name = str(frame.get("image_name") or "")
            if split not in {"train", "val"} or not image_name or "/" in image_name or "\\" in image_name:
                continue
            try:
                import cv2

                preview = render_dataset_preview(self.dataset_root, split, image_name)
                preview_dir = preview_root / split
                preview_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(preview_dir / image_name), preview)
                frame["preview_url"] = f"/api/pipeline/models/{model_id}/preview/{split}/{image_name}"
            except Exception:
                logger.exception("Failed to snapshot dataset preview for model %s: %s/%s", model_id, split, image_name)
        return frames

    def resolve_model_preview(self, model_id: str, split: str, filename: str) -> Path:
        if split not in {"train", "val"}:
            raise ValueError(f"Invalid split: {split}")
        if "/" in filename or "\\" in filename or filename.startswith("."):
            raise ValueError("Invalid filename")
        root = (self.model_library.model_dir(model_id) / "previews" / split).resolve()
        path = (root / filename).resolve()
        if not str(path).startswith(str(root)):
            raise ValueError("Invalid image path")
        if not path.exists():
            raise FileNotFoundError(f"Model preview not found: {filename}")
        return path

    def list_models(self) -> dict[str, Any]:
        state = self._load_state()
        models = [record.to_dict() for record in self.model_library.list_models()]
        active_id = state.active_model_id
        if active_id and not self.model_library.exists(active_id):
            active_id = None
        return {"models": models, "active_id": active_id}

    def activate_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if state.train_status == "running" and self._train_thread_alive():
                raise RuntimeError("학습이 진행 중입니다. 먼저 중지하세요.")
            if _detect_is_active(state, threads=self._detect_threads):
                raise RuntimeError("탐지가 진행 중입니다. 먼저 중지하세요.")

            record = self.model_library.get(model_id)
            weights = Path(record.weights_path)
            if not weights.exists():
                raise FileNotFoundError(f"모델 가중치 파일이 없습니다: {weights}")

            state.active_model_id = record.id
            state.active_model_name = record.name
            state.train_weights = record.weights_path
            state.train_status = "completed"
            state.train_progress = "done"
            state.train_epochs = record.epochs
            state.train_epoch = record.epochs
            state.train_progress_pct = 1.0
            state.train_error = None
            self._save_state(state)
            return {"activated": True, "model": record.to_dict()}

    def model_frames(self, model_id: str, *, limit: int = 48) -> dict[str, Any]:
        record = self.model_library.get(model_id)
        frames = list(record.dataset_frames or [])
        return {
            "model": record.to_dict(),
            "source": "model",
            "total": len(frames),
            "frames": frames[:limit],
        }

    def delete_model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            deleted = self.model_library.delete(model_id)
            if state.active_model_id == model_id:
                state.active_model_id = None
                state.active_model_name = None
                state.train_weights = None
                state.train_status = "idle"
                state.train_progress = None
                state.train_epoch = 0
                state.train_epochs = 0
                state.train_progress_pct = 0.0
                self._save_state(state)
            return {"deleted": deleted}

    def rename_model(self, model_id: str, name: str) -> dict[str, Any]:
        with self._lock:
            record = self.model_library.rename(model_id, name)
            state = self._load_state()
            if state.active_model_id == model_id:
                state.active_model_name = record.name
                self._save_state(state)
            return {"renamed": True, "model": record.to_dict()}

    def _active_weights(self, state: PipelineState) -> Path | None:
        if state.active_model_id and self.model_library.exists(state.active_model_id):
            weights = self.model_library.weights_path(state.active_model_id)
            if weights.exists():
                return weights
        if state.train_weights and Path(state.train_weights).exists():
            return Path(state.train_weights)
        return None

    def _detection_model_specs(
        self,
        state: PipelineState,
        *,
        model_path: Path | None = None,
        model_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []
        for model_id in model_ids or []:
            record = self.model_library.get(model_id)
            weights = Path(record.weights_path)
            if not weights.exists():
                raise FileNotFoundError(f"모델 가중치 파일이 없습니다: {weights}")
            specs.append(
                {
                    "id": record.id,
                    "name": record.name,
                    "path": str(weights.resolve()),
                    "task_type": record.task_type,
                }
            )
        if specs:
            return specs

        if model_path is None:
            model_path = self._active_weights(state)
        if model_path is None:
            task = load_task_type(self.dataset_root)
            model_path = Path("models/brailer_detect.pt" if task == "detect" else "models/brailer_seg.pt")
        if not model_path.exists():
            raise FileNotFoundError(
                "학습된 모델이 없습니다. 먼저 학습하거나 모델 라이브러리에서 모델을 선택하세요."
            )

        active_id = state.active_model_id if self.model_library.exists(state.active_model_id) else None
        if active_id and Path(self.model_library.weights_path(active_id)).resolve() == model_path.resolve():
            record = self.model_library.get(active_id)
            return [
                {
                    "id": record.id,
                    "name": record.name,
                    "path": str(model_path.resolve()),
                    "task_type": record.task_type,
                }
            ]
        return [{"id": "model_1", "name": model_path.stem, "path": str(model_path.resolve())}]

    @staticmethod
    def _clear_detect_overlay(state: PipelineState) -> None:
        state.detect_overlay_job_id = None
        state.detect_overlay_preview_path = None
        state.detect_overlay_frame_index = None
        state.detect_overlay_timestamp_sec = None
        state.detect_overlay_width = 0
        state.detect_overlay_height = 0
        state.detect_overlay_detections = None
        state.detect_overlay_updated_at = None

    def _finish_detection_cancelled(self) -> None:
        state = self._load_state()
        state.detect_status = "cancelled"
        state.detect_job_id = None
        state.detect_queue_pending = 0
        state.detect_video_name = None
        state.detect_batch_index = 0
        state.detect_error = "사용자가 중지함"
        state.detect_batch_total = self._batch_total
        state.detect_batch_done = self._batch_done
        summary = timeline_summary(self._timeline_path)
        state.detect_timeline_events = summary["event_count"]
        self._clear_detect_overlay(state)
        self._save_state(state)

    def _recover_stale_cancel(self) -> None:
        state = self._load_state()
        if state.detect_status != "cancelling":
            return
        if any(thread.is_alive() for thread in self._detect_threads.values()):
            return
        with self._lock:
            state = self._load_state()
            if state.detect_status == "cancelling":
                self._finish_detection_cancelled()

    def _detect_threads_alive(self) -> bool:
        return any(thread.is_alive() for thread in self._detect_threads.values())

    def _detect_session_alive(self) -> bool:
        return self._detect_threads_alive() or bool(self._detect_queue)

    def _finish_stale_detect(self) -> None:
        state = self._load_state()
        state.detect_status = "cancelled"
        state.detect_job_id = None
        state.detect_queue_pending = 0
        state.detect_video_name = None
        state.detect_batch_index = 0
        state.detect_error = "탐지 세션이 끊겨 중지됨 (서버 재시작 등)"
        state.detect_batch_total = self._batch_total
        state.detect_batch_done = self._batch_done
        summary = timeline_summary(self._timeline_path)
        state.detect_timeline_events = summary["event_count"]
        self._clear_detect_overlay(state)
        self._save_state(state)

    def _recover_stale_detect(self) -> None:
        state = self._load_state()
        if state.detect_status not in ("running", "cancelling"):
            return
        if self._detect_session_alive():
            return
        with self._lock:
            state = self._load_state()
            if state.detect_status in ("running", "cancelling") and not self._detect_session_alive():
                logger.warning("Recovering stale detect_status=%s (no active threads/queue)", state.detect_status)
                self._detect_queue.clear()
                self._finish_stale_detect()

    def _recover_crashed_detect_worker(self) -> None:
        state = self._load_state()
        job_id = state.detect_job_id
        if state.detect_status != "running" or not job_id:
            return

        job_dir = self._job_dir(job_id)
        error_file = job_dir / "worker_error.txt"
        if not error_file.exists():
            return
        try:
            if time.time() - error_file.stat().st_mtime < 2:
                return
            detail = error_file.read_text(encoding="utf-8").strip()
        except OSError:
            return

        try:
            job = self.get_detect_job(job_id)
        except FileNotFoundError:
            return
        if job.status != "running":
            return

        proc = self._detect_procs.pop(job_id, None)
        if proc is not None:
            self._terminate_process(proc)

        if _is_cuda_oom_error(detail):
            template = self._detect_queue[0].copy() if self._detect_queue else {}
            model_path = template.get("model_path") or self._active_weights(state)
            if model_path is not None:
                retry_item = {
                    "video_name": state.detect_video_name or job.video_name,
                    "video_path": job_dir / "video.mp4",
                    "model_path": model_path,
                    "frame_stride": int(template.get("frame_stride", 5)),
                    "confidence": float(template.get("confidence", 0.6)),
                    "imgsz": int(template.get("imgsz", 416)),
                    "use_sam": bool(template.get("use_sam", False)),
                    "device": "cpu",
                    "batch_index": state.detect_batch_index or 0,
                }
                job.status = "error"
                job.error = "GPU 메모리 부족으로 CPU에서 재시도합니다."
                self._save_job(job)
                self._detect_queue.insert(0, retry_item)
                state.detect_job_id = None
                state.detect_error = "GPU 메모리 부족으로 CPU에서 재시도 중입니다."
                state.detect_queue_pending = len(self._detect_queue)
                self._save_state(state)
                logger.warning("Recovered crashed CUDA OOM job %s; retrying on CPU", job_id)
                self._start_next_queued_or_finish(success=False)
                return

        job.status = "error"
        job.error = detail
        self._save_job(job)
        self._batch_done += 1
        self._batch_failed += 1
        state.detect_batch_done = self._batch_done
        state.detect_error = detail
        self._save_state(state)
        logger.error("Recovered crashed detection worker for job %s", job_id)
        self._start_next_queued_or_finish(success=False)

    def _recover_completed_detect_worker(self) -> None:
        state = self._load_state()
        job_id = state.detect_job_id
        if state.detect_status != "running" or not job_id:
            return

        job_dir = self._job_dir(job_id)
        manifest_path = job_dir / "detections.json"
        if not manifest_path.exists():
            return

        try:
            job = self.get_detect_job(job_id)
        except FileNotFoundError:
            return
        if job.status != "running":
            return

        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        progress_file = job_dir / "progress.json"
        if progress_file.exists():
            try:
                progress = json.loads(progress_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                progress = {}
            processed = int(progress.get("processed", 0))
            total = int(progress.get("total", 0))
            if total > 0 and processed < total:
                return

        video_name = state.detect_video_name or job.video_name
        manifest["video_name"] = video_name
        video_start = parse_video_start_time(video_name)
        if video_start:
            manifest["video_start"] = video_start.isoformat()
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

        merge_job_manifest(
            self._timeline_path,
            job_id=job_id,
            video_name=video_name,
            manifest=manifest,
        )

        job.status = "completed"
        job.progress = 1.0
        job.processed_frames = int(manifest.get("frames_processed", 0))
        job.total_frames = job.processed_frames
        job.frames_with_detections = int(manifest.get("frames_with_detections", 0))
        job.manifest_path = str(manifest_path.resolve())
        self._save_job(job)

        self._batch_done = max(self._batch_done, state.detect_batch_done) + 1
        summary = timeline_summary(self._timeline_path)
        state.detect_progress_pct = 1.0
        state.detect_processed_frames = job.processed_frames
        state.detect_total_frames = job.total_frames
        state.detect_frames_with_objects = job.frames_with_detections
        state.detect_batch_done = self._batch_done
        state.detect_timeline_events = summary["event_count"]
        self._save_state(state)

        proc = self._detect_procs.pop(job_id, None)
        if proc is not None:
            self._terminate_process(proc)

        logger.warning("Recovered completed detection job %s from manifest", job_id)
        self._start_next_queued_or_finish(success=True)

    def _load_state(self) -> PipelineState:
        if not self._state_path.exists():
            return PipelineState(updated_at=_now_iso())
        raw = self._state_path.read_text(encoding="utf-8")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data, _ = json.JSONDecoder().raw_decode(raw.strip())
            logger.warning("Recovered corrupted pipeline state from %s", self._state_path)
            allowed = {item.name for item in fields(PipelineState)}
            filtered = {key: value for key, value in data.items() if key in allowed}
            recovered = PipelineState(**filtered)
            self._save_state(recovered)
            return recovered
        allowed = {item.name for item in fields(PipelineState)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        if "updated_at" not in filtered:
            filtered["updated_at"] = _now_iso()
        return PipelineState(**filtered)

    def _save_state(self, state: PipelineState) -> None:
        state.updated_at = _now_iso()
        payload = json.dumps(state.to_dict(), indent=2, ensure_ascii=False)
        temp_path = self._state_path.with_name(f"{self._state_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(payload, encoding="utf-8")
            temp_path.replace(self._state_path)
        finally:
            temp_path.unlink(missing_ok=True)

    def get_state(self) -> dict[str, Any]:
        self._recover_stale_cancel()
        self._recover_stale_train()
        self._recover_completed_detect_worker()
        self._recover_crashed_detect_worker()
        self._recover_stale_detect()
        state = self._load_state()
        if state.detect_status in {"cancelled", "completed", "error", "idle"} and (
            state.detect_job_id or state.detect_video_name or state.detect_batch_index
        ):
            state.detect_job_id = None
            state.detect_video_name = None
            state.detect_batch_index = 0
            self._save_state(state)
        payload = state.to_dict()
        meta_path = self.dataset_root / "import_meta.json"
        if meta_path.exists():
            payload["dataset_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["detect_timeline"] = timeline_summary(self._timeline_path)
        payload["models"] = [record.to_dict() for record in self.model_library.list_models()]
        if state.active_model_id and not self.model_library.exists(state.active_model_id):
            payload["active_model_id"] = None
            payload["active_model_name"] = None
        return payload

    def get_timeline(self, *, offset: int = 0, limit: int = 60) -> dict[str, Any]:
        return list_timeline(self._timeline_path, offset=offset, limit=limit)

    def get_timeline_segment(self, segment_id: str) -> dict[str, Any]:
        return get_segment_frames(self._timeline_path, segment_id)

    def reset_timeline(self) -> dict[str, Any]:
        reset_timeline(self._timeline_path)
        state = self._load_state()
        state.detect_timeline_events = 0
        self._save_state(state)
        return timeline_summary(self._timeline_path)

    def compact_timeline(self, *, max_gap_sec: float = 10.0) -> dict[str, Any]:
        result = compact_timeline_segments(self._timeline_path, max_gap_sec=max_gap_sec)
        state = self._load_state()
        state.detect_timeline_events = result["event_count"]
        self._save_state(state)
        return result

    def _detect_jobs_dir(self) -> Path:
        return self.root / "detect_jobs"

    def _saved_result_dir(self, result_id: str) -> Path:
        if not SAVE_ID_RE.match(result_id) or result_id in {".", ".."}:
            raise ValueError("Invalid saved result id")
        return self._saved_results_dir / result_id

    def _timeline_job_ids(self, timeline: dict[str, Any]) -> list[str]:
        job_ids: set[str] = set()
        for video in timeline.get("videos", []) or []:
            if video.get("job_id"):
                job_ids.add(str(video["job_id"]))
        for event in timeline.get("events", []) or []:
            if event.get("job_id"):
                job_ids.add(str(event["job_id"]))
        return sorted(job_ids)

    def list_saved_results(self) -> list[dict[str, Any]]:
        if not self._saved_results_dir.exists():
            return []
        results: list[dict[str, Any]] = []
        for directory in sorted(self._saved_results_dir.iterdir(), reverse=True):
            meta_path = directory / "metadata.json"
            if not meta_path.exists():
                continue
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            metadata["id"] = directory.name
            results.append(metadata)
        return results

    def save_current_results(self, name: str) -> dict[str, Any]:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("저장 이름을 입력하세요.")
        with self._lock:
            state = self._load_state()
            if _detect_is_active(state, threads=self._detect_threads):
                raise RuntimeError("탐지가 진행 중입니다. 완료 후 저장하세요.")
            if not self._timeline_path.exists():
                raise FileNotFoundError("저장할 탐지 타임라인이 없습니다.")

            timeline = json.loads(self._timeline_path.read_text(encoding="utf-8"))
            summary = timeline_summary(self._timeline_path)
            result_id = _safe_saved_result_id(clean_name)
            result_dir = self._saved_result_dir(result_id)
            if result_dir.exists():
                result_id = f"{result_id}-{uuid.uuid4().hex[:6]}"
                result_dir = self._saved_result_dir(result_id)
            jobs_dir = result_dir / "detect_jobs"
            result_dir.mkdir(parents=True, exist_ok=False)
            jobs_dir.mkdir(parents=True, exist_ok=True)

            job_ids = self._timeline_job_ids(timeline)
            copied_job_ids: list[str] = []
            try:
                shutil.copy2(self._timeline_path, result_dir / "detect_timeline.json")
                for job_id in job_ids:
                    src = self._job_dir(job_id)
                    if not src.exists():
                        continue
                    shutil.copytree(src, jobs_dir / job_id, copy_function=_copy_or_link)
                    copied_job_ids.append(job_id)
                metadata = {
                    "id": result_id,
                    "name": clean_name,
                    "saved_at": _now_iso(),
                    "segment_count": summary["segment_count"],
                    "event_count": summary["event_count"],
                    "video_count": summary["video_count"],
                    "job_count": len(copied_job_ids),
                    "job_ids": copied_job_ids,
                }
                (result_dir / "metadata.json").write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                shutil.rmtree(result_dir, ignore_errors=True)
                raise
            return metadata

    def load_saved_results(self, result_id: str) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if _detect_is_active(state, threads=self._detect_threads):
                raise RuntimeError("탐지가 진행 중입니다. 완료 후 불러오세요.")

            result_dir = self._saved_result_dir(result_id)
            timeline_src = result_dir / "detect_timeline.json"
            jobs_src = result_dir / "detect_jobs"
            meta_path = result_dir / "metadata.json"
            if not timeline_src.exists() or not meta_path.exists():
                raise FileNotFoundError("저장된 탐지 결과를 찾지 못했습니다.")

            backup_dir = self._restore_backups_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
            backup_dir.mkdir(parents=True, exist_ok=True)
            current_jobs = self._detect_jobs_dir()
            if self._timeline_path.exists():
                shutil.move(str(self._timeline_path), str(backup_dir / "detect_timeline.json"))
            if current_jobs.exists():
                shutil.move(str(current_jobs), str(backup_dir / "detect_jobs"))

            try:
                shutil.copy2(timeline_src, self._timeline_path)
                if jobs_src.exists():
                    shutil.copytree(jobs_src, current_jobs, copy_function=_copy_or_link)
                else:
                    current_jobs.mkdir(parents=True, exist_ok=True)
                summary = timeline_summary(self._timeline_path)
                state.detect_timeline_events = summary["event_count"]
                state.detect_status = "completed" if summary["segment_count"] else "idle"
                state.detect_error = None
                state.detect_job_id = None
                state.detect_video_name = None
                state.detect_batch_index = 0
                state.detect_queue_pending = 0
                self._clear_detect_overlay(state)
                self._save_state(state)
            except Exception:
                if self._timeline_path.exists():
                    self._timeline_path.unlink()
                if current_jobs.exists():
                    shutil.rmtree(current_jobs)
                if (backup_dir / "detect_timeline.json").exists():
                    shutil.move(str(backup_dir / "detect_timeline.json"), str(self._timeline_path))
                if (backup_dir / "detect_jobs").exists():
                    shutil.move(str(backup_dir / "detect_jobs"), str(current_jobs))
                raise

            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            metadata["id"] = result_dir.name
            metadata["loaded"] = True
            metadata["backup_id"] = backup_dir.name
            metadata["segment_count"] = summary["segment_count"]
            metadata["event_count"] = summary["event_count"]
            metadata["video_count"] = summary["video_count"]
            return metadata

    def import_cvat(
        self,
        annotations_path: Path,
        video_path: Path | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if state.import_status == "running":
                raise RuntimeError("Import already running")
            state.import_status = "running"
            state.import_error = None
            self._save_state(state)

        try:
            result = import_cvat(
                annotations_path,
                self.dataset_root,
                video_path=video_path,
                config_dir=self.config_dir,
            )
            state = self._load_state()
            state.import_status = "completed"
            state.import_result = result.to_dict()
            self._save_state(state)
            return result.to_dict()
        except Exception as exc:
            logger.exception("CVAT import failed")
            state = self._load_state()
            state.import_status = "error"
            state.import_error = str(exc)
            self._save_state(state)
            raise

    def _train_thread_alive(self) -> bool:
        return self._train_thread is not None and self._train_thread.is_alive()

    def start_training(
        self,
        *,
        epochs: int = 50,
        batch: int = 4,
        imgsz: int = 416,
        device: str | int = 0,
    ) -> dict[str, Any]:
        state = self._load_state()
        if state.train_status == "running":
            if self._train_thread_alive():
                raise RuntimeError("Training already running")
            logger.warning("Recovering stale train_status=running (thread not alive)")
            state.train_status = "idle"
            state.train_progress = None
            self._save_state(state)
        if not (self.config_dir / "dataset.yaml").exists():
            raise FileNotFoundError("dataset.yaml not found. Import CVAT zip first.")

        self._clear_train_cancel()
        state.train_status = "running"
        state.train_progress = "starting"
        state.train_epoch = 0
        state.train_epochs = epochs
        state.train_batch = 0
        state.train_batches = 0
        state.train_progress_pct = 0.0
        state.train_error = None
        self._save_state(state)

        thread = threading.Thread(
            target=self._run_training,
            args=(epochs, batch, imgsz, device),
            daemon=True,
        )
        self._train_thread = thread
        thread.start()
        return {"started": True}

    def _update_train_progress(
        self,
        epoch: int,
        total_epochs: int,
        batch_index: int = 0,
        total_batches: int = 0,
    ) -> None:
        state = self._load_state()
        state.train_epoch = epoch
        state.train_epochs = total_epochs
        state.train_batch = batch_index
        state.train_batches = total_batches
        if total_batches:
            epoch_base = max(epoch - 1, 0)
            batch_fraction = batch_index / max(total_batches, 1)
            progress = (epoch_base + batch_fraction) / max(total_epochs, 1)
        else:
            progress = epoch / max(total_epochs, 1)
        state.train_progress_pct = min(progress, 0.999)
        if self._train_cancel.is_set() or state.train_progress == "cancelling":
            state.train_progress = "cancelling"
        else:
            suffix = f" · batch {batch_index}/{total_batches}" if total_batches else ""
            state.train_progress = f"epoch {epoch}/{total_epochs}{suffix}"
        self._save_state(state)

    def _run_training(self, epochs: int, batch: int, imgsz: int, device: str | int) -> None:
        try:
            state = self._load_state()
            state.train_progress = "training"
            state.train_epochs = epochs
            state.train_batch = 0
            state.train_batches = 0
            self._save_state(state)

            weights = run_training(
                dataset_yaml=self.config_dir / "dataset.yaml",
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                task_type=load_task_type(self.dataset_root),
                on_epoch_end=self._update_train_progress,
                on_batch_end=self._update_train_progress,
                should_cancel=self._train_cancel.is_set,
            )
            state = self._load_state()
            record = self._register_trained_model(weights, epochs)
            state.train_status = "completed"
            if record is not None:
                state.train_weights = record.weights_path
                state.active_model_id = record.id
                state.active_model_name = record.name
            else:
                state.train_weights = str(weights.resolve())
            state.train_progress = "done"
            state.train_epoch = epochs
            state.train_progress_pct = 1.0
            self._save_state(state)
        except TrainingCancelled:
            logger.info("Training cancelled by user")
            state = self._load_state()
            state.train_status = "cancelled"
            state.train_progress = "cancelled"
            state.train_error = "사용자가 중지함"
            self._save_state(state)
        except Exception as exc:
            logger.exception("Training failed")
            state = self._load_state()
            if self._train_cancel.is_set():
                state.train_status = "cancelled"
                state.train_progress = "cancelled"
                state.train_error = "사용자가 중지함"
            else:
                state.train_status = "error"
                state.train_error = str(exc)
            self._save_state(state)
        finally:
            self._clear_train_cancel()

    def _job_dir(self, job_id: str) -> Path:
        return self.root / "detect_jobs" / job_id

    def start_detection(
        self,
        video_path: Path,
        video_name: str,
        *,
        model_path: Path | None = None,
        model_ids: list[str] | None = None,
        frame_stride: int = 5,
        confidence: float = 0.6,
        imgsz: int = 416,
        use_sam: bool = False,
        device: str | int = 0,
    ) -> DetectJob | dict[str, Any]:
        return self.start_detection_batch(
            [{"video_path": video_path, "video_name": video_name}],
            model_path=model_path,
            model_ids=model_ids,
            frame_stride=frame_stride,
            confidence=confidence,
            imgsz=imgsz,
            use_sam=use_sam,
            device=device,
        )

    def _prepare_batch_videos(self, videos: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Stage local uploads; keep remote Lake URLs for lazy download."""
        staging_root = self.root / "_detect_staging" / uuid.uuid4().hex[:12]
        staging_root.mkdir(parents=True, exist_ok=True)
        prepared: list[dict[str, Any]] = []
        for index, video in enumerate(videos):
            video_name = video["video_name"]
            if video.get("remote_url"):
                prepared.append({"video_name": video_name, "remote_url": video["remote_url"]})
                continue
            video_path: Path = video["video_path"]
            staged_path = staging_root / f"{index:03d}_{Path(video_name).name}"
            shutil.copy2(video_path, staged_path)
            prepared.append({"video_path": staged_path, "video_name": video_name})
        return prepared

    def start_detection_batch(
        self,
        videos: list[dict[str, Any]],
        *,
        model_path: Path | None = None,
        model_ids: list[str] | None = None,
        frame_stride: int = 5,
        confidence: float = 0.6,
        imgsz: int = 416,
        use_sam: bool = False,
        device: str | int = 0,
    ) -> dict[str, Any]:
        if not videos:
            raise ValueError("No videos provided")

        state = self._load_state()
        model_specs = self._detection_model_specs(state, model_path=model_path, model_ids=model_ids)

        self._clear_detection_cancel()
        self._recover_stale_detect()
        prepared_videos = self._prepare_batch_videos(videos)
        queued_jobs: list[dict[str, Any]] = []
        first_item: dict[str, Any] | None = None
        with self._lock:
            state = self._load_state()
            running = state.detect_status == "running" and self._detect_session_alive()
            batch_index_base = self._batch_total if running else 0

            if not running:
                self._batch_total = len(prepared_videos)
                self._batch_done = 0
                self._batch_failed = 0
                state.detect_batch_total = self._batch_total
                state.detect_batch_done = 0
                state.detect_batch_index = 0
                state.detect_status = "running"
                state.detect_error = None
                state.detect_processed_frames = 0
                state.detect_total_frames = 0
                state.detect_progress_pct = 0.0
            else:
                self._batch_total += len(prepared_videos)
                state.detect_batch_total = self._batch_total

            for index, prepared in enumerate(prepared_videos):
                item = {
                    "video_name": prepared["video_name"],
                    "model_path": Path(model_specs[0]["path"]),
                    "model_specs": model_specs,
                    "frame_stride": frame_stride,
                    "confidence": confidence,
                    "imgsz": imgsz,
                    "use_sam": use_sam,
                    "device": device,
                    "batch_index": batch_index_base + index + 1,
                }
                if prepared.get("remote_url"):
                    item["remote_url"] = prepared["remote_url"]
                else:
                    item["video_path"] = prepared["video_path"]
                if not running and index == 0:
                    first_item = item
                    queued_jobs.append(
                        {"video_name": prepared["video_name"], "status": "starting"}
                    )
                    running = True
                else:
                    self._detect_queue.append(item)
                    queued_jobs.append({"video_name": prepared["video_name"], "status": "queued"})

            state.detect_queue_pending = len(self._detect_queue)
            state.detect_batch_total = self._batch_total
            state.detect_batch_done = self._batch_done
            self._save_state(state)

        if first_item is not None:
            try:
                job = self._launch_detection(first_item)
                queued_jobs[0] = {
                    "job_id": job.job_id,
                    "video_name": job.video_name,
                    "status": "running",
                }
            except DetectionCancelled:
                logger.info("First detection job cancelled before start")
                with self._lock:
                    self._detect_queue.clear()
                    state = self._load_state()
                    state.detect_status = "cancelled"
                    state.detect_error = "사용자가 중지함"
                    state.detect_queue_pending = 0
                    state.detect_job_id = None
                    state.detect_video_name = None
                    state.detect_batch_index = 0
                    self._clear_detect_overlay(state)
                    self._save_state(state)
                queued_jobs[0] = {
                    "video_name": first_item["video_name"],
                    "status": "cancelled",
                }
            except Exception as exc:
                # First video failed to start; skip it and let the rest of the
                # batch continue instead of aborting everything.
                logger.exception("Failed to launch first detection job; skipping")
                self._batch_done += 1
                self._batch_failed += 1
                state = self._load_state()
                state.detect_batch_done = self._batch_done
                state.detect_error = str(exc)
                self._save_state(state)
                queued_jobs[0] = {
                    "video_name": first_item["video_name"],
                    "status": "error",
                    "error": str(exc),
                }
                self._start_next_queued_or_finish(success=False)

        return {
            "batch_size": len(prepared_videos),
            "batch_total": self._batch_total,
            "batch_done": self._batch_done,
            "queue_pending": len(self._detect_queue),
            "jobs": queued_jobs,
            "job": queued_jobs[0] if queued_jobs and queued_jobs[0].get("job_id") else None,
        }

    def start_stream_detection(
        self,
        stream_url: str,
        *,
        model_path: Path | None = None,
        model_ids: list[str] | None = None,
        frame_stride: int = 5,
        confidence: float = 0.6,
        imgsz: int = 416,
        use_sam: bool = False,
        device: str | int = 0,
    ) -> dict[str, Any]:
        stream_url = stream_url.strip()
        if not stream_url:
            raise ValueError("스트림 주소를 입력하세요.")

        state = self._load_state()
        model_specs = self._detection_model_specs(state, model_path=model_path, model_ids=model_ids)
        model_path = Path(model_specs[0]["path"])

        self._recover_stale_detect()
        state = self._load_state()
        if _detect_is_active(state, threads=self._detect_threads):
            raise RuntimeError("탐지가 진행 중입니다. 먼저 중지하세요.")

        self._clear_detection_cancel()
        self._detect_queue.clear()
        self._batch_total = 1
        self._batch_done = 0
        self._batch_failed = 0

        now = datetime.now()
        video_name = f"live_stream_{now:%y%m%d_%H%M%S}.mp4"
        job_id = uuid.uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "stop.txt").unlink(missing_ok=True)

        job = DetectJob(
            job_id=job_id,
            video_name=video_name,
            created_at=_now_iso(),
            status="running",
        )
        self._save_job(job)
        meta = {
            "video_name": video_name,
            "video_start": parse_video_start_time(video_name).isoformat(),
            "stream_url": stream_url,
            "models": model_specs,
        }
        (directory / "video_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state.detect_status = "running"
        state.detect_job_id = job_id
        state.detect_video_name = video_name
        state.detect_batch_index = 1
        state.detect_batch_total = 1
        state.detect_batch_done = 0
        state.detect_queue_pending = 0
        state.detect_processed_frames = 0
        state.detect_total_frames = 0
        state.detect_frames_with_objects = 0
        state.detect_progress_pct = 0.0
        state.detect_overlay_job_id = None
        state.detect_overlay_preview_path = None
        state.detect_overlay_frame_index = None
        state.detect_overlay_timestamp_sec = None
        state.detect_overlay_width = 0
        state.detect_overlay_height = 0
        state.detect_overlay_detections = None
        state.detect_overlay_updated_at = None
        state.detect_error = None
        self._save_state(state)

        thread = threading.Thread(
            target=self._run_stream_detection,
            args=(job_id, stream_url, model_path, model_specs, frame_stride, confidence, imgsz, use_sam, device, video_name),
            daemon=True,
        )
        self._detect_threads[job_id] = thread
        thread.start()
        return {
            "batch_size": 1,
            "batch_total": 1,
            "batch_done": 0,
            "queue_pending": 0,
            "jobs": [{"job_id": job_id, "video_name": video_name, "status": "running"}],
            "job": {"job_id": job_id, "video_name": video_name, "status": "running"},
        }

    def _launch_detection(self, item: dict[str, Any]) -> DetectJob:
        video_name: str = item["video_name"]
        model_path: Path = item["model_path"]
        model_specs: list[dict[str, Any]] = list(item.get("model_specs") or [{"path": str(model_path)}])
        frame_stride: int = item["frame_stride"]
        confidence: float = item["confidence"]
        imgsz: int = int(item.get("imgsz") or 416)
        use_sam: bool = bool(item.get("use_sam", False))
        device: str | int = item["device"]

        job_id = uuid.uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target_video = directory / "video.mp4"
        if self._detection_cancelled():
            raise DetectionCancelled("Detection cancelled by user")

        state = self._load_state()
        state.detect_status = "running"
        state.detect_video_name = video_name
        state.detect_batch_index = int(item.get("batch_index") or 0)
        state.detect_processed_frames = 0
        state.detect_total_frames = 0
        state.detect_progress_pct = 0.0
        state.detect_error = None
        self._save_state(state)

        if item.get("remote_url"):
            try:
                download_video(
                    item["remote_url"],
                    target_video,
                    should_cancel=self._detection_cancelled,
                )
            except DetectionCancelled:
                raise
        else:
            video_path: Path = item["video_path"]
            target_video.write_bytes(video_path.read_bytes())
            if "_detect_staging" in video_path.as_posix():
                video_path.unlink(missing_ok=True)
                # Remove the per-batch staging folder once it has been drained
                # so it doesn't accumulate over time.
                staging_dir = video_path.parent
                try:
                    if staging_dir.name and not any(staging_dir.iterdir()):
                        staging_dir.rmdir()
                except OSError:
                    pass

        video_start = parse_video_start_time(video_name)
        job = DetectJob(
            job_id=job_id,
            video_name=video_name,
            created_at=_now_iso(),
            status="running",
        )
        self._save_job(job)

        meta = {
            "video_name": video_name,
            "video_start": video_start.isoformat() if video_start else None,
            "models": model_specs,
        }
        (directory / "video_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state = self._load_state()
        if self._detection_cancelled() or state.detect_status == "cancelling":
            raise DetectionCancelled("Detection cancelled by user")
        state.detect_status = "running"
        state.detect_job_id = job_id
        state.detect_video_name = video_name
        state.detect_batch_index = int(item.get("batch_index") or 0)
        state.detect_progress_pct = 0.0
        state.detect_processed_frames = 0
        state.detect_total_frames = 0
        state.detect_frames_with_objects = 0
        state.detect_overlay_job_id = None
        state.detect_overlay_preview_path = None
        state.detect_overlay_frame_index = None
        state.detect_overlay_timestamp_sec = None
        state.detect_overlay_width = 0
        state.detect_overlay_height = 0
        state.detect_overlay_detections = None
        state.detect_overlay_updated_at = None
        state.detect_error = None
        state.detect_queue_pending = len(self._detect_queue)
        state.detect_batch_total = self._batch_total
        state.detect_batch_done = self._batch_done
        planned = self._planned_frame_count(target_video, frame_stride)
        job.total_frames = planned
        self._save_job(job)
        state.detect_total_frames = planned
        state.detect_processed_frames = 0
        state.detect_progress_pct = 0.0
        self._save_state(state)

        thread = threading.Thread(
            target=self._run_detection,
            args=(job_id, target_video, model_path, model_specs, frame_stride, confidence, imgsz, use_sam, device, video_name),
            daemon=True,
        )
        self._detect_threads[job_id] = thread
        thread.start()
        return job

    def _update_detect_progress(
        self,
        job_id: str,
        processed: int,
        total: int,
        with_detections: int,
    ) -> None:
        job = self.get_detect_job(job_id)
        job.processed_frames = processed
        job.total_frames = total
        job.frames_with_detections = with_detections
        job.progress = processed / total if total > 0 else 0.0
        self._save_job(job)

        state = self._load_state()
        if state.detect_job_id == job_id:
            state.detect_progress_pct = job.progress
            state.detect_processed_frames = processed
            state.detect_total_frames = total
            state.detect_frames_with_objects = with_detections
            self._save_state(state)

    @staticmethod
    def _planned_frame_count(video_path: Path, frame_stride: int) -> int:
        import cv2

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return 1
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        stride = max(frame_stride, 1)
        if total_frames <= 0:
            return 1
        return max(len(range(0, total_frames, stride)), 1)

    def _run_detection_subprocess(
        self,
        job_id: str,
        video_path: Path,
        video_name: str,
        model_path: Path,
        model_specs: list[dict[str, Any]],
        frame_stride: int,
        confidence: float,
        imgsz: int,
        use_sam: bool,
        device: str | int,
    ) -> dict[str, Any]:
        """Run YOLO inference in a separate process so a CUDA crash can't wedge
        the server. Returns the detection manifest the worker wrote to disk."""
        job_dir = self._job_dir(job_id)
        progress_file = job_dir / "progress.json"
        events_file = job_dir / "events.jsonl"
        error_file = job_dir / "worker_error.txt"
        log_file = job_dir / "worker.log"
        for stale in (progress_file, events_file, error_file):
            stale.unlink(missing_ok=True)
        model_specs_file = job_dir / "model_specs.json"
        model_specs_file.write_text(json.dumps(model_specs, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "brailer_monitor.web.detect_worker",
            "--video",
            str(video_path),
            "--model",
            str(model_path),
            "--model-specs-file",
            str(model_specs_file),
            "--output",
            str(job_dir),
            "--frame-stride",
            str(frame_stride),
            "--confidence",
            str(confidence),
            "--imgsz",
            str(imgsz),
            "--sam",
            "yes" if use_sam else "no",
            "--device",
            str(device),
            "--progress-file",
            str(progress_file),
            "--events-file",
            str(events_file),
        ]

        last_progress: dict[str, Any] | None = None
        events_offset = 0
        event_manifest = {
            "frame_stride": frame_stride,
            "total_frames": self._planned_frame_count(video_path, frame_stride),
        }

        def _drain_progress() -> None:
            nonlocal last_progress
            if not progress_file.exists():
                return
            try:
                prog = json.loads(progress_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return
            if prog and prog != last_progress:
                last_progress = prog
                self._update_detect_progress(
                    job_id,
                    int(prog.get("processed", 0)),
                    int(prog.get("total", 0)) or 1,
                    int(prog.get("with", 0)),
                )

        def _drain_events() -> None:
            nonlocal events_offset
            if not events_file.exists():
                return
            try:
                with events_file.open("rb") as handle:
                    handle.seek(events_offset)
                    data = handle.read()
            except OSError:
                return
            if not data:
                return
            newline_at = data.rfind(b"\n")
            if newline_at < 0:
                return
            chunk = data[: newline_at + 1]
            events_offset += len(chunk)
            changed = False
            for raw_line in chunk.splitlines():
                if not raw_line.strip():
                    continue
                try:
                    frame = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                changed = merge_frame_detection(
                    self._timeline_path,
                    job_id=job_id,
                    video_name=video_name,
                    frame=frame,
                    manifest=event_manifest,
                ) > 0 or changed
            if changed:
                summary = timeline_summary(self._timeline_path)
                state = self._load_state()
                if state.detect_job_id == job_id:
                    state.detect_timeline_events = summary["event_count"]
                    state.detect_overlay_job_id = job_id
                    state.detect_overlay_preview_path = frame.get("preview_path")
                    state.detect_overlay_frame_index = frame.get("frame_index")
                    state.detect_overlay_timestamp_sec = frame.get("timestamp_sec")
                    state.detect_overlay_width = int(frame.get("width") or 0)
                    state.detect_overlay_height = int(frame.get("height") or 0)
                    state.detect_overlay_detections = frame.get("detections") or []
                    state.detect_overlay_updated_at = _now_iso()
                    self._save_state(state)

        with open(log_file, "w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.config_dir.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        self._detect_procs[job_id] = proc
        try:
            while proc.poll() is None:
                if self._detection_cancelled():
                    self._terminate_process(proc)
                    raise DetectionCancelled("Detection cancelled by user")
                _drain_progress()
                _drain_events()
                time.sleep(0.5)

            _drain_progress()
            _drain_events()
            if self._detection_cancelled():
                raise DetectionCancelled("Detection cancelled by user")

            if proc.returncode != 0:
                detail = ""
                if error_file.exists():
                    detail = error_file.read_text(encoding="utf-8").strip()
                if not detail and log_file.exists():
                    detail = log_file.read_text(encoding="utf-8").strip()[-2000:]
                raise RuntimeError(
                    detail or f"탐지 작업이 코드 {proc.returncode}로 종료되었습니다."
                )

            manifest_path = job_dir / "detections.json"
            if not manifest_path.exists():
                raise RuntimeError("탐지 결과 파일(detections.json)이 생성되지 않았습니다.")
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        finally:
            self._detect_procs.pop(job_id, None)

    def _run_stream_detection_subprocess(
        self,
        job_id: str,
        stream_url: str,
        video_name: str,
        model_path: Path,
        model_specs: list[dict[str, Any]],
        frame_stride: int,
        confidence: float,
        imgsz: int,
        use_sam: bool,
        device: str | int,
    ) -> dict[str, Any]:
        job_dir = self._job_dir(job_id)
        progress_file = job_dir / "progress.json"
        events_file = job_dir / "events.jsonl"
        stop_file = job_dir / "stop.txt"
        error_file = job_dir / "worker_error.txt"
        log_file = job_dir / "worker.log"
        for stale in (progress_file, events_file, error_file, stop_file):
            stale.unlink(missing_ok=True)
        model_specs_file = job_dir / "model_specs.json"
        model_specs_file.write_text(json.dumps(model_specs, ensure_ascii=False, indent=2), encoding="utf-8")

        cmd = [
            sys.executable,
            "-m",
            "brailer_monitor.web.detect_worker",
            "--stream-url",
            stream_url,
            "--model",
            str(model_path),
            "--model-specs-file",
            str(model_specs_file),
            "--output",
            str(job_dir),
            "--frame-stride",
            str(frame_stride),
            "--confidence",
            str(confidence),
            "--imgsz",
            str(imgsz),
            "--sam",
            "yes" if use_sam else "no",
            "--device",
            str(device),
            "--progress-file",
            str(progress_file),
            "--events-file",
            str(events_file),
            "--stop-file",
            str(stop_file),
        ]

        last_progress: dict[str, Any] | None = None
        events_offset = 0
        event_manifest = {"frame_stride": frame_stride, "total_frames": 0}

        def _drain_progress() -> None:
            nonlocal last_progress
            if not progress_file.exists():
                return
            try:
                prog = json.loads(progress_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return
            if prog and prog != last_progress:
                last_progress = prog
                self._update_detect_progress(
                    job_id,
                    int(prog.get("processed", 0)),
                    int(prog.get("total", 0)),
                    int(prog.get("with", 0)),
                )

        def _drain_events() -> None:
            nonlocal events_offset
            if not events_file.exists():
                return
            try:
                with events_file.open("rb") as handle:
                    handle.seek(events_offset)
                    data = handle.read()
            except OSError:
                return
            if not data:
                return
            newline_at = data.rfind(b"\n")
            if newline_at < 0:
                return
            chunk = data[: newline_at + 1]
            events_offset += len(chunk)
            changed = False
            for raw_line in chunk.splitlines():
                if not raw_line.strip():
                    continue
                try:
                    frame = json.loads(raw_line.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                changed = merge_frame_detection(
                    self._timeline_path,
                    job_id=job_id,
                    video_name=video_name,
                    frame=frame,
                    manifest=event_manifest,
                ) > 0 or changed
            if changed:
                summary = timeline_summary(self._timeline_path)
                state = self._load_state()
                if state.detect_job_id == job_id:
                    state.detect_timeline_events = summary["event_count"]
                    state.detect_overlay_job_id = job_id
                    state.detect_overlay_preview_path = frame.get("preview_path")
                    state.detect_overlay_frame_index = frame.get("frame_index")
                    state.detect_overlay_timestamp_sec = frame.get("timestamp_sec")
                    state.detect_overlay_width = int(frame.get("width") or 0)
                    state.detect_overlay_height = int(frame.get("height") or 0)
                    state.detect_overlay_detections = frame.get("detections") or []
                    state.detect_overlay_updated_at = _now_iso()
                    self._save_state(state)

        with open(log_file, "w", encoding="utf-8") as log_handle:
            proc = subprocess.Popen(
                cmd,
                cwd=str(self.config_dir.parent),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        self._detect_procs[job_id] = proc
        try:
            stop_requested_at: float | None = None
            while proc.poll() is None:
                if self._detection_cancelled():
                    stop_file.write_text("stop", encoding="utf-8")
                    if stop_requested_at is None:
                        stop_requested_at = time.time()
                    elif time.time() - stop_requested_at > 20:
                        self._terminate_process(proc)
                        raise DetectionCancelled("Detection cancelled by user")
                _drain_progress()
                _drain_events()
                time.sleep(0.5)

            _drain_progress()
            _drain_events()
            if proc.returncode != 0:
                detail = ""
                if error_file.exists():
                    detail = error_file.read_text(encoding="utf-8").strip()
                if not detail and log_file.exists():
                    detail = log_file.read_text(encoding="utf-8").strip()[-2000:]
                raise RuntimeError(
                    detail or f"스트림 탐지 작업이 코드 {proc.returncode}로 종료되었습니다."
                )

            manifest_path = job_dir / "detections.json"
            if not manifest_path.exists():
                raise RuntimeError("스트림 탐지 결과 파일(detections.json)이 생성되지 않았습니다.")
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        finally:
            self._detect_procs.pop(job_id, None)

    @staticmethod
    def _terminate_process(proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Detection worker pid %s did not exit after kill", proc.pid)

    def _run_detection(
        self,
        job_id: str,
        video_path: Path,
        model_path: Path,
        model_specs: list[dict[str, Any]],
        frame_stride: int,
        confidence: float,
        imgsz: int,
        use_sam: bool,
        device: str | int,
        video_name: str,
    ) -> None:
        job = self.get_detect_job(job_id)
        try:
            if self._detection_cancelled():
                raise DetectionCancelled("Detection cancelled by user")

            planned = self._planned_frame_count(video_path, frame_stride)
            self._update_detect_progress(job_id, 0, planned, 0)

            try:
                manifest = self._run_detection_subprocess(
                    job_id,
                    video_path,
                    video_name,
                    model_path,
                    model_specs,
                    frame_stride,
                    confidence,
                    imgsz,
                    use_sam,
                    device,
                )
            except RuntimeError as exc:
                if _is_cpu_device(device) or not _is_cuda_oom_error(str(exc)):
                    raise
                logger.warning(
                    "CUDA out of memory for job %s; retrying detection on CPU",
                    job_id,
                )
                state = self._load_state()
                if state.detect_job_id == job_id:
                    state.detect_error = "GPU 메모리 부족으로 CPU에서 재시도 중입니다."
                    state.detect_processed_frames = 0
                    state.detect_progress_pct = 0.0
                    self._save_state(state)
                self._update_detect_progress(job_id, 0, planned, 0)
                manifest = self._run_detection_subprocess(
                    job_id,
                    video_path,
                    video_name,
                    model_path,
                    model_specs,
                    frame_stride,
                    confidence,
                    imgsz,
                    use_sam,
                    "cpu",
                )
            if self._detection_cancelled():
                raise DetectionCancelled("Detection cancelled by user")

            manifest["video_name"] = video_name
            video_start = parse_video_start_time(video_name)
            if video_start:
                manifest["video_start"] = video_start.isoformat()
            manifest_path = self._job_dir(job_id) / "detections.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            added = merge_job_manifest(
                self._timeline_path,
                job_id=job_id,
                video_name=video_name,
                manifest=manifest,
                replace_job=True,
            )

            job = self.get_detect_job(job_id)
            job.status = "completed"
            job.progress = 1.0
            job.processed_frames = manifest["frames_processed"]
            job.total_frames = manifest["frames_processed"]
            job.frames_with_detections = manifest["frames_with_detections"]
            job.manifest_path = str(manifest_path.resolve())
            self._save_job(job)

            self._batch_done += 1
            summary = timeline_summary(self._timeline_path)
            state = self._load_state()
            state.detect_progress_pct = 1.0
            state.detect_processed_frames = job.processed_frames
            state.detect_total_frames = job.total_frames
            state.detect_frames_with_objects = job.frames_with_detections
            state.detect_batch_done = self._batch_done
            state.detect_timeline_events = summary["event_count"]
            self._save_state(state)

            self._start_next_queued_or_finish(success=True)
        except DetectionCancelled:
            logger.info("Detection cancelled for job %s", job_id)
            job = self.get_detect_job(job_id)
            job.status = "cancelled"
            job.error = "사용자가 중지함"
            self._save_job(job)
            self._finish_detection_cancelled()
        except Exception as exc:
            logger.exception("Detection failed for job %s", job_id)
            if self._detection_cancelled():
                job = self.get_detect_job(job_id)
                job.status = "cancelled"
                job.error = "사용자가 중지함"
                self._save_job(job)
                self._finish_detection_cancelled()
                return
            job = self.get_detect_job(job_id)
            job.status = "error"
            job.error = str(exc)
            self._save_job(job)
            # Skip the failed video and continue with the rest of the batch.
            # (A crashed worker subprocess can't corrupt the server, so the
            # next video starts fresh.)
            self._batch_done += 1
            self._batch_failed += 1
            state = self._load_state()
            state.detect_batch_done = self._batch_done
            state.detect_error = str(exc)
            self._save_state(state)
            self._start_next_queued_or_finish(success=False)
        finally:
            self._detect_threads.pop(job_id, None)
            self._detect_procs.pop(job_id, None)

    def _run_stream_detection(
        self,
        job_id: str,
        stream_url: str,
        model_path: Path,
        model_specs: list[dict[str, Any]],
        frame_stride: int,
        confidence: float,
        imgsz: int,
        use_sam: bool,
        device: str | int,
        video_name: str,
    ) -> None:
        try:
            manifest = self._run_stream_detection_subprocess(
                job_id,
                stream_url,
                video_name,
                model_path,
                model_specs,
                frame_stride,
                confidence,
                imgsz,
                use_sam,
                device,
            )
            manifest["video_name"] = video_name
            video_start = parse_video_start_time(video_name)
            if video_start:
                manifest["video_start"] = video_start.isoformat()
            manifest_path = self._job_dir(job_id) / "detections.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            merge_job_manifest(
                self._timeline_path,
                job_id=job_id,
                video_name=video_name,
                manifest=manifest,
                replace_job=True,
            )

            job = self.get_detect_job(job_id)
            job.status = "completed"
            job.progress = 1.0
            job.processed_frames = int(manifest.get("frames_processed", 0))
            job.total_frames = job.processed_frames
            job.frames_with_detections = int(manifest.get("frames_with_detections", 0))
            job.manifest_path = str(manifest_path.resolve())
            self._save_job(job)

            self._batch_done = 1
            summary = timeline_summary(self._timeline_path)
            state = self._load_state()
            state.detect_status = "completed"
            state.detect_job_id = job_id
            state.detect_progress_pct = 1.0
            state.detect_processed_frames = job.processed_frames
            state.detect_total_frames = job.total_frames
            state.detect_frames_with_objects = job.frames_with_detections
            state.detect_queue_pending = 0
            state.detect_batch_total = 1
            state.detect_batch_done = 1
            state.detect_batch_index = 0
            state.detect_video_name = None
            state.detect_timeline_events = summary["event_count"]
            state.detect_error = None
            self._clear_detect_overlay(state)
            self._save_state(state)
        except DetectionCancelled:
            logger.info("Stream detection cancelled for job %s", job_id)
            job = self.get_detect_job(job_id)
            job.status = "cancelled"
            job.error = "사용자가 중지함"
            self._save_job(job)
            self._finish_detection_cancelled()
        except Exception as exc:
            logger.exception("Stream detection failed for job %s", job_id)
            job = self.get_detect_job(job_id)
            job.status = "error"
            job.error = str(exc)
            self._save_job(job)
            self._batch_done = 1
            self._batch_failed = 1
            state = self._load_state()
            state.detect_status = "error"
            state.detect_error = str(exc)
            state.detect_batch_done = 1
            state.detect_queue_pending = 0
            state.detect_video_name = None
            state.detect_batch_index = 0
            self._clear_detect_overlay(state)
            self._save_state(state)
        finally:
            self._detect_threads.pop(job_id, None)
            self._detect_procs.pop(job_id, None)
            self._clear_detection_cancel()

    def _start_next_queued_or_finish(self, *, success: bool) -> None:
        next_item: dict[str, Any] | None = None
        with self._lock:
            if self._detection_cancelled():
                self._finish_detection_cancelled()
                return

            if self._detect_queue:
                next_item = self._detect_queue.pop(0)
                state = self._load_state()
                state.detect_status = "running"
                state.detect_job_id = None
                state.detect_video_name = next_item["video_name"]
                state.detect_batch_index = int(next_item.get("batch_index") or 0)
                state.detect_queue_pending = len(self._detect_queue)
                self._save_state(state)
            else:
                state = self._load_state()
                # Batch ran to the end. Mark "error" only if every video failed;
                # otherwise the batch is "completed" (possibly with some skips).
                if self._batch_failed and self._batch_failed >= self._batch_done:
                    state.detect_status = "error"
                    state.detect_error = f"{self._batch_failed}개 영상 모두 탐지에 실패했습니다."
                else:
                    state.detect_status = "completed"
                    state.detect_error = (
                        f"{self._batch_failed}개 영상 탐지 실패 · 나머지는 완료"
                        if self._batch_failed
                        else None
                    )
                state.detect_video_name = None
                state.detect_batch_index = 0
                state.detect_queue_pending = 0
                state.detect_batch_total = self._batch_total
                state.detect_batch_done = self._batch_done
                summary = timeline_summary(self._timeline_path)
                state.detect_timeline_events = summary["event_count"]
                self._clear_detect_overlay(state)
                self._save_state(state)
                return

        if next_item is None:
            return
        if self._detection_cancelled():
            self._finish_detection_cancelled()
            return
        try:
            self._launch_detection(next_item)
        except DetectionCancelled:
            self._finish_detection_cancelled()
        except Exception as exc:
            # The queued video failed to start (download/IO). Skip it and move
            # on so one bad video can't stall the whole batch.
            logger.exception("Failed to launch queued detection; skipping")
            self._batch_done += 1
            self._batch_failed += 1
            state = self._load_state()
            state.detect_batch_done = self._batch_done
            state.detect_error = str(exc)
            self._save_state(state)
            self._start_next_queued_or_finish(success=False)

    def _save_job(self, job: DetectJob) -> None:
        path = self._job_dir(job.job_id) / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def get_detect_job(self, job_id: str) -> DetectJob:
        path = self._job_dir(job_id) / "job.json"
        if not path.exists():
            raise FileNotFoundError(f"Detect job not found: {job_id}")
        data = json.loads(path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(DetectJob)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        return DetectJob(**filtered)

    def list_detect_jobs(self) -> list[DetectJob]:
        jobs_dir = self.root / "detect_jobs"
        if not jobs_dir.exists():
            return []
        jobs: list[DetectJob] = []
        for directory in sorted(jobs_dir.iterdir(), reverse=True):
            if (directory / "job.json").exists():
                try:
                    jobs.append(self.get_detect_job(directory.name))
                except Exception:
                    continue
        return jobs

    def get_detection_manifest(
        self,
        job_id: str,
        *,
        detections_only: bool = False,
        offset: int = 0,
        limit: int | None = None,
    ) -> dict[str, Any]:
        path = self._job_dir(job_id) / "detections.json"
        if not path.exists():
            raise FileNotFoundError("Detection results not ready")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        frames = manifest.get("frames", [])
        if detections_only:
            frames = [frame for frame in frames if frame.get("detections")]
        total_matching = len(frames)
        if limit is not None:
            frames = frames[offset : offset + limit]
        payload = {
            key: value
            for key, value in manifest.items()
            if key != "frames"
        }
        payload["frames"] = frames
        payload["total_matching"] = total_matching
        payload["offset"] = offset
        payload["limit"] = limit
        return payload

    def get_detection_video_path(self, job_id: str) -> Path:
        manifest = self.get_detection_manifest(job_id)
        video = manifest.get("video")
        if not video:
            raise FileNotFoundError("Detection video path not found")
        path = Path(str(video))
        if not path.exists():
            raise FileNotFoundError("Detection video not found")
        return path
