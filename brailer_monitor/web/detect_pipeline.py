"""CVAT import, YOLO training, and video detection job management."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..cvat_import import import_cvat
from ..train import load_task_type, run_training
from ..video_detect import detect_video

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PipelineState:
    import_status: str = "idle"
    import_result: dict[str, Any] | None = None
    import_error: str | None = None
    train_status: str = "idle"
    train_progress: str | None = None
    train_epoch: int = 0
    train_epochs: int = 0
    train_progress_pct: float = 0.0
    train_weights: str | None = None
    train_error: str | None = None
    detect_status: str = "idle"
    detect_job_id: str | None = None
    detect_progress_pct: float = 0.0
    detect_processed_frames: int = 0
    detect_total_frames: int = 0
    detect_frames_with_objects: int = 0
    detect_error: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
        self._state_path = self.root / "pipeline_state.json"
        self._lock = threading.Lock()
        self._train_thread: threading.Thread | None = None
        self._detect_threads: dict[str, threading.Thread] = {}

    def _load_state(self) -> PipelineState:
        if not self._state_path.exists():
            return PipelineState(updated_at=_now_iso())
        data = json.loads(self._state_path.read_text(encoding="utf-8"))
        allowed = {item.name for item in fields(PipelineState)}
        filtered = {key: value for key, value in data.items() if key in allowed}
        if "updated_at" not in filtered:
            filtered["updated_at"] = _now_iso()
        return PipelineState(**filtered)

    def _save_state(self, state: PipelineState) -> None:
        state.updated_at = _now_iso()
        self._state_path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")

    def get_state(self) -> dict[str, Any]:
        state = self._load_state()
        payload = state.to_dict()
        meta_path = self.dataset_root / "import_meta.json"
        if meta_path.exists():
            payload["dataset_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        return payload

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

    def start_training(
        self,
        *,
        epochs: int = 50,
        batch: int = 8,
        imgsz: int = 640,
        device: str | int = 0,
    ) -> dict[str, Any]:
        state = self._load_state()
        if state.train_status == "running":
            raise RuntimeError("Training already running")
        if not (self.config_dir / "dataset.yaml").exists():
            raise FileNotFoundError("dataset.yaml not found. Import CVAT zip first.")

        state.train_status = "running"
        state.train_progress = "starting"
        state.train_epoch = 0
        state.train_epochs = epochs
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

    def _update_train_progress(self, epoch: int, total_epochs: int) -> None:
        state = self._load_state()
        state.train_epoch = epoch
        state.train_epochs = total_epochs
        state.train_progress_pct = epoch / max(total_epochs, 1)
        state.train_progress = f"epoch {epoch}/{total_epochs}"
        self._save_state(state)

    def _run_training(self, epochs: int, batch: int, imgsz: int, device: str | int) -> None:
        try:
            state = self._load_state()
            state.train_progress = "training"
            state.train_epochs = epochs
            self._save_state(state)

            weights = run_training(
                dataset_yaml=self.config_dir / "dataset.yaml",
                epochs=epochs,
                batch=batch,
                imgsz=imgsz,
                device=device,
                task_type=load_task_type(self.dataset_root),
                on_epoch_end=self._update_train_progress,
            )
            state = self._load_state()
            state.train_status = "completed"
            state.train_weights = str(weights.resolve())
            state.train_progress = "done"
            state.train_epoch = epochs
            state.train_progress_pct = 1.0
            self._save_state(state)
        except Exception as exc:
            logger.exception("Training failed")
            state = self._load_state()
            state.train_status = "error"
            state.train_error = str(exc)
            self._save_state(state)

    def _job_dir(self, job_id: str) -> Path:
        return self.root / "detect_jobs" / job_id

    def start_detection(
        self,
        video_path: Path,
        video_name: str,
        *,
        model_path: Path | None = None,
        frame_stride: int = 5,
        confidence: float = 0.35,
        device: str | int = 0,
    ) -> DetectJob:
        state = self._load_state()
        if model_path is None:
            task = load_task_type(self.dataset_root)
            default = Path("models/brailer_detect.pt" if task == "detect" else "models/brailer_seg.pt")
            model_path = Path(state.train_weights) if state.train_weights else default
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

        job_id = uuid.uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target_video = directory / "video.mp4"
        target_video.write_bytes(video_path.read_bytes())

        job = DetectJob(
            job_id=job_id,
            video_name=video_name,
            created_at=_now_iso(),
            status="running",
        )
        self._save_job(job)

        state.detect_status = "running"
        state.detect_job_id = job_id
        state.detect_progress_pct = 0.0
        state.detect_processed_frames = 0
        state.detect_total_frames = 0
        state.detect_frames_with_objects = 0
        state.detect_error = None
        self._save_state(state)

        thread = threading.Thread(
            target=self._run_detection,
            args=(job_id, target_video, model_path, frame_stride, confidence, device),
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
        job.progress = processed / max(total, 1)
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

    def _run_detection(
        self,
        job_id: str,
        video_path: Path,
        model_path: Path,
        frame_stride: int,
        confidence: float,
        device: str | int,
    ) -> None:
        job = self.get_detect_job(job_id)
        try:
            planned = self._planned_frame_count(video_path, frame_stride)
            self._update_detect_progress(job_id, 0, planned, 0)

            def on_progress(processed: int, total: int, with_det: int) -> None:
                self._update_detect_progress(job_id, processed, total, with_det)

            manifest = detect_video(
                video_path,
                model_path,
                output_dir=self._job_dir(job_id),
                frame_stride=frame_stride,
                confidence=confidence,
                device=device,
                on_progress=on_progress,
            )
            job = self.get_detect_job(job_id)
            job.status = "completed"
            job.progress = 1.0
            job.processed_frames = manifest["frames_processed"]
            job.total_frames = manifest["frames_processed"]
            job.frames_with_detections = manifest["frames_with_detections"]
            job.manifest_path = str((self._job_dir(job_id) / "detections.json").resolve())
            self._save_job(job)

            state = self._load_state()
            state.detect_status = "completed"
            state.detect_progress_pct = 1.0
            state.detect_processed_frames = job.processed_frames
            state.detect_total_frames = job.total_frames
            state.detect_frames_with_objects = job.frames_with_detections
            self._save_state(state)
        except Exception as exc:
            logger.exception("Detection failed for job %s", job_id)
            job.status = "error"
            job.error = str(exc)
            self._save_job(job)
            state = self._load_state()
            state.detect_status = "error"
            state.detect_error = str(exc)
            self._save_state(state)
        finally:
            self._detect_threads.pop(job_id, None)

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

    def get_detection_manifest(self, job_id: str) -> dict[str, Any]:
        path = self._job_dir(job_id) / "detections.json"
        if not path.exists():
            raise FileNotFoundError("Detection results not ready")
        return json.loads(path.read_text(encoding="utf-8"))
