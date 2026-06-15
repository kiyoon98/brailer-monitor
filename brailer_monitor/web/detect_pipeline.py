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
from ..detect_timeline import (
    get_segment_frames,
    list_timeline,
    merge_job_manifest,
    reset_timeline,
    timeline_summary,
)
from ..lake_video_source import download_video
from ..train import TrainingCancelled, load_task_type, run_training
from ..video_detect import DetectionCancelled, detect_video
from ..video_time import parse_video_start_time

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
    detect_queue_pending: int = 0
    detect_batch_total: int = 0
    detect_batch_done: int = 0
    detect_timeline_events: int = 0
    detect_error: str | None = None
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _detect_is_active(state: PipelineState, *, threads: dict[str, threading.Thread]) -> bool:
    return (
        state.detect_status in {"running", "cancelling"}
        or (state.detect_queue_pending or 0) > 0
        or bool(threads)
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
        self._state_path = self.root / "pipeline_state.json"
        self._lock = threading.Lock()
        self._train_thread: threading.Thread | None = None
        self._detect_threads: dict[str, threading.Thread] = {}
        self._timeline_path = self.root / "detect_timeline.json"
        self._detect_queue: list[dict[str, Any]] = []
        self._batch_total = 0
        self._batch_done = 0
        self._detect_cancel = threading.Event()
        self._train_cancel = threading.Event()

    def _detection_cancelled(self) -> bool:
        return self._detect_cancel.is_set()

    def _clear_detection_cancel(self) -> None:
        self._detect_cancel.clear()

    def _clear_train_cancel(self) -> None:
        self._train_cancel.clear()

    def cancel_detection(self) -> dict[str, Any]:
        with self._lock:
            state = self._load_state()
            if state.detect_status == "cancelled":
                return {"cancelled": False, "reason": "already_cancelled"}
            if not _detect_is_active(state, threads=self._detect_threads):
                return {"cancelled": False, "reason": "not_running"}

            self._detect_cancel.set()
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
            if state.train_status != "running" or not self._train_thread_alive():
                return {"cancelled": False, "reason": "not_running"}

            self._train_cancel.set()
            state.train_progress = "cancelling"
            self._save_state(state)
            return {"cancelled": True}

    def _finish_detection_cancelled(self) -> None:
        state = self._load_state()
        state.detect_status = "cancelled"
        state.detect_queue_pending = 0
        state.detect_error = "사용자가 중지함"
        state.detect_batch_total = self._batch_total
        state.detect_batch_done = self._batch_done
        summary = timeline_summary(self._timeline_path)
        state.detect_timeline_events = summary["segment_count"]
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
        self._recover_stale_cancel()
        state = self._load_state()
        payload = state.to_dict()
        meta_path = self.dataset_root / "import_meta.json"
        if meta_path.exists():
            payload["dataset_meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        payload["detect_timeline"] = timeline_summary(self._timeline_path)
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
        batch: int = 8,
        imgsz: int = 640,
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
                should_cancel=self._train_cancel.is_set,
            )
            state = self._load_state()
            state.train_status = "completed"
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
        frame_stride: int = 5,
        confidence: float = 0.35,
        device: str | int = 0,
    ) -> DetectJob | dict[str, Any]:
        return self.start_detection_batch(
            [{"video_path": video_path, "video_name": video_name}],
            model_path=model_path,
            frame_stride=frame_stride,
            confidence=confidence,
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
        frame_stride: int = 5,
        confidence: float = 0.35,
        device: str | int = 0,
    ) -> dict[str, Any]:
        if not videos:
            raise ValueError("No videos provided")

        state = self._load_state()
        if model_path is None:
            task = load_task_type(self.dataset_root)
            default = Path("models/brailer_detect.pt" if task == "detect" else "models/brailer_seg.pt")
            model_path = Path(state.train_weights) if state.train_weights else default
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

        self._clear_detection_cancel()
        prepared_videos = self._prepare_batch_videos(videos)
        queued_jobs: list[dict[str, Any]] = []
        with self._lock:
            state = self._load_state()
            running = state.detect_status == "running"

            if not running:
                self._batch_total = len(prepared_videos)
                self._batch_done = 0
                state.detect_batch_total = self._batch_total
                state.detect_batch_done = 0
                state.detect_status = "running"
                state.detect_error = None
            else:
                self._batch_total += len(prepared_videos)
                state.detect_batch_total = self._batch_total

            for index, prepared in enumerate(prepared_videos):
                item = {
                    "video_name": prepared["video_name"],
                    "model_path": model_path,
                    "frame_stride": frame_stride,
                    "confidence": confidence,
                    "device": device,
                }
                if prepared.get("remote_url"):
                    item["remote_url"] = prepared["remote_url"]
                else:
                    item["video_path"] = prepared["video_path"]
                if not running and index == 0:
                    job = self._launch_detection(item)
                    queued_jobs.append(
                        {"job_id": job.job_id, "video_name": prepared["video_name"], "status": "running"}
                    )
                    running = True
                else:
                    self._detect_queue.append(item)
                    queued_jobs.append({"video_name": prepared["video_name"], "status": "queued"})

            state = self._load_state()
            state.detect_queue_pending = len(self._detect_queue)
            state.detect_batch_total = self._batch_total
            state.detect_batch_done = self._batch_done
            self._save_state(state)

        return {
            "batch_size": len(prepared_videos),
            "batch_total": self._batch_total,
            "batch_done": self._batch_done,
            "queue_pending": len(self._detect_queue),
            "jobs": queued_jobs,
            "job": queued_jobs[0] if queued_jobs and queued_jobs[0].get("job_id") else None,
        }

    def _launch_detection(self, item: dict[str, Any]) -> DetectJob:
        video_name: str = item["video_name"]
        model_path: Path = item["model_path"]
        frame_stride: int = item["frame_stride"]
        confidence: float = item["confidence"]
        device: str | int = item["device"]

        job_id = uuid.uuid4().hex[:12]
        directory = self._job_dir(job_id)
        directory.mkdir(parents=True, exist_ok=True)
        target_video = directory / "video.mp4"
        if self._detection_cancelled():
            raise DetectionCancelled("Detection cancelled by user")
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
        }
        (directory / "video_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state = self._load_state()
        if self._detection_cancelled() or state.detect_status == "cancelling":
            raise DetectionCancelled("Detection cancelled by user")
        state.detect_status = "running"
        state.detect_job_id = job_id
        state.detect_progress_pct = 0.0
        state.detect_processed_frames = 0
        state.detect_total_frames = 0
        state.detect_frames_with_objects = 0
        state.detect_error = None
        state.detect_queue_pending = len(self._detect_queue)
        state.detect_batch_total = self._batch_total
        state.detect_batch_done = self._batch_done
        self._save_state(state)

        thread = threading.Thread(
            target=self._run_detection,
            args=(job_id, target_video, model_path, frame_stride, confidence, device, video_name),
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
        video_name: str,
    ) -> None:
        job = self.get_detect_job(job_id)
        try:
            if self._detection_cancelled():
                raise DetectionCancelled("Detection cancelled by user")

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
                should_cancel=self._detection_cancelled,
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
            state.detect_timeline_events = summary["segment_count"]
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
            state = self._load_state()
            state.detect_status = "error"
            state.detect_error = str(exc)
            self._save_state(state)
            self._detect_queue.clear()
        finally:
            self._detect_threads.pop(job_id, None)

    def _start_next_queued_or_finish(self, *, success: bool) -> None:
        next_item: dict[str, Any] | None = None
        with self._lock:
            if self._detection_cancelled():
                self._finish_detection_cancelled()
                return

            if self._detect_queue:
                next_item = self._detect_queue.pop(0)
                state = self._load_state()
                state.detect_queue_pending = len(self._detect_queue)
                self._save_state(state)
            else:
                state = self._load_state()
                state.detect_status = "completed" if success else state.detect_status
                state.detect_queue_pending = 0
                state.detect_batch_total = self._batch_total
                state.detect_batch_done = self._batch_done
                summary = timeline_summary(self._timeline_path)
                state.detect_timeline_events = summary["segment_count"]
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
