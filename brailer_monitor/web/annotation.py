"""Manual polygon annotation job storage."""

from __future__ import annotations

import json
import logging
import shutil
import threading
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..label_format import load_frame_label, parse_yolo_seg_line
from ..oneshot import ReferenceSignature, build_reference, detect_oneshot, load_sam_model

logger = logging.getLogger(__name__)


@dataclass
class AnnotationJob:
    job_id: str
    video_name: str
    created_at: str
    updated_at: str
    fps: float = 15.0
    duration_sec: float = 0.0
    frame_count: int = 0
    width: int = 0
    height: int = 0
    annotated_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameRecord:
    frame_id: str
    timestamp_sec: float
    frame_index: int
    image: str
    has_label: bool = False
    source: str = "manual"
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_SAM_MODEL = Path(__file__).resolve().parents[2] / "models" / "mobile_sam.pt"


@dataclass
class AnnotationManager:
    root: Path
    sam_model_path: Path = DEFAULT_SAM_MODEL

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._auto_threads: dict[str, threading.Thread] = {}

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create_job(self, video_path: Path, original_name: str) -> AnnotationJob:
        job_id = uuid.uuid4().hex[:12]
        directory = self.job_dir(job_id)
        frames_dir = directory / "frames"
        labels_dir = directory / "labels"
        for path in (directory, frames_dir, labels_dir):
            path.mkdir(parents=True, exist_ok=True)

        target = directory / "video.mp4"
        target.write_bytes(video_path.read_bytes())

        cap = cv2.VideoCapture(str(target))
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        now = _now_iso()
        job = AnnotationJob(
            job_id=job_id,
            video_name=original_name,
            created_at=now,
            updated_at=now,
            fps=fps,
            duration_sec=total / fps if fps else 0,
            frame_count=total,
            width=width,
            height=height,
        )
        self._save_job(job)
        self._save_manifest(job_id, [])
        return job

    def get_job(self, job_id: str) -> AnnotationJob:
        path = self.job_dir(job_id) / "job.json"
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        return AnnotationJob(**json.loads(path.read_text(encoding="utf-8")))

    def list_jobs(self) -> list[AnnotationJob]:
        jobs: list[AnnotationJob] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if directory.is_dir() and (directory / "job.json").exists():
                try:
                    jobs.append(self.get_job(directory.name))
                except Exception:
                    continue
        return jobs

    def capture_frame(self, job_id: str, timestamp_sec: float) -> FrameRecord:
        job = self.get_job(job_id)
        directory = self.job_dir(job_id)
        video = directory / "video.mp4"

        timestamp_sec = max(0.0, min(timestamp_sec, job.duration_sec))
        frame_index = int(round(timestamp_sec * job.fps))

        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Failed to read frame at {timestamp_sec}s")

        frame_id = f"t{int(timestamp_sec * 1000):07d}_f{frame_index:05d}"
        image_name = f"{frame_id}.jpg"
        image_path = directory / "frames" / image_name
        cv2.imwrite(str(image_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        record = FrameRecord(
            frame_id=frame_id,
            timestamp_sec=round(timestamp_sec, 3),
            frame_index=frame_index,
            image=image_name,
            has_label=(directory / "labels" / f"{frame_id}.txt").exists(),
        )

        manifest = self._load_manifest(job_id)
        manifest = [r for r in manifest if r["frame_id"] != frame_id]
        manifest.append(record.to_dict())
        manifest.sort(key=lambda item: item["timestamp_sec"])
        self._save_manifest(job_id, manifest)
        self._touch_job(job_id)
        return record

    def list_frames(self, job_id: str) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        directory = self.job_dir(job_id)
        frames: list[dict[str, Any]] = []

        for item in self._load_manifest(job_id):
            label_path = directory / "labels" / f"{item['frame_id']}.txt"
            label_info = None
            if label_path.exists():
                seg = load_frame_label(label_path)
                if seg:
                    label_info = seg.to_dict(job.width, job.height)
            frames.append(
                {
                    **item,
                    "has_label": label_path.exists(),
                    "label": label_info,
                    "source": item.get("source", "manual" if label_path.exists() else None),
                    "score": item.get("score"),
                }
            )
        return frames

    def get_frame(self, job_id: str, frame_id: str) -> dict[str, Any]:
        for frame in self.list_frames(job_id):
            if frame["frame_id"] == frame_id:
                return frame
        raise FileNotFoundError(f"Frame not found: {frame_id}")

    def build_reference_from_job(self, job_id: str) -> ReferenceSignature:
        """Build reference signature from manually labeled class-0 frames."""
        directory = self.job_dir(job_id)
        job = self.get_job(job_id)
        samples: list[tuple[np.ndarray, list[tuple[float, float]]]] = []

        for item in self._load_manifest(job_id):
            if not item.get("has_label"):
                continue
            if item.get("source") == "auto":
                continue
            label_path = directory / "labels" / f"{item['frame_id']}.txt"
            seg = load_frame_label(label_path)
            if seg is None or seg.class_id != 0:
                continue
            image_path = directory / "frames" / item["image"]
            frame = cv2.imread(str(image_path))
            if frame is None:
                continue
            samples.append((frame, seg.polygon_norm))

        if not samples:
            raise ValueError("No manual brailer_loaded reference labels found")
        return build_reference(samples)

    def _autodetect_state_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "autodetect.json"

    def get_autodetect_state(self, job_id: str) -> dict[str, Any]:
        path = self._autodetect_state_path(job_id)
        if not path.exists():
            return {"status": "idle", "progress": 0.0, "processed": 0, "detected": 0, "total": 0}
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_autodetect_state(self, job_id: str, state: dict[str, Any]) -> None:
        path = self._autodetect_state_path(job_id)
        path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def start_auto_detect(
        self,
        job_id: str,
        *,
        interval_sec: float = 5.0,
        threshold: float = 0.55,
    ) -> dict[str, Any]:
        current = self.get_autodetect_state(job_id)
        if current.get("status") == "running":
            raise RuntimeError("Auto-detect already running for this job")

        thread = threading.Thread(
            target=self._run_auto_detect,
            args=(job_id, interval_sec, threshold),
            daemon=True,
        )
        self._auto_threads[job_id] = thread
        thread.start()
        return {"started": True, "job_id": job_id}

    def _run_auto_detect(
        self,
        job_id: str,
        interval_sec: float,
        threshold: float,
    ) -> None:
        try:
            job = self.get_job(job_id)
            signature = self.build_reference_from_job(job_id)
            directory = self.job_dir(job_id)
            video = directory / "video.mp4"

            sam_model = None
            if self.sam_model_path.exists():
                sam_model = load_sam_model(str(self.sam_model_path))

            timestamps: list[float] = []
            t = 0.0
            while t <= job.duration_sec:
                timestamps.append(t)
                t += interval_sec

            existing_ts = {
                round(item["timestamp_sec"], 2)
                for item in self._load_manifest(job_id)
            }

            state = {
                "status": "running",
                "progress": 0.0,
                "processed": 0,
                "detected": 0,
                "total": len(timestamps),
                "error": None,
            }
            self._save_autodetect_state(job_id, state)

            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or job.fps

            for index, ts in enumerate(timestamps):
                if round(ts, 2) in existing_ts:
                    state["processed"] = index + 1
                    state["progress"] = (index + 1) / len(timestamps)
                    self._save_autodetect_state(job_id, state)
                    continue

                frame_index = int(round(ts * fps))
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue

                try:
                    result = detect_oneshot(frame, signature, sam_model, threshold=threshold)
                except Exception as exc:
                    logger.warning(
                        "Auto-detect skipped frame at %.1fs (index %d): %s",
                        ts,
                        frame_index,
                        exc,
                    )
                    result = None

                state["processed"] = index + 1
                state["progress"] = (index + 1) / len(timestamps)

                if result is not None:
                    record = self.capture_frame(job_id, ts)
                    self.save_label(
                        job_id,
                        record.frame_id,
                        [[x, y] for x, y in result.polygon_norm],
                        class_id=0,
                        source="auto",
                        score=result.confidence,
                    )
                    state["detected"] += 1
                    existing_ts.add(round(ts, 2))

                self._save_autodetect_state(job_id, state)

            cap.release()
            state["status"] = "completed"
            state["progress"] = 1.0
            self._save_autodetect_state(job_id, state)
        except Exception as exc:
            logger.exception("Auto-detect failed for job %s", job_id)
            err_state = self.get_autodetect_state(job_id)
            err_state["status"] = "error"
            err_state["error"] = str(exc)
            self._save_autodetect_state(job_id, err_state)
        finally:
            self._auto_threads.pop(job_id, None)

    def save_label(
        self,
        job_id: str,
        frame_id: str,
        polygon_norm: list[list[float]],
        class_id: int = 0,
        *,
        source: str = "manual",
        score: float | None = None,
    ) -> dict[str, Any]:
        if len(polygon_norm) < 3:
            raise ValueError("Polygon must have at least 3 points")

        directory = self.job_dir(job_id)
        image_path = directory / "frames" / f"{frame_id}.jpg"
        if not image_path.exists():
            raise FileNotFoundError(f"Frame image not found: {frame_id}")

        parts = [str(class_id)]
        for x, y in polygon_norm:
            parts.append(f"{float(x):.6f}")
            parts.append(f"{float(y):.6f}")
        line = " ".join(parts) + "\n"

        label_path = directory / "labels" / f"{frame_id}.txt"
        label_path.write_text(line, encoding="utf-8")

        manifest = self._load_manifest(job_id)
        for item in manifest:
            if item["frame_id"] == frame_id:
                item["has_label"] = True
                item["source"] = source
                item["score"] = score
        self._save_manifest(job_id, manifest)
        self._update_annotated_count(job_id)
        self._touch_job(job_id)

        job = self.get_job(job_id)
        seg = parse_yolo_seg_line(line)
        label_info = seg.to_dict(job.width, job.height) if seg else None
        return {"frame_id": frame_id, "saved": True, "label": label_info}

    def delete_label(self, job_id: str, frame_id: str) -> dict[str, Any]:
        label_path = self.job_dir(job_id) / "labels" / f"{frame_id}.txt"
        if label_path.exists():
            label_path.unlink()

        manifest = self._load_manifest(job_id)
        for item in manifest:
            if item["frame_id"] == frame_id:
                item["has_label"] = False
                item["source"] = None
                item["score"] = None
        self._save_manifest(job_id, manifest)
        self._update_annotated_count(job_id)
        self._touch_job(job_id)
        return {"frame_id": frame_id, "deleted": True}

    def delete_frame(self, job_id: str, frame_id: str) -> dict[str, Any]:
        directory = self.job_dir(job_id)
        (directory / "frames" / f"{frame_id}.jpg").unlink(missing_ok=True)
        (directory / "labels" / f"{frame_id}.txt").unlink(missing_ok=True)

        manifest = [r for r in self._load_manifest(job_id) if r["frame_id"] != frame_id]
        self._save_manifest(job_id, manifest)
        self._update_annotated_count(job_id)
        self._touch_job(job_id)
        return {"frame_id": frame_id, "removed": True}

    def export_dataset(self, job_id: str, dataset_root: Path) -> dict[str, int]:
        """Copy manual annotations to data/dataset for YOLO training."""
        job = self.get_job(job_id)
        directory = self.job_dir(job_id)
        train_img = dataset_root / "images" / "train"
        train_lbl = dataset_root / "labels" / "train"
        for path in (train_img, train_lbl):
            path.mkdir(parents=True, exist_ok=True)

        count = 0
        for item in self._load_manifest(job_id):
            if not item.get("has_label"):
                continue
            frame_id = item["frame_id"]
            src_img = directory / "frames" / f"{frame_id}.jpg"
            src_lbl = directory / "labels" / f"{frame_id}.txt"
            if src_img.exists() and src_lbl.exists():
                shutil.copy2(src_img, train_img / f"{job_id}_{frame_id}.jpg")
                shutil.copy2(src_lbl, train_lbl / f"{job_id}_{frame_id}.txt")
                count += 1
        return {"exported": count}

    def _load_manifest(self, job_id: str) -> list[dict[str, Any]]:
        path = self.job_dir(job_id) / "manifest.json"
        if not path.exists():
            return []
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_manifest(self, job_id: str, manifest: list[dict[str, Any]]) -> None:
        path = self.job_dir(job_id) / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_job(self, job: AnnotationJob) -> None:
        path = self.job_dir(job.job_id) / "job.json"
        path.write_text(json.dumps(job.to_dict(), indent=2), encoding="utf-8")

    def _touch_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        job.updated_at = _now_iso()
        self._save_job(job)

    def _update_annotated_count(self, job_id: str) -> None:
        job = self.get_job(job_id)
        labels_dir = self.job_dir(job_id) / "labels"
        job.annotated_count = len(list(labels_dir.glob("*.txt")))
        self._save_job(job)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
