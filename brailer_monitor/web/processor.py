"""Background job processing for the web viewer."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from ..frame_extractor import ExtractOptions, extract_brailer_frames, save_segment_manifest, scan_brailer_segments
from ..label_format import load_frame_label
from ..labeling import label_extracted_frames

logger = logging.getLogger(__name__)


class JobStatus(StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class JobState:
    job_id: str
    status: JobStatus
    video_name: str
    created_at: str
    updated_at: str
    progress: float = 0.0
    message: str = ""
    frame_count: int = 0
    labeled_count: int = 0
    segment_count: int = 0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class JobManager:
    root: Path
    sam_model: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def create_job(self, video_path: Path, original_name: str) -> JobState:
        job_id = uuid.uuid4().hex[:12]
        directory = self.job_dir(job_id)
        frames_dir = directory / "frames"
        labels_dir = directory / "labels"
        previews_dir = directory / "previews"
        for path in (directory, frames_dir, labels_dir, previews_dir):
            path.mkdir(parents=True, exist_ok=True)

        target = directory / "video.mp4"
        if video_path.resolve() != target.resolve():
            target.write_bytes(video_path.read_bytes())

        now = _now_iso()
        state = JobState(
            job_id=job_id,
            status=JobStatus.UPLOADED,
            video_name=original_name,
            created_at=now,
            updated_at=now,
            message="Video uploaded. Ready to process.",
        )
        self._save_state(state)
        return state

    def get_state(self, job_id: str) -> JobState:
        path = self.job_dir(job_id) / "status.json"
        if not path.exists():
            raise FileNotFoundError(f"Job not found: {job_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["status"] = JobStatus(payload["status"])
        return JobState(**payload)

    def list_jobs(self) -> list[JobState]:
        jobs: list[JobState] = []
        for directory in sorted(self.root.iterdir(), reverse=True):
            if directory.is_dir() and (directory / "status.json").exists():
                try:
                    jobs.append(self.get_state(directory.name))
                except Exception:
                    continue
        return jobs

    def start_processing(
        self,
        job_id: str,
        *,
        scan_stride: int = 15,
        extract_stride: int = 15,
        skip_label: bool = False,
    ) -> None:
        state = self.get_state(job_id)
        if state.status == JobStatus.PROCESSING:
            return
        thread = threading.Thread(
            target=self._run_pipeline,
            args=(job_id, scan_stride, extract_stride, skip_label),
            daemon=True,
        )
        thread.start()

    def _run_pipeline(
        self,
        job_id: str,
        scan_stride: int,
        extract_stride: int,
        skip_label: bool,
    ) -> None:
        directory = self.job_dir(job_id)
        video = directory / "video.mp4"
        frames_dir = directory / "frames"
        labels_dir = directory / "labels"
        previews_dir = directory / "previews"

        try:
            self._update(job_id, JobStatus.PROCESSING, 5, "Scanning brailer segments...")
            opts = ExtractOptions(scan_stride=scan_stride, extract_stride=extract_stride)
            segments, fps, _ = scan_brailer_segments(video, opts)

            self._update(job_id, JobStatus.PROCESSING, 20, "Extracting frames...")
            extracted, segments = extract_brailer_frames(
                video,
                frames_dir,
                prefix=job_id,
                options=opts,
                segments=segments,
                preview_dir=previews_dir,
            )
            save_segment_manifest(
                segments,
                extracted,
                directory / "segments.json",
                video_path=video,
                fps=fps,
            )

            labeled = 0
            records: list[dict] = []
            if not skip_label and extracted:
                self._update(job_id, JobStatus.PROCESSING, 40, "Running SAM labeling...")
                from ultralytics import SAM

                sam = SAM(str(self.sam_model))
                total = len(extracted)
                batch_records: list[dict] = []
                for index, item in enumerate(extracted, start=1):
                    batch_records = label_extracted_frames(
                        [item], labels_dir, sam, preview_dir=previews_dir
                    )
                    if batch_records:
                        labeled += 1
                        records.extend(batch_records)
                    progress = 40 + (index / total) * 55
                    self._update(
                        job_id,
                        JobStatus.PROCESSING,
                        progress,
                        f"Labeling frame {index}/{total}...",
                    )
            elif extracted:
                records = [item.to_dict() for item in extracted]

            (directory / "label_manifest.json").write_text(
                json.dumps(records, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            state = self.get_state(job_id)
            state.frame_count = len(extracted)
            state.labeled_count = labeled if not skip_label else sum(
                1 for item in extracted if (labels_dir / f"{item.image_path.stem}.txt").exists()
            )
            state.segment_count = len(segments)
            state.progress = 100.0
            state.message = f"Done: {state.labeled_count}/{state.frame_count} frames labeled"
            state.status = JobStatus.COMPLETED
            state.updated_at = _now_iso()
            self._save_state(state)
        except Exception as exc:
            logger.exception("Job %s failed", job_id)
            state = self.get_state(job_id)
            state.status = JobStatus.FAILED
            state.error = str(exc)
            state.message = "Processing failed"
            state.updated_at = _now_iso()
            self._save_state(state)

    def list_frames(self, job_id: str) -> list[dict[str, Any]]:
        directory = self.job_dir(job_id)
        manifest_path = directory / "label_manifest.json"
        manifest: dict[str, dict] = {}
        if manifest_path.exists():
            for item in json.loads(manifest_path.read_text(encoding="utf-8")):
                manifest[item.get("image", "")] = item

        frames: list[dict[str, Any]] = []
        for image_path in sorted((directory / "frames").glob("*.jpg")):
            label_path = directory / "labels" / f"{image_path.stem}.txt"
            meta = manifest.get(image_path.name, {})
            frame_index = meta.get("frame_index")
            timestamp_sec = meta.get("timestamp_sec")
            if frame_index is None:
                frame_index = _parse_frame_index(image_path.stem)
            if timestamp_sec is None:
                timestamp_sec = _parse_timestamp(image_path.stem)

            label_info = None
            if label_path.exists():
                import cv2

                img = cv2.imread(str(image_path))
                h, w = (img.shape[:2] if img is not None else (720, 1280))
                seg = load_frame_label(label_path)
                if seg:
                    label_info = seg.to_dict(w, h)

            frames.append(
                {
                    "id": image_path.stem,
                    "image": image_path.name,
                    "frame_index": frame_index,
                    "timestamp_sec": timestamp_sec,
                    "segment_id": meta.get("segment_id"),
                    "has_label": label_path.exists(),
                    "label": label_info,
                    "bbox": meta.get("bbox"),
                    "area_ratio": meta.get("area_ratio") or (label_info or {}).get("area_ratio"),
                    "points": meta.get("points") or (label_info or {}).get("point_count"),
                }
            )
        return frames

    def get_segments(self, job_id: str) -> dict[str, Any]:
        path = self.job_dir(job_id) / "segments.json"
        if not path.exists():
            return {"segments": [], "frame_count": 0}
        return json.loads(path.read_text(encoding="utf-8"))

    def _update(self, job_id: str, status: JobStatus, progress: float, message: str) -> None:
        state = self.get_state(job_id)
        state.status = status
        state.progress = round(progress, 1)
        state.message = message
        state.updated_at = _now_iso()
        self._save_state(state)

    def _save_state(self, state: JobState) -> None:
        with self._lock:
            path = self.job_dir(state.job_id) / "status.json"
            path.write_text(json.dumps(state.to_dict(), indent=2), encoding="utf-8")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_timestamp(stem: str) -> float | None:
    # lake_win_00125s_f01875 or {jobid}_00125s_f01875
    for part in stem.split("_"):
        if part.endswith("s") and part[:-1].isdigit():
            return float(part[:-1])
    return None


def _parse_frame_index(stem: str) -> int | None:
    if stem.startswith("f") and stem[1:].isdigit():
        return int(stem[1:])
    for part in stem.split("_"):
        if part.startswith("f") and part[1:].isdigit():
            return int(part[1:])
    return None
