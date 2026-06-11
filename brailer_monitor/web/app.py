"""Web app: CVAT import, YOLO training, video detection."""

from __future__ import annotations

import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .annotation import AnnotationManager
from .detect_pipeline import DetectPipelineManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
JOBS_ROOT = PROJECT_ROOT / "data" / "web_jobs"
PIPELINE_ROOT = PROJECT_ROOT / "data" / "pipeline"
DATASET_ROOT = PROJECT_ROOT / "data" / "dataset"
CONFIG_DIR = PROJECT_ROOT / "config"
RAW_VIDEO = PROJECT_ROOT / "data" / "raw" / "JJR-102283_stream04_260310_202016.mp4"

manager = AnnotationManager(root=JOBS_ROOT)
pipeline = DetectPipelineManager(
    root=PIPELINE_ROOT,
    dataset_root=DATASET_ROOT,
    config_dir=CONFIG_DIR,
)

app = FastAPI(title="Brailer Monitor", version="0.3.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class CaptureRequest(BaseModel):
    timestamp_sec: float = Field(ge=0)


class LabelRequest(BaseModel):
    polygon_norm: list[list[float]]
    class_id: int = 0


class AutoDetectRequest(BaseModel):
    interval_sec: float = Field(default=5.0, ge=0.5, le=60.0)
    threshold: float = Field(default=0.55, ge=0.1, le=1.0)


class TrainRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    batch: int = Field(default=8, ge=1, le=64)
    imgsz: int = Field(default=640, ge=320, le=1280)
    device: str | int = 0


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "detect.html")


@app.get("/annotate")
async def annotate_page() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/pipeline/state")
async def pipeline_state() -> dict:
    return {"state": pipeline.get_state()}


@app.post("/api/pipeline/import-cvat")
async def pipeline_import_cvat(
    file: UploadFile = File(...),
    video: UploadFile | None = File(None),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No annotations file")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".zip", ".xml"}:
        raise HTTPException(status_code=400, detail="Upload CVAT .zip or annotations.xml")

    temp = PIPELINE_ROOT / "_uploads"
    temp.mkdir(parents=True, exist_ok=True)
    ann_path = temp / file.filename
    video_path: Path | None = None

    with ann_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        if video and video.filename:
            v_suffix = Path(video.filename).suffix.lower()
            if v_suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
                raise HTTPException(status_code=400, detail="Unsupported video format")
            video_path = temp / video.filename
            with video_path.open("wb") as handle:
                shutil.copyfileobj(video.file, handle)

        return pipeline.import_cvat(ann_path, video_path=video_path)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        ann_path.unlink(missing_ok=True)
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)


@app.post("/api/pipeline/train")
async def pipeline_train(body: TrainRequest) -> dict:
    try:
        return pipeline.start_training(
            epochs=body.epochs,
            batch=body.batch,
            imgsz=body.imgsz,
            device=body.device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/pipeline/detect")
async def pipeline_detect(
    file: UploadFile = File(...),
    frame_stride: int = Form(5),
    confidence: float = Form(0.35),
    device: str | int = Form(0),
) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp = PIPELINE_ROOT / "_uploads"
    temp.mkdir(parents=True, exist_ok=True)
    temp_path = temp / file.filename
    with temp_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)
    try:
        job = pipeline.start_detection(
            temp_path,
            file.filename,
            frame_stride=frame_stride,
            confidence=confidence,
            device=device,
        )
        return {"job": job.to_dict()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        temp_path.unlink(missing_ok=True)


@app.get("/api/pipeline/detect/{job_id}")
async def pipeline_get_detect_job(job_id: str) -> dict:
    try:
        return {"job": pipeline.get_detect_job(job_id).to_dict()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/{job_id}/results")
async def pipeline_detect_results(job_id: str) -> dict:
    try:
        return {"manifest": pipeline.get_detection_manifest(job_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/{job_id}/previews/{filename}")
async def pipeline_detect_preview(job_id: str, filename: str) -> FileResponse:
    path = pipeline._job_dir(job_id) / "previews" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path)


@app.get("/api/jobs")
async def list_jobs() -> dict:
    jobs = manager.list_jobs()
    return {"jobs": [job.to_dict() for job in jobs]}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(status_code=400, detail="Unsupported video format")

    temp = JOBS_ROOT / "_uploads"
    temp.mkdir(parents=True, exist_ok=True)
    temp_path = temp / file.filename
    with temp_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    job = manager.create_job(temp_path, file.filename)
    temp_path.unlink(missing_ok=True)
    return {"job": job.to_dict()}


@app.post("/api/open-local")
async def open_local_video() -> dict:
    """Open the default raw EM video if present."""
    if not RAW_VIDEO.exists():
        raise HTTPException(status_code=404, detail=f"Video not found: {RAW_VIDEO}")
    job = manager.create_job(RAW_VIDEO, RAW_VIDEO.name)
    return {"job": job.to_dict()}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict:
    try:
        return {"job": manager.get_job(job_id).to_dict()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/capture")
async def capture_frame(job_id: str, body: CaptureRequest) -> dict:
    try:
        record = manager.capture_frame(job_id, body.timestamp_sec)
        return {"frame": record.to_dict()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/frames")
async def list_frames(job_id: str) -> dict:
    try:
        frames = manager.list_frames(job_id)
        return {"frames": frames, "count": len(frames)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/frames/{frame_id}")
async def get_frame(job_id: str, frame_id: str) -> dict:
    try:
        return {"frame": manager.get_frame(job_id, frame_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/jobs/{job_id}/frames/{frame_id}/label")
async def save_label(job_id: str, frame_id: str, body: LabelRequest) -> dict:
    try:
        return manager.save_label(job_id, frame_id, body.polygon_norm, body.class_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}/frames/{frame_id}/label")
async def delete_label(job_id: str, frame_id: str) -> dict:
    try:
        return manager.delete_label(job_id, frame_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/jobs/{job_id}/frames/{frame_id}")
async def delete_frame(job_id: str, frame_id: str) -> dict:
    try:
        return manager.delete_frame(job_id, frame_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/auto-detect")
async def start_auto_detect(job_id: str, body: AutoDetectRequest) -> dict:
    try:
        return manager.start_auto_detect(
            job_id,
            interval_sec=body.interval_sec,
            threshold=body.threshold,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/auto-detect")
async def get_auto_detect_status(job_id: str) -> dict:
    try:
        manager.get_job(job_id)
        return {"state": manager.get_autodetect_state(job_id)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/jobs/{job_id}/export")
async def export_dataset(job_id: str) -> dict:
    try:
        dataset_root = PROJECT_ROOT / "data" / "dataset"
        result = manager.export_dataset(job_id, dataset_root)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/jobs/{job_id}/media/frames/{filename}")
async def get_frame_image(job_id: str, filename: str) -> FileResponse:
    path = manager.job_dir(job_id) / "frames" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path)


@app.get("/api/jobs/{job_id}/media/video")
async def get_video(job_id: str) -> FileResponse:
    path = manager.job_dir(job_id) / "video.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4")
