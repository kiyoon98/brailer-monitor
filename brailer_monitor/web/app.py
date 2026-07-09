"""Web app: CVAT import, YOLO training, video detection."""

from __future__ import annotations

import json
from functools import partial
import shutil
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
import cv2

from ..cvat_import import inspect_cvat
from ..dataset_preview import list_dataset_frames, render_dataset_preview
from ..detect_report import write_detection_report_bundle
from ..lake_video_source import (
    build_lake_config_from_selection,
    discover_videos_in_range,
    list_candidate_videos,
    load_lake_component_spec,
)
from .annotation import AnnotationManager
from .detect_pipeline import DetectPipelineManager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = Path(__file__).resolve().parent / "static"
JOBS_ROOT = PROJECT_ROOT / "data" / "web_jobs"
PIPELINE_ROOT = PROJECT_ROOT / "data" / "pipeline"
REPORTS_ROOT = PIPELINE_ROOT / "reports"
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


class LakeRangeRequest(BaseModel):
    media: str | None = None
    year_folder: str | None = None
    vessel: str | None = None
    stream: str | None = None
    minute_offsets: list[int] | None = None
    minute_slots: list[int] | None = None
    second_suffixes: list[str] | None = None
    model_ids: list[str] | None = None
    start_month: int = Field(ge=1, le=12)
    start_day: int = Field(ge=1, le=31)
    start_hour: int = Field(ge=0, le=23)
    end_month: int = Field(ge=1, le=12)
    end_day: int = Field(ge=1, le=31)
    end_hour: int = Field(ge=0, le=23)
    frame_stride: int = Field(default=5, ge=1)
    confidence: float = Field(default=0.8, ge=0.05, le=1.0)
    imgsz: int = Field(default=416, ge=320, le=1280)
    use_sam: bool = True
    skip_dark_video: bool = True
    device: str | int = 0
    check_exists: bool = True


class StreamDetectRequest(BaseModel):
    stream_url: str = Field(default="http://127.0.0.1:8081/live_04.m3u8", min_length=1)
    model_ids: list[str] | None = None
    frame_stride: int = Field(default=5, ge=1)
    confidence: float = Field(default=0.8, ge=0.05, le=1.0)
    imgsz: int = Field(default=416, ge=320, le=1280)
    use_sam: bool = True
    device: str | int = 0


class TimelineCompactRequest(BaseModel):
    max_gap_sec: float = Field(default=8.0, ge=0.0, le=60.0)
    merge_segments: bool = True
    remove_position_outliers: bool = True
    remove_size_outliers: bool = True
    remove_tall_thin_boxes: bool = True
    remove_static_short_tracks: bool = True
    remove_temporal_isolated: bool = True
    remove_color_outliers: bool = True


class SaveDetectionResultsRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class DetectionReportRequest(BaseModel):
    saved_result_id: str | None = None


class LabelRequest(BaseModel):
    polygon_norm: list[list[float]]
    class_id: int = 0


class AutoDetectRequest(BaseModel):
    interval_sec: float = Field(default=5.0, ge=0.5, le=60.0)
    threshold: float = Field(default=0.55, ge=0.1, le=1.0)


class TrainRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    batch: int = Field(default=4, ge=1, le=64)
    imgsz: int = Field(default=416, ge=320, le=1280)
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


@app.get("/api/pipeline/dataset/frames")
async def pipeline_dataset_frames(
    split: str = "all",
    offset: int = 0,
    limit: int = 60,
) -> dict:
    try:
        return list_dataset_frames(
            DATASET_ROOT,
            split=split,
            offset=offset,
            limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline/dataset/preview/{split}/{filename}")
async def pipeline_dataset_preview(split: str, filename: str, width: int = 0) -> Response:
    try:
        vis = render_dataset_preview(DATASET_ROOT, split, filename)
        if width > 0:
            width = max(120, min(width, 1280))
            height, original_width = vis.shape[:2]
            if original_width > width:
                scale = width / original_width
                vis = cv2.resize(
                    vis,
                    (width, max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
        quality = 78 if width else 88
        ok, buf = cv2.imencode(".jpg", vis, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            raise RuntimeError("Failed to encode preview")
        return Response(content=buf.tobytes(), media_type="image/jpeg")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipeline/preview-cvat")
async def pipeline_preview_cvat(file: UploadFile = File(...)) -> dict:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No annotations file")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".zip", ".xml"}:
        raise HTTPException(status_code=400, detail="Upload CVAT .zip or annotations.xml")

    temp = PIPELINE_ROOT / "_uploads"
    temp.mkdir(parents=True, exist_ok=True)
    ann_path = temp / file.filename

    with ann_path.open("wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        return inspect_cvat(ann_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        ann_path.unlink(missing_ok=True)


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


@app.post("/api/pipeline/train/stop")
async def pipeline_train_stop() -> dict:
    return pipeline.cancel_training()


@app.post("/api/pipeline/train/reset")
async def pipeline_train_reset() -> dict:
    try:
        return pipeline.reset_training()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/pipeline/train/reset-and-start")
async def pipeline_train_reset_and_start(body: TrainRequest) -> dict:
    try:
        return pipeline.reset_and_start_training(
            epochs=body.epochs,
            batch=body.batch,
            imgsz=body.imgsz,
            device=body.device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


class RenameModelRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


@app.get("/api/pipeline/models")
async def pipeline_models() -> dict:
    return pipeline.list_models()


@app.post("/api/pipeline/models/{model_id}/activate")
async def pipeline_model_activate(model_id: str) -> dict:
    try:
        return pipeline.activate_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/api/pipeline/models/{model_id}/frames")
async def pipeline_model_frames(model_id: str, limit: int = 48) -> dict:
    try:
        return pipeline.model_frames(model_id, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pipeline/models/{model_id}/preview/{split}/{filename}")
async def pipeline_model_preview(model_id: str, split: str, filename: str):
    try:
        return FileResponse(pipeline.resolve_model_preview(model_id, split, filename))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pipeline/models/{model_id}/rename")
async def pipeline_model_rename(model_id: str, body: RenameModelRequest) -> dict:
    try:
        return pipeline.rename_model(model_id, body.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/pipeline/models/{model_id}")
async def pipeline_model_delete(model_id: str) -> dict:
    try:
        return pipeline.delete_model(model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipeline/detect/stop")
async def pipeline_detect_stop() -> dict:
    return pipeline.cancel_detection()


@app.post("/api/pipeline/detect")
async def pipeline_detect(
    files: list[UploadFile] = File(...),
    model_ids: list[str] | None = Form(None),
    frame_stride: int = Form(5),
    confidence: float = Form(0.8),
    imgsz: int = Form(416),
    use_sam: bool = Form(True),
    skip_dark_video: bool = Form(True),
    device: str | int = Form(0),
) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="No video files")

    temp = PIPELINE_ROOT / "_uploads"
    temp.mkdir(parents=True, exist_ok=True)
    saved: list[tuple[Path, str]] = []

    try:
        for upload in files:
            if not upload.filename:
                continue
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
                raise HTTPException(status_code=400, detail=f"Unsupported format: {upload.filename}")
            temp_path = temp / upload.filename
            with temp_path.open("wb") as handle:
                shutil.copyfileobj(upload.file, handle)
            saved.append({"video_path": temp_path, "video_name": upload.filename})

        if not saved:
            raise HTTPException(status_code=400, detail="No valid video files")

        return pipeline.start_detection_batch(
            saved,
            model_ids=model_ids,
            frame_stride=frame_stride,
            confidence=confidence,
            imgsz=imgsz,
            use_sam=use_sam,
            skip_dark_video=skip_dark_video,
            device=device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        for item in saved:
            item["video_path"].unlink(missing_ok=True)


@app.get("/api/pipeline/lake-videos/config")
async def pipeline_lake_video_config() -> dict:
    spec = load_lake_component_spec(CONFIG_DIR / "lake_video.json")
    return {
        "base_host": spec["base_host"],
        "components": spec["components"],
        "minute_offsets": spec["minute_offsets"],
        "minute_slots": spec["minute_slots"],
    }


def _load_lake_config(body: LakeRangeRequest):
    return build_lake_config_from_selection(
        {
            "media": body.media,
            "year_folder": body.year_folder,
            "vessel": body.vessel,
            "stream": body.stream,
            "minute_offsets": body.minute_offsets,
            "minute_slots": body.minute_slots,
            "second_suffixes": body.second_suffixes,
        },
        path=CONFIG_DIR / "lake_video.json",
    )


@app.post("/api/pipeline/lake-videos/discover")
async def pipeline_lake_video_discover(body: LakeRangeRequest) -> dict:
    try:
        config = _load_lake_config(body)
        candidates = list_candidate_videos(
            start_month=body.start_month,
            start_day=body.start_day,
            start_hour=body.start_hour,
            end_month=body.end_month,
            end_day=body.end_day,
            end_hour=body.end_hour,
            config=config,
        )
        videos = await run_in_threadpool(
            partial(
                discover_videos_in_range,
                start_month=body.start_month,
                start_day=body.start_day,
                start_hour=body.start_hour,
                end_month=body.end_month,
                end_day=body.end_day,
                end_hour=body.end_hour,
                config=config,
                check_exists=body.check_exists,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "candidate_count": len(candidates),
        "found_count": len(videos),
        "profile": config.profile_id,
        "base_url": config.base_url,
        "file_prefix": config.file_prefix,
        "year": config.year,
        "minute_slots": list(config.minute_slots),
        "second_suffixes": list(config.second_suffixes),
        "videos": videos[:24],
        "sample_missing": max(0, len(candidates) - len(videos)),
    }


@app.post("/api/pipeline/detect/lake")
async def pipeline_detect_lake(body: LakeRangeRequest) -> dict:
    try:
        config = _load_lake_config(body)
        videos = await run_in_threadpool(
            partial(
                discover_videos_in_range,
                start_month=body.start_month,
                start_day=body.start_day,
                start_hour=body.start_hour,
                end_month=body.end_month,
                end_day=body.end_day,
                end_hour=body.end_hour,
                config=config,
                check_exists=body.check_exists,
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not videos:
        raise HTTPException(status_code=404, detail="해당 구간에서 비디오를 찾지 못했습니다.")

    payload = [{"video_name": video["filename"], "remote_url": video["url"]} for video in videos]
    try:
        return pipeline.start_detection_batch(
            payload,
            model_ids=body.model_ids,
            frame_stride=body.frame_stride,
            confidence=body.confidence,
            imgsz=body.imgsz,
            use_sam=body.use_sam,
            skip_dark_video=body.skip_dark_video,
            device=body.device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pipeline/detect/stream")
async def pipeline_detect_stream(body: StreamDetectRequest) -> dict:
    try:
        return pipeline.start_stream_detection(
            body.stream_url,
            model_ids=body.model_ids,
            frame_stride=body.frame_stride,
            confidence=body.confidence,
            imgsz=body.imgsz,
            use_sam=body.use_sam,
            device=body.device,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/timeline")
async def pipeline_detect_timeline(offset: int = 0, limit: int = 60) -> dict:
    return {"timeline": pipeline.get_timeline(offset=offset, limit=limit)}


@app.get("/api/pipeline/detect/timeline/segment/{segment_id}")
async def pipeline_detect_timeline_segment(segment_id: str) -> dict:
    try:
        return pipeline.get_timeline_segment(segment_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/pipeline/detect/timeline/reset")
async def pipeline_detect_timeline_reset() -> dict:
    return {"timeline": pipeline.reset_timeline()}


@app.get("/api/pipeline/detect/timeline/reset/redirect")
async def pipeline_detect_timeline_reset_redirect() -> RedirectResponse:
    pipeline.reset_timeline()
    return RedirectResponse(url="/", status_code=303)


@app.post("/api/pipeline/detect/timeline/compact")
async def pipeline_detect_timeline_compact(body: TimelineCompactRequest | None = None) -> dict:
    body = body or TimelineCompactRequest()
    try:
        return {
            "timeline": pipeline.compact_timeline(
                max_gap_sec=body.max_gap_sec,
                merge_segments=body.merge_segments,
                remove_position_outliers=body.remove_position_outliers,
                remove_size_outliers=body.remove_size_outliers,
                remove_tall_thin_boxes=body.remove_tall_thin_boxes,
                remove_static_short_tracks=body.remove_static_short_tracks,
                remove_temporal_isolated=body.remove_temporal_isolated,
                remove_color_outliers=body.remove_color_outliers,
            )
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/results")
async def pipeline_detect_saved_results() -> dict:
    return {"results": pipeline.list_saved_results()}


@app.post("/api/pipeline/detect/results")
async def pipeline_detect_save_results(body: SaveDetectionResultsRequest) -> dict:
    try:
        return {"result": pipeline.save_current_results(body.name)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/pipeline/detect/results/{result_id}/load")
async def pipeline_detect_load_results(result_id: str) -> dict:
    try:
        return {"result": pipeline.load_saved_results(result_id), "timeline": pipeline.get_timeline(offset=0, limit=1)}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _generate_detection_report(saved_result_id: str | None = None) -> dict:
    effective_saved_result_id = saved_result_id or pipeline.loaded_saved_result_id()
    timeline_path = (
        pipeline.saved_result_timeline_path(effective_saved_result_id)
        if effective_saved_result_id
        else PIPELINE_ROOT / "detect_timeline.json"
    )
    asset_url_prefix = (
        f"/api/pipeline/detect/results/{quote(effective_saved_result_id, safe='')}/jobs"
        if effective_saved_result_id
        else "/api/pipeline/detect"
    )
    result = write_detection_report_bundle(
        timeline_path,
        REPORTS_ROOT,
        asset_url_prefix=asset_url_prefix,
    )
    result["source"] = {
        "type": "saved_result" if effective_saved_result_id else "current",
        "saved_result_id": effective_saved_result_id,
    }
    return result


def _list_detection_reports() -> list[dict]:
    if not REPORTS_ROOT.exists():
        return []
    grouped: dict[str, dict] = {}
    for path in sorted(REPORTS_ROOT.glob("detection-report-*.*"), reverse=True):
        if path.suffix.lower() not in {".html", ".csv", ".json"}:
            continue
        stem = path.stem
        item = grouped.setdefault(
            stem,
            {
                "id": stem,
                "created_at": "",
                "source_summary": "-",
                "model_summary": "-",
                "confidence_summary": "-",
                "segment_count": None,
                "detection_frame_count": None,
                "video_count": None,
                "files": {},
            },
        )
        kind = path.suffix.lower().lstrip(".")
        item["files"][kind] = path.name
        stat = path.stat()
        item["mtime"] = max(float(item.get("mtime") or 0.0), stat.st_mtime)
        item["size_bytes"] = int(item.get("size_bytes") or 0) + stat.st_size
        if kind == "json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
            item["created_at"] = str(payload.get("generated_at") or "")
            item["source_summary"] = str(payload.get("source_summary") or "-")
            item["model_summary"] = str(payload.get("model_summary") or "-")
            item["confidence_summary"] = str(payload.get("confidence_summary") or "-")
            item["segment_count"] = payload.get("segment_count")
            item["detection_frame_count"] = payload.get("detection_frame_count")
            item["video_count"] = payload.get("video_count")
    reports = list(grouped.values())
    reports.sort(key=lambda item: (item.get("created_at") or "", item.get("mtime") or 0.0), reverse=True)
    return reports


def _delete_detection_report(report_id: str) -> list[str]:
    if Path(report_id).name != report_id or not report_id.startswith("detection-report-"):
        raise ValueError("Invalid report id")
    deleted: list[str] = []
    for suffix in (".html", ".csv", ".json"):
        path = REPORTS_ROOT / f"{report_id}{suffix}"
        if path.exists():
            path.unlink()
            deleted.append(path.name)
    if not deleted:
        raise FileNotFoundError("Report not found")
    return deleted


@app.get("/api/pipeline/detect/reports")
async def pipeline_detect_reports() -> dict:
    return {"reports": _list_detection_reports()}


@app.delete("/api/pipeline/detect/reports/{report_id}")
async def pipeline_detect_report_delete(report_id: str) -> dict:
    try:
        deleted = _delete_detection_report(report_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid report id")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Report not found")
    return {"deleted": deleted, "reports": _list_detection_reports()}


@app.get("/api/pipeline/detect/reports/{report_id}/delete/redirect")
async def pipeline_detect_report_delete_redirect(report_id: str) -> RedirectResponse:
    try:
        _delete_detection_report(report_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid report id")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Report not found")
    return RedirectResponse(url="/", status_code=303)


@app.post("/api/pipeline/detect/report/create")
@app.post("/api/pipeline/detect/report")
async def pipeline_detect_report(body: DetectionReportRequest | None = None) -> dict:
    try:
        return _generate_detection_report(body.saved_result_id if body else None)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/report/create")
@app.get("/api/pipeline/detect/report")
async def pipeline_detect_report_get() -> dict:
    try:
        return _generate_detection_report()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/report/{filename}")
async def pipeline_detect_report_file(filename: str) -> FileResponse:
    path = REPORTS_ROOT / filename
    if path.name != filename or path.suffix.lower() not in {".html", ".csv", ".json"}:
        raise HTTPException(status_code=400, detail="Invalid report filename")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Report not found")
    media_type = {
        ".html": "text/html; charset=utf-8",
        ".csv": "text/csv; charset=utf-8",
        ".json": "application/json",
    }[path.suffix.lower()]
    if path.suffix.lower() == ".html":
        return FileResponse(
            path,
            media_type=media_type,
            headers={"Content-Disposition": f'inline; filename="{path.name}"'},
        )
    return FileResponse(path, media_type=media_type, filename=path.name)


@app.get("/api/pipeline/detect/results/{result_id}/jobs/{job_id}/previews/{filename}")
async def pipeline_detect_saved_result_preview(result_id: str, job_id: str, filename: str) -> FileResponse:
    if Path(job_id).name != job_id or Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid saved result asset path")
    try:
        result_dir = pipeline.saved_result_timeline_path(result_id).parent
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = result_dir / "detect_jobs" / job_id / "previews" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path)


@app.get("/api/pipeline/detect/results/{result_id}/jobs/{job_id}/video")
async def pipeline_detect_saved_result_video(result_id: str, job_id: str) -> FileResponse:
    if Path(job_id).name != job_id:
        raise HTTPException(status_code=400, detail="Invalid saved result asset path")
    try:
        result_dir = pipeline.saved_result_timeline_path(result_id).parent
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    path = result_dir / "detect_jobs" / job_id / "video.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


@app.get("/api/pipeline/detect/{job_id}")
async def pipeline_get_detect_job(job_id: str) -> dict:
    try:
        return {"job": pipeline.get_detect_job(job_id).to_dict()}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/{job_id}/results")
async def pipeline_detect_results(
    job_id: str,
    detections_only: bool = False,
    offset: int = 0,
    limit: int | None = None,
) -> dict:
    try:
        return {
            "manifest": pipeline.get_detection_manifest(
                job_id,
                detections_only=detections_only,
                offset=offset,
                limit=limit,
            )
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/pipeline/detect/{job_id}/previews/{filename}")
async def pipeline_detect_preview(job_id: str, filename: str) -> FileResponse:
    if Path(filename).name != filename:
        raise HTTPException(status_code=400, detail="Invalid preview filename")
    path = pipeline._job_dir(job_id) / "previews" / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Preview not found")
    return FileResponse(path)


@app.get("/api/pipeline/detect/{job_id}/video")
async def pipeline_detect_video(job_id: str) -> FileResponse:
    try:
        path = pipeline.get_detection_video_path(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path,
        media_type="video/mp4",
        headers={"Content-Disposition": f'inline; filename="{path.name}"'},
    )


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
