"""Command-line interface for brailer monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from .aggregation import summarize
from .calibration import load_calibration, load_capacity_config
from .events import load_events_json, save_events_csv, save_events_json
from .frame_extractor import ExtractOptions, extract_brailer_frames, save_segment_manifest, scan_brailer_segments
from .labeling import label_extracted_frames, split_train_val, update_dataset_yaml
from .pipeline import AnalyzeOptions, analyze_video


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def cmd_analyze(args: argparse.Namespace) -> int:
    calibration = load_calibration(Path(args.calibration))
    capacity = load_capacity_config(Path(args.capacity))
    options = AnalyzeOptions(
        model_path=Path(args.model),
        frame_stride=args.frame_stride,
        tracker=args.tracker,
        device=args.device,
        max_frames=args.max_frames,
    )
    events = analyze_video(Path(args.video), calibration, capacity, options)
    out = Path(args.out)
    save_events_json(events, out)
    if args.csv:
        save_events_csv(events, Path(args.csv))
    print(f"Wrote {len(events)} events to {out}")
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    events = load_events_json(Path(args.events))
    capacity = load_capacity_config(Path(args.capacity)) if args.capacity else None
    threshold = (
        capacity.review_confidence_threshold if capacity else args.confidence_threshold
    )
    summary = summarize(events, review_confidence_threshold=threshold)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        f"Transfers: {summary.transfer_count}, "
        f"est. catch: {summary.total_weight_kg_est:.1f} kg -> {out}"
    )
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .detector import BrailerDetector

    engine = BrailerDetector.export_tensorrt(
        weights_path=Path(args.weights),
        output_dir=Path(args.output_dir),
        imgsz=args.imgsz,
        device=args.device,
    )
    print(f"Exported TensorRT engine: {engine}")
    return 0


def cmd_extract_frames(args: argparse.Namespace) -> int:
    video = Path(args.video)
    out_dir = Path(args.out)
    preview_dir = Path(args.preview) if args.preview else None
    opts = ExtractOptions(
        scan_stride=args.scan_stride,
        extract_stride=args.extract_stride,
        gap_tolerance_sec=args.gap_tolerance,
        segment_padding_sec=args.padding,
        draw_bbox_preview=preview_dir is not None,
    )

    segments, fps, _ = scan_brailer_segments(video, opts)
    print(f"Segments: {len(segments)}")
    for seg in segments:
        print(
            f"  #{seg.segment_id}: {seg.start_sec:.1f}s - {seg.end_sec:.1f}s "
            f"({seg.detection_count} detections)"
        )

    extracted, segments = extract_brailer_frames(
        video,
        out_dir,
        prefix=args.prefix,
        options=opts,
        segments=segments,
        preview_dir=preview_dir,
    )
    manifest = Path(args.manifest)
    save_segment_manifest(segments, extracted, manifest, video_path=video, fps=fps)
    print(f"Extracted {len(extracted)} frames -> {out_dir}")
    print(f"Manifest: {manifest}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from ultralytics import SAM

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    preview_dir = Path(args.preview) if args.preview else None
    sam_path = Path(args.sam_model)
    if not sam_path.exists():
        raise FileNotFoundError(f"SAM model not found: {sam_path}")

    from .frame_extractor import ExtractedFrame

    image_files = sorted(images_dir.glob("*.jpg"))
    frames = [
        ExtractedFrame(
            image_path=img,
            frame_index=0,
            timestamp_sec=0.0,
            segment_id=0,
            bbox=None,
        )
        for img in image_files
    ]

    sam_model = SAM(str(sam_path))
    records = label_extracted_frames(frames, labels_dir, sam_model, preview_dir=preview_dir)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.split:
        dataset_root = Path(args.dataset_root).resolve()
        train_n, val_n = split_train_val(images_dir, labels_dir, dataset_root, args.val_ratio, args.seed)
        project_root = dataset_root.parent.parent if dataset_root.name == "dataset" else dataset_root.parent
        update_dataset_yaml(project_root, train_n, val_n)
        print(f"Train: {train_n}, Val: {val_n}")

    print(f"Labeled {len(records)} frames -> {labels_dir}")
    return 0


def cmd_oneshot(args: argparse.Namespace) -> int:
    import cv2

    from .label_format import parse_yolo_seg_line
    from .labeling import yolo_seg_line
    from .oneshot import build_reference, detect_oneshot, load_sam_model

    ref_image = cv2.imread(str(args.ref_image))
    if ref_image is None:
        raise FileNotFoundError(f"Reference image not found: {args.ref_image}")

    label_line = Path(args.ref_label).read_text(encoding="utf-8").strip().splitlines()[0]
    seg = parse_yolo_seg_line(label_line)
    if seg is None:
        raise ValueError(f"Invalid reference label: {args.ref_label}")

    signature = build_reference([(ref_image, seg.polygon_norm)])

    sam_model = None
    sam_path = Path(args.sam_model)
    if sam_path.exists():
        sam_model = load_sam_model(str(sam_path))

    video = Path(args.video)
    out_labels = Path(args.out_labels)
    preview_dir = Path(args.preview) if args.preview else None
    out_labels.mkdir(parents=True, exist_ok=True)
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(args.interval_sec * fps)))

    detected = 0
    processed = 0
    frame_index = 0

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break

        result = detect_oneshot(frame, signature, sam_model, threshold=args.threshold)
        processed += 1
        timestamp_sec = frame_index / fps

        if result is not None:
            stem = f"oneshot_t{int(timestamp_sec * 1000):07d}_f{frame_index:05d}"
            label_path = out_labels / f"{stem}.txt"
            label_path.write_text(
                yolo_seg_line(0, result.polygon_norm) + "\n",
                encoding="utf-8",
            )

            if args.out_images:
                images_dir = Path(args.out_images)
                images_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(images_dir / f"{stem}.jpg"), frame)

            if preview_dir:
                h, w = frame.shape[:2]
                vis = frame.copy()
                pts = np.array(
                    [[int(x * w), int(y * h)] for x, y in result.polygon_norm],
                    dtype=np.int32,
                )
                cv2.polylines(vis, [pts], True, (0, 255, 0), 2)
                x1, y1, x2, y2 = result.bbox_xyxy
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 1)
                cv2.putText(
                    vis,
                    f"{result.confidence:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.imwrite(str(preview_dir / f"{stem}_preview.jpg"), vis)

            detected += 1
            print(f"  detected @ {timestamp_sec:.1f}s score={result.confidence:.3f}")

        frame_index += stride

    cap.release()
    print(f"Processed {processed} frames, detected {detected} -> {out_labels}")
    return 0


def cmd_import_cvat(args: argparse.Namespace) -> int:
    from .cvat_import import import_cvat

    video_path = Path(args.video) if args.video else None
    result = import_cvat(
        Path(args.annotations),
        Path(args.dataset_root),
        video_path=video_path,
        config_dir=Path(args.config_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        clean=not args.no_clean,
    )
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
    return 0


def cmd_detect_video(args: argparse.Namespace) -> int:
    from .train import load_task_type
    from .video_detect import detect_video

    task = load_task_type(Path(args.dataset_root))
    default_model = "models/brailer_detect.pt" if task == "detect" else "models/brailer_seg.pt"
    model_path = None if args.sea_only else Path(args.model) if args.model else Path(default_model)

    manifest = detect_video(
        Path(args.video),
        model_path,
        output_dir=Path(args.out),
        frame_stride=args.frame_stride,
        confidence=args.confidence,
        device=args.device,
        use_segmentation=None if args.segmentation == "auto" else args.segmentation == "yes",
        calculate_sea_ratio=args.sea_ratio or args.sea_only,
        sea_only=args.sea_only,
        sea_device=args.sea_device,
        sea_analysis_interval_sec=args.sea_analysis_interval_sec,
        detect_roi_margins={
            "top": args.roi_top,
            "right": args.roi_right,
            "bottom": args.roi_bottom,
            "left": args.roi_left,
        },
        max_frames=args.max_frames,
        save_previews=not args.no_preview,
    )
    print(
        f"Frames: {manifest['frames_processed']}, "
        f"with detections: {manifest['frames_with_detections']} -> {args.out}/detections.json"
    )
    return 0


def _parse_storage_hour(value: str, *, default_year: int) -> datetime:
    text = value.strip().replace(" ", "T")
    for fmt in ("%Y-%m-%dT%H", "%m-%dT%H"):
        try:
            parsed = datetime.strptime(text, fmt)
        except ValueError:
            continue
        if fmt == "%m-%dT%H":
            parsed = parsed.replace(year=default_year)
        return parsed
    raise ValueError(f"Invalid storage hour: {value} (expected YYYY-MM-DDTHH or MM-DDTHH)")


def _sea_record_emitter(args: argparse.Namespace):
    from .sea_area_scan import format_sea_sample, format_sea_summary

    output_handle = None
    if args.jsonl_out:
        output_path = Path(args.jsonl_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_handle = output_path.open("w", encoding="utf-8")

    def emit(record: dict[str, object]) -> None:
        if args.json_lines:
            line = json.dumps(record, ensure_ascii=False)
        elif record.get("type") == "summary":
            line = format_sea_summary(record)
        else:
            line = format_sea_sample(record)
        print(line, flush=True)
        if output_handle is not None:
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            output_handle.flush()

    return emit, output_handle


def _sea_storage_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    from .lake_video_source import build_lake_config_from_selection, discover_videos_in_range

    if args.max_videos is not None and args.max_videos < 1:
        raise ValueError("--max-videos must be at least 1")
    if args.download_timeout <= 0:
        raise ValueError("--download-timeout must be greater than 0")
    if args.url:
        filename = Path(urlparse(args.url).path).name or "storage-video.mp4"
        return [{"filename": filename, "url": args.url}]

    config_path = Path(args.config)
    config = build_lake_config_from_selection(
        {
            "repository": args.repository,
            "media": args.media,
            "year_folder": args.year_folder,
            "vessel": args.vessel,
            "stream": args.camera_stream,
            "minute_offsets": args.minute_offsets,
            "second_suffixes": args.second_suffixes,
        },
        path=config_path,
    )
    if not args.start or not args.end:
        raise ValueError("storage range mode requires both --start and --end")
    start = _parse_storage_hour(args.start, default_year=config.year)
    end = _parse_storage_hour(args.end, default_year=config.year)
    if start.year != config.year or end.year != config.year:
        raise ValueError(f"Storage range year must match {config.year} from --year-folder")
    if end < start:
        raise ValueError("Storage end hour is earlier than start hour")

    videos = discover_videos_in_range(
        start_month=start.month,
        start_day=start.day,
        start_hour=start.hour,
        end_month=end.month,
        end_day=end.day,
        end_hour=end.hour,
        config=config,
        check_exists=not args.no_check_exists,
    )
    if args.max_videos is not None:
        videos = videos[: args.max_videos]
    return videos


def _run_sea_storage(
    args: argparse.Namespace,
    *,
    emit,
    download_root: Path,
    analyzer,
) -> int:
    from .lake_video_source import download_video
    from .sea_area_scan import scan_recorded_sea_area

    videos = _sea_storage_candidates(args)
    if not videos:
        print("No storage videos found for the selected range.", file=sys.stderr, flush=True)
        return 1

    print(f"Storage videos found: {len(videos)}", file=sys.stderr, flush=True)
    completed = 0
    failed = 0
    total_samples = 0
    weighted_ratio = 0.0
    minimum: float | None = None
    maximum: float | None = None
    total_frames_visited = 0

    for index, video in enumerate(videos, start=1):
        filename = video["filename"]
        url = video["url"]
        local_path = download_root / filename
        try:
            if not local_path.exists() or local_path.stat().st_size <= 0:
                print(f"[{index}/{len(videos)}] downloading {filename}", file=sys.stderr, flush=True)
                download_video(url, local_path, timeout=args.download_timeout)
            else:
                print(f"[{index}/{len(videos)}] using cached {filename}", file=sys.stderr, flush=True)
            def emit_sample(sample: dict[str, object]) -> None:
                sample["source"] = url
                emit(sample)

            summary = scan_recorded_sea_area(
                local_path,
                source_name=filename,
                frame_stride=args.frame_stride,
                max_samples=args.max_samples,
                on_sample=emit_sample,
                analyzer=analyzer,
            )
            summary["source"] = url
            emit(summary)
            count = int(summary.get("samples") or 0)
            average = summary.get("avg_sea_ratio")
            min_ratio = summary.get("min_sea_ratio")
            max_ratio = summary.get("max_sea_ratio")
            total_samples += count
            total_frames_visited += int(summary.get("frames_visited") or 0)
            if average is not None:
                weighted_ratio += float(average) * count
            if min_ratio is not None:
                minimum = float(min_ratio) if minimum is None else min(minimum, float(min_ratio))
            if max_ratio is not None:
                maximum = float(max_ratio) if maximum is None else max(maximum, float(max_ratio))
            completed += 1
        except Exception as exc:
            failed += 1
            print(f"[{index}/{len(videos)}] failed {filename}: {exc}", file=sys.stderr, flush=True)

    overall_ratio = weighted_ratio / total_samples if total_samples else None
    emit(
        {
            "type": "summary",
            "source_type": "storage",
            "source": "lake-storage",
            "source_name": "storage total",
            "samples": total_samples,
            "avg_sea_ratio": round(overall_ratio, 4) if overall_ratio is not None else None,
            "avg_sea_percent": round(overall_ratio * 100.0, 2) if overall_ratio is not None else None,
            "min_sea_ratio": round(minimum, 4) if minimum is not None else None,
            "max_sea_ratio": round(maximum, 4) if maximum is not None else None,
            "frames_visited": total_frames_visited,
            "videos_completed": completed,
            "videos_failed": failed,
            "end_reason": "completed" if failed == 0 else "completed_with_errors",
        }
    )
    return 0 if completed > 0 else 1


def cmd_sea_area(args: argparse.Namespace) -> int:
    from .sea_area_analysis import SeaAreaAnalyzer
    from .sea_area_scan import scan_stream_sea_area

    emit, output_handle = _sea_record_emitter(args)
    analyzer = SeaAreaAnalyzer(device=args.device, engine=args.sea_engine)
    try:
        if args.sea_source == "stream":
            summary = scan_stream_sea_area(
                args.url,
                frame_stride=args.frame_stride,
                max_samples=args.max_samples,
                duration_sec=args.duration_sec,
                on_sample=emit,
                analyzer=analyzer,
            )
            emit(summary)
            return 0

        if args.download_dir:
            download_root = Path(args.download_dir)
            download_root.mkdir(parents=True, exist_ok=True)
            return _run_sea_storage(args, emit=emit, download_root=download_root, analyzer=analyzer)
        with tempfile.TemporaryDirectory(prefix="brailer-sea-area-") as temp_dir:
            return _run_sea_storage(args, emit=emit, download_root=Path(temp_dir), analyzer=analyzer)
    except KeyboardInterrupt:
        print("Sea-area scan stopped by user.", file=sys.stderr, flush=True)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Sea-area scan failed: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        if output_handle is not None:
            output_handle.close()


def cmd_train(args: argparse.Namespace) -> int:
    from .train import run_training

    run_training(
        dataset_yaml=Path(args.dataset),
        base_model=args.base_model or None,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project or None,
        name=args.name or None,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Brailer monitor — detect and estimate catch volume")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a recorded video and emit brailer events")
    analyze.add_argument("video", help="Path to recorded video file")
    analyze.add_argument("--calibration", default="config/calibration.json")
    analyze.add_argument("--capacity", default="config/standard_capacity.json")
    analyze.add_argument("--model", default="models/brailer_seg.engine")
    analyze.add_argument("--out", default="output/events.json")
    analyze.add_argument("--csv", help="Optional CSV output path")
    analyze.add_argument("--frame-stride", type=int, default=2)
    analyze.add_argument("--tracker", default="bytetrack.yaml")
    analyze.add_argument("--device", default=0)
    analyze.add_argument("--max-frames", type=int, default=None)
    analyze.set_defaults(func=cmd_analyze)

    extract = sub.add_parser("extract-frames", help="Detect brailer segments and extract frames")
    extract.add_argument("video", help="Path to recorded video file")
    extract.add_argument("--out", default="data/dataset/staging/images")
    extract.add_argument("--manifest", default="data/dataset/segments.json")
    extract.add_argument("--preview", default="data/dataset/staging/preview")
    extract.add_argument("--prefix", default="lake_win", help="Filename prefix for extracted frames")
    extract.add_argument("--scan-stride", type=int, default=15, help="Coarse scan interval (frames)")
    extract.add_argument("--extract-stride", type=int, default=15, help="Extract interval within segments")
    extract.add_argument("--gap-tolerance", type=float, default=3.0, help="Seconds without brailer to close segment")
    extract.add_argument("--padding", type=float, default=0.5, help="Seconds padding before/after segment")
    extract.set_defaults(func=cmd_extract_frames)

    label = sub.add_parser("label", help="Auto-label extracted frames with SAM (YOLO-seg format)")
    label.add_argument("--images", default="data/dataset/staging/images")
    label.add_argument("--labels", default="data/dataset/staging/labels")
    label.add_argument("--preview", default="data/dataset/staging/preview")
    label.add_argument("--sam-model", default="models/mobile_sam.pt")
    label.add_argument("--manifest", default="data/dataset/label_manifest.json")
    label.add_argument("--split", action="store_true", help="Split into train/val and update dataset.yaml")
    label.add_argument("--dataset-root", default="data/dataset")
    label.add_argument("--val-ratio", type=float, default=0.2)
    label.add_argument("--seed", type=int, default=42)
    label.set_defaults(func=cmd_label)

    summ = sub.add_parser("summarize", help="Aggregate brailer events into catch summary")
    summ.add_argument("events", help="Path to events JSON")
    summ.add_argument("--capacity", help="Optional capacity config for review threshold")
    summ.add_argument("--confidence-threshold", type=float, default=0.65)
    summ.add_argument("--out", default="output/summary.json")
    summ.set_defaults(func=cmd_summarize)

    export = sub.add_parser("export", help="Export YOLO weights to TensorRT engine")
    export.add_argument("weights", help="Path to .pt weights")
    export.add_argument("--output-dir", default="models")
    export.add_argument("--imgsz", type=int, default=640)
    export.add_argument("--device", type=int, default=0)
    export.set_defaults(func=cmd_export)

    import_cvat = sub.add_parser("import-cvat", help="Import CVAT 1.1 (.zip or .xml) to YOLO dataset")
    import_cvat.add_argument("annotations", help="CVAT export .zip or annotations.xml")
    import_cvat.add_argument(
        "--video",
        help="Source video file (required when annotations zip has no images)",
    )
    import_cvat.add_argument("--dataset-root", default="data/dataset")
    import_cvat.add_argument("--config-dir", default="config")
    import_cvat.add_argument("--val-ratio", type=float, default=0.2)
    import_cvat.add_argument("--seed", type=int, default=42)
    import_cvat.add_argument("--no-clean", action="store_true", help="Do not wipe existing dataset")
    import_cvat.set_defaults(func=cmd_import_cvat)

    detect_vid = sub.add_parser("detect-video", help="Run YOLO on video; per-frame detection manifest")
    detect_vid.add_argument("video", help="Video file path")
    detect_vid.add_argument("--model", default=None, help="YOLO weights (default from import meta)")
    detect_vid.add_argument("--out", default="output/detect")
    detect_vid.add_argument("--dataset-root", default="data/dataset")
    detect_vid.add_argument("--frame-stride", type=int, default=5)
    detect_vid.add_argument("--confidence", type=float, default=0.6)
    detect_vid.add_argument("--device", default=0)
    detect_vid.add_argument("--segmentation", choices=["auto", "yes", "no"], default="auto")
    detect_vid.add_argument("--sea-ratio", action="store_true", help="Calculate per-frame sea area ratio")
    detect_vid.add_argument(
        "--sea-only",
        action="store_true",
        help="Analyze sea area without loading or running an object detection model",
    )
    detect_vid.add_argument("--sea-device", default="cpu", help="Semantic sea model device (default: cpu)")
    detect_vid.add_argument(
        "--sea-analysis-interval-sec",
        type=float,
        default=5.0,
        help="Sea analysis interval in seconds; 0 means every processed frame (max: 300)",
    )
    detect_vid.add_argument("--roi-top", type=float, default=0.0)
    detect_vid.add_argument("--roi-right", type=float, default=0.15)
    detect_vid.add_argument("--roi-bottom", type=float, default=0.0)
    detect_vid.add_argument("--roi-left", type=float, default=0.15)
    detect_vid.add_argument("--max-frames", type=int, default=None)
    detect_vid.add_argument("--no-preview", action="store_true")
    detect_vid.set_defaults(func=cmd_detect_video)

    sea_area = sub.add_parser("sea-area", help="Measure visible sea area from Lake storage or a live stream")
    sea_sources = sea_area.add_subparsers(dest="sea_source", required=True)

    sea_storage = sea_sources.add_parser("storage", help="Scan videos from Lake storage")
    sea_storage.add_argument("--url", help="Single storage video URL; bypass Lake range discovery")
    sea_storage.add_argument("--start", help="Range start hour (YYYY-MM-DDTHH or MM-DDTHH)")
    sea_storage.add_argument("--end", help="Range end hour (YYYY-MM-DDTHH or MM-DDTHH)")
    sea_storage.add_argument("--repository", default=None, help="Storage repository profile")
    sea_storage.add_argument("--media", default=None, help="Lake media folder, e.g. lake_win")
    sea_storage.add_argument("--year-folder", default=None, help="Lake year folder, e.g. 2026_decrypted")
    sea_storage.add_argument("--vessel", default=None, help="Vessel name, e.g. JJR-102283")
    sea_storage.add_argument("--camera-stream", default=None, help="Stored camera stream, e.g. stream04")
    sea_storage.add_argument("--minute-offsets", default=None, help="Comma-separated start minute offsets, e.g. 0,1,2,3,4")
    sea_storage.add_argument("--second-suffixes", default=None, help="Comma-separated second suffixes, e.g. 16")
    sea_storage.add_argument("--config", default="config/lake_video.json")
    sea_storage.add_argument("--no-check-exists", action="store_true", help="Do not probe candidate URLs before scanning")
    sea_storage.add_argument("--max-videos", type=int, default=None, help="Maximum number of discovered videos")
    sea_storage.add_argument("--download-dir", default=None, help="Keep or reuse downloaded storage videos here")
    sea_storage.add_argument("--download-timeout", type=float, default=120.0)

    sea_stream = sea_sources.add_parser("stream", help="Scan a live HLS/HTTP stream")
    sea_stream.add_argument("--url", default="http://127.0.0.1:8081/live_04.m3u8")
    sea_stream.add_argument("--duration-sec", type=float, default=None, help="Stop after this many seconds")

    for sea_parser in (sea_storage, sea_stream):
        sea_parser.add_argument("--frame-stride", type=int, default=30, help="Calculate sea area every N frames")
        sea_parser.add_argument("--max-samples", type=int, default=None, help="Maximum sampled frames per source")
        sea_parser.add_argument("--sea-engine", choices=["hybrid", "legacy"], default="hybrid")
        sea_parser.add_argument("--device", default="cpu", help="Semantic model device (default: cpu)")
        sea_parser.add_argument("--json-lines", action="store_true", help="Print frame records as JSON Lines")
        sea_parser.add_argument("--jsonl-out", default=None, help="Also write frame and summary records to a JSONL file")
        sea_parser.set_defaults(func=cmd_sea_area)

    train = sub.add_parser("train", help="Train YOLO from config/dataset.yaml (after import-cvat)")
    train.add_argument("--dataset", default="config/dataset.yaml")
    train.add_argument("--base-model", default=None, help="Auto from import meta if omitted")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default=0)
    train.add_argument("--project", default=None)
    train.add_argument("--name", default=None)
    train.set_defaults(func=cmd_train)

    oneshot = sub.add_parser("oneshot", help="One-shot brailer detection from reference label (no training)")
    oneshot.add_argument("--ref-image", required=True, help="Reference frame image")
    oneshot.add_argument("--ref-label", required=True, help="Reference YOLO-seg label (.txt)")
    oneshot.add_argument("--video", required=True, help="Video to scan")
    oneshot.add_argument("--out-labels", default="output/oneshot/labels")
    oneshot.add_argument("--out-images", default="output/oneshot/images", help="Save detected frame images")
    oneshot.add_argument("--preview", default="output/oneshot/preview", help="Preview overlays")
    oneshot.add_argument("--sam-model", default="models/mobile_sam.pt")
    oneshot.add_argument("--interval-sec", type=float, default=5.0, help="Sample interval in seconds")
    oneshot.add_argument("--threshold", type=float, default=0.55, help="Detection confidence threshold")
    oneshot.set_defaults(func=cmd_oneshot)

    web = sub.add_parser("web", help="Start web viewer for upload and label review")
    web.add_argument("--host", default="0.0.0.0")
    web.add_argument("--port", type=int, default=8080)
    web.add_argument("--reload", action="store_true")
    web.set_defaults(func=cmd_web)

    return parser


def cmd_web(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "brailer_monitor.web.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
