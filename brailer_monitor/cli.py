"""Command-line interface for brailer monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

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
    model_path = Path(args.model) if args.model else Path(default_model)

    manifest = detect_video(
        Path(args.video),
        model_path,
        output_dir=Path(args.out),
        frame_stride=args.frame_stride,
        confidence=args.confidence,
        device=args.device,
        use_segmentation=None if args.segmentation == "auto" else args.segmentation == "yes",
        max_frames=args.max_frames,
        save_previews=not args.no_preview,
    )
    print(
        f"Frames: {manifest['frames_processed']}, "
        f"with detections: {manifest['frames_with_detections']} -> {args.out}/detections.json"
    )
    return 0


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
    detect_vid.add_argument("--confidence", type=float, default=0.35)
    detect_vid.add_argument("--device", default=0)
    detect_vid.add_argument("--segmentation", choices=["auto", "yes", "no"], default="auto")
    detect_vid.add_argument("--max-frames", type=int, default=None)
    detect_vid.add_argument("--no-preview", action="store_true")
    detect_vid.set_defaults(func=cmd_detect_video)

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
