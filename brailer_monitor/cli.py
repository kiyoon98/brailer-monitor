"""Command-line interface for brailer monitoring."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .aggregation import summarize
from .calibration import load_calibration, load_capacity_config
from .events import load_events_json, save_events_csv, save_events_json
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


def cmd_train(args: argparse.Namespace) -> int:
    from .train import run_training

    run_training(
        dataset_yaml=Path(args.dataset),
        base_model=args.base_model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
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

    train = sub.add_parser("train", help="Train a custom YOLO11-seg model")
    train.add_argument("--dataset", default="config/dataset.yaml")
    train.add_argument("--base-model", default="yolo11n-seg.pt")
    train.add_argument("--epochs", type=int, default=100)
    train.add_argument("--imgsz", type=int, default=640)
    train.add_argument("--batch", type=int, default=8)
    train.add_argument("--device", default=0)
    train.add_argument("--project", default="runs/segment")
    train.add_argument("--name", default="brailer_seg")
    train.set_defaults(func=cmd_train)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _configure_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
