"""Run a single video detection in an isolated subprocess.

A CUDA "illegal memory access" (cudaErrorIllegalAddress) permanently corrupts
the CUDA context of the *whole* process: every later CUDA call in that process
keeps failing. The web server runs detection in background threads, so such an
error would otherwise wedge the server until it is restarted manually.

To stay recoverable, the pipeline launches this module as a separate process
for each detection job. If the GPU dies here, only this process dies; the server
keeps a clean CUDA context and the next job simply spawns a fresh worker.

Progress is reported by atomically writing a small JSON file that the parent
polls. Cancellation is handled by the parent terminating this process.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import traceback
from pathlib import Path


def _is_cuda_oom_error(message: str) -> bool:
    lowered = message.lower()
    return "cuda" in lowered and (
        "out of memory" in lowered
        or "memoryallocation" in lowered
        or "cudaerrormemoryallocation" in lowered
    )


def _is_cpu_device(device: str | int) -> bool:
    return isinstance(device, str) and device.lower() == "cpu"


def _write_progress(
    progress_file: Path,
    processed: int,
    total: int,
    with_det: int,
    sea_stats: dict | None = None,
) -> None:
    tmp = progress_file.with_suffix(progress_file.suffix + ".tmp")
    try:
        payload = {"processed": processed, "total": total, "with": with_det}
        if sea_stats:
            payload.update(sea_stats)
        tmp.write_text(
            json.dumps(payload),
            encoding="utf-8",
        )
        os.replace(tmp, progress_file)
    except Exception:
        # Progress reporting is best-effort; never let it crash detection.
        pass


def _append_event(events_file: Path, frame: dict) -> None:
    try:
        with events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(frame, ensure_ascii=False))
            handle.write("\n")
    except Exception:
        # Incremental timeline reporting is best-effort; the final manifest is
        # still written and merged when the worker completes.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated single-video detection worker")
    parser.add_argument("--video")
    parser.add_argument("--stream-url")
    parser.add_argument("--model", action="append")
    parser.add_argument("--model-specs-file")
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="0")
    parser.add_argument("--segmentation", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--sam", choices=["yes", "no"], default="yes")
    parser.add_argument("--sea-ratio", choices=["yes", "no"], default="no")
    parser.add_argument("--sea-only", choices=["yes", "no"], default="no")
    parser.add_argument("--sea-engine", choices=["hybrid", "legacy"], default="hybrid")
    parser.add_argument("--sea-device", default="cpu")
    parser.add_argument("--sea-analysis-interval-sec", type=float, default=5.0)
    parser.add_argument("--sea-state-file")
    parser.add_argument("--roi-top", type=float, default=0.15)
    parser.add_argument("--roi-right", type=float, default=0.15)
    parser.add_argument("--roi-bottom", type=float, default=0.15)
    parser.add_argument("--roi-left", type=float, default=0.15)
    parser.add_argument("--skip-dark-video", choices=["yes", "no"], default="no")
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--events-file")
    parser.add_argument("--stop-file")
    args = parser.parse_args(argv)
    if not args.video and not args.stream_url:
        parser.error("--video or --stream-url is required")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_file = output_dir / "worker_error.txt"
    progress_file = Path(args.progress_file)

    device: str | int = args.device
    if isinstance(device, str) and device.lstrip("-").isdigit():
        device = int(device)

    use_segmentation = None if args.segmentation == "auto" else args.segmentation == "yes"
    use_sam = args.sam == "yes"
    calculate_sea_ratio = args.sea_ratio == "yes"
    sea_only = args.sea_only == "yes"
    if sea_only:
        calculate_sea_ratio = True
        use_sam = False
    sea_state_path = Path(args.sea_state_file) if args.sea_state_file else None
    sea_state_snapshot = sea_state_path.read_bytes() if sea_state_path is not None and sea_state_path.exists() else None
    skip_dark_video = args.skip_dark_video == "yes"
    detect_roi = {
        "top": args.roi_top,
        "right": args.roi_right,
        "bottom": args.roi_bottom,
        "left": args.roi_left,
    }
    model_specs = None
    if args.model_specs_file:
        model_specs = json.loads(Path(args.model_specs_file).read_text(encoding="utf-8"))
    model_paths = [Path(path) for path in (args.model or [])]
    if not sea_only and not model_specs and not model_paths:
        parser.error("--model or --model-specs-file is required")
    primary_model = (
        model_paths[0]
        if model_paths
        else Path(str(model_specs[0]["path"])) if model_specs else None
    )

    def on_progress(
        processed: int,
        total: int,
        with_det: int,
        sea_stats: dict | None = None,
    ) -> None:
        _write_progress(progress_file, processed, total, with_det, sea_stats)

    events_file = Path(args.events_file) if args.events_file else None

    def on_detection(frame: dict) -> None:
        if events_file is not None:
            _append_event(events_file, frame)

    stop_file = Path(args.stop_file) if args.stop_file else None

    def should_cancel() -> bool:
        return stop_file is not None and stop_file.exists()

    def _handle_signal(signum: int, frame: object) -> None:
        if stop_file is not None:
            try:
                stop_file.write_text("stop", encoding="utf-8")
            except Exception:
                pass
            return
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        from ..video_detect import detect_stream, detect_video

        if args.stream_url:
            detect_stream(
                args.stream_url,
                primary_model,
                output_dir=output_dir,
                model_specs=model_specs,
                frame_stride=args.frame_stride,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device=device,
                use_segmentation=use_segmentation,
                use_sam=use_sam,
                calculate_sea_ratio=calculate_sea_ratio,
                sea_only=sea_only,
                sea_engine=args.sea_engine,
                sea_device=args.sea_device,
                sea_analysis_interval_sec=args.sea_analysis_interval_sec,
                sea_state_path=sea_state_path,
                detect_roi_margins=detect_roi,
                on_progress=on_progress,
                on_detection=on_detection,
                should_cancel=should_cancel,
            )
            return 0

        try:
            detect_video(
                Path(args.video),
                primary_model,
                output_dir=output_dir,
                model_specs=model_specs,
                frame_stride=args.frame_stride,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device=device,
                use_segmentation=use_segmentation,
                use_sam=use_sam,
                calculate_sea_ratio=calculate_sea_ratio,
                sea_only=sea_only,
                sea_engine=args.sea_engine,
                sea_device=args.sea_device,
                sea_analysis_interval_sec=args.sea_analysis_interval_sec,
                sea_state_path=sea_state_path,
                detect_roi_margins=detect_roi,
                skip_dark_video=skip_dark_video,
                on_progress=on_progress,
                on_detection=on_detection,
                should_cancel=should_cancel,
            )
        except Exception as exc:
            if _is_cpu_device(device) or not _is_cuda_oom_error(str(exc)):
                raise
            print(
                "CUDA out of memory; retrying this video on CPU.",
                file=sys.stderr,
                flush=True,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            if sea_state_path is not None:
                if sea_state_snapshot is None:
                    sea_state_path.unlink(missing_ok=True)
                else:
                    sea_state_path.parent.mkdir(parents=True, exist_ok=True)
                    sea_state_path.write_bytes(sea_state_snapshot)
            detect_video(
                Path(args.video),
                primary_model,
                output_dir=output_dir,
                model_specs=model_specs,
                frame_stride=args.frame_stride,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device="cpu",
                use_segmentation=use_segmentation,
                use_sam=use_sam,
                calculate_sea_ratio=calculate_sea_ratio,
                sea_only=sea_only,
                sea_engine=args.sea_engine,
                sea_device=args.sea_device,
                sea_analysis_interval_sec=args.sea_analysis_interval_sec,
                sea_state_path=sea_state_path,
                detect_roi_margins=detect_roi,
                skip_dark_video=skip_dark_video,
                on_progress=on_progress,
                on_detection=on_detection,
                should_cancel=should_cancel,
            )
        return 0
    except Exception as exc:  # noqa: BLE001 - report any failure to the parent
        message = f"{type(exc).__name__}: {exc}"
        try:
            error_file.write_text(
                message + "\n\n" + traceback.format_exc(),
                encoding="utf-8",
            )
        except Exception:
            pass
        print(message, file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
